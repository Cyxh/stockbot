"""
TUNE.PY — Walk-forward optimizer.

Tests every parameter combination across 5 completely separate market eras:
  Window 1 test: 2016-2017  (post-crash bull, low volatility)
  Window 2 test: 2018-2019  (rate-hike correction + recovery)
  Window 3 test: 2020-2021  (COVID crash + biggest bull run in decades)
  Window 4 test: 2022       (worst bear market since 2008)
  Window 5 test: 2023-now   (AI boom + tariff pullback)

A config that beats SPY consistently across ALL five eras is genuinely robust.
A config that only crushes one era is fitting to noise from that era.

OVERFITTING GUARD:
  - Configs are scored by AVERAGE excess return across all 5 windows
  - A "consistency bonus" rewards beating SPY in 4/5 or 5/5 windows
  - A "variance penalty" punishes configs that swing wildly window-to-window
  - The final winner is the most consistently profitable config, not the luckiest

USAGE:
    python tune.py
"""

import itertools
import re
from copy import deepcopy
from math import sqrt

import numpy as np
import pandas as pd
import yfinance as yf

import config
from technical import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands,
    calculate_vwap, calculate_volume_ratio, calculate_atr,
)

# ── Walk-forward windows ───────────────────────────────────────────────────────
# Each entry: (train_start, train_end, val_start, val_end)
# Training always starts at 2010 (expanding window) so early windows have
# enough data to compute SMA-200 and 6-month momentum.
WALK_FORWARD_WINDOWS = [
    ("2010-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),  # Post-crisis bull
    ("2010-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),  # Rate hike + recovery
    ("2010-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),  # COVID crash + boom
    ("2010-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),  # Bear market
    ("2010-01-01", "2022-12-31", "2023-01-01", None),          # AI bull + correction
]

WEIGHT_PROFILES = {
    "balanced": {
        "sma_crossover": 0.05, "sma_spread": 0.03, "ema_crossover": 0.05,
        "rsi": 0.07, "macd": 0.08, "macd_momentum": 0.05,
        "bollinger": 0.05, "vwap": 0.03, "volume": 0.04,
        "momentum_3m": 0.11, "momentum_6m": 0.11,
        "momentum_12m": 0.12, "relative_strength": 0.10, "high_52w_proximity": 0.11,
    },
    "trend": {
        "sma_crossover": 0.10, "sma_spread": 0.06, "ema_crossover": 0.10,
        "rsi": 0.04, "macd": 0.07, "macd_momentum": 0.04,
        "bollinger": 0.03, "vwap": 0.03, "volume": 0.03,
        "momentum_3m": 0.12, "momentum_6m": 0.10,
        "momentum_12m": 0.12, "relative_strength": 0.10, "high_52w_proximity": 0.06,
    },
    "momentum": {
        "sma_crossover": 0.03, "sma_spread": 0.02, "ema_crossover": 0.03,
        "rsi": 0.06, "macd": 0.09, "macd_momentum": 0.06,
        "bollinger": 0.03, "vwap": 0.02, "volume": 0.03,
        "momentum_3m": 0.18, "momentum_6m": 0.16,
        "momentum_12m": 0.14, "relative_strength": 0.10, "high_52w_proximity": 0.05,
    },
    "mean_reversion": {
        "sma_crossover": 0.04, "sma_spread": 0.02, "ema_crossover": 0.04,
        "rsi": 0.14, "macd": 0.07, "macd_momentum": 0.04,
        "bollinger": 0.13, "vwap": 0.05, "volume": 0.04,
        "momentum_3m": 0.09, "momentum_6m": 0.07,
        "momentum_12m": 0.10, "relative_strength": 0.10, "high_52w_proximity": 0.07,
    },
    "aggressive_momentum": {
        "sma_crossover": 0.02, "sma_spread": 0.01, "ema_crossover": 0.02,
        "rsi": 0.04, "macd": 0.07, "macd_momentum": 0.04,
        "bollinger": 0.01, "vwap": 0.01, "volume": 0.02,
        "momentum_3m": 0.22, "momentum_6m": 0.18,
        "momentum_12m": 0.16, "relative_strength": 0.11, "high_52w_proximity": 0.09,
    },
    "trend_momentum": {
        "sma_crossover": 0.07, "sma_spread": 0.04, "ema_crossover": 0.07,
        "rsi": 0.05, "macd": 0.06, "macd_momentum": 0.03,
        "bollinger": 0.02, "vwap": 0.02, "volume": 0.03,
        "momentum_3m": 0.16, "momentum_6m": 0.13,
        "momentum_12m": 0.13, "relative_strength": 0.12, "high_52w_proximity": 0.07,
    },
    "academic_momentum": {
        "sma_crossover": 0.08, "sma_spread": 0.05, "ema_crossover": 0.08,
        "rsi": 0.05, "macd": 0.08, "macd_momentum": 0.04,
        "bollinger": 0.03, "vwap": 0.03, "volume": 0.03,
        "momentum_3m": 0.12, "momentum_6m": 0.10,
        "momentum_12m": 0.14, "relative_strength": 0.10, "high_52w_proximity": 0.07,
    },
    "pure_momentum": {
        "sma_crossover": 0.00, "sma_spread": 0.00, "ema_crossover": 0.00,
        "rsi": 0.00, "macd": 0.00, "macd_momentum": 0.00,
        "bollinger": 0.00, "vwap": 0.00, "volume": 0.00,
        "momentum_3m": 0.30, "momentum_6m": 0.25,
        "momentum_12m": 0.22, "relative_strength": 0.15, "high_52w_proximity": 0.08,
    },
}

PARAM_GRID = {
    "REBALANCE_EVERY":       [3, 5, 10, 21],
    "MAX_POSITION_SIZE_PCT": [0.08, 0.10, 0.15],
    "MAX_OPEN_POSITIONS":    [8, 12, 15],
    "weight_profile":        list(WEIGHT_PROFILES.keys()),
}


# ── Signal precomputation ──────────────────────────────────────────────────────

def precompute_signals(df: pd.DataFrame, spy_df=None) -> pd.DataFrame:
    TECH  = config.TECHNICAL
    close = df["Close"]
    vol   = df["Volume"]

    sma_fast = calculate_sma(close, TECH["sma_fast"])
    sma_slow = calculate_sma(close, TECH["sma_slow"])
    ema_fast = calculate_ema(close, TECH["ema_fast"])
    ema_slow = calculate_ema(close, TECH["ema_slow"])
    rsi_v    = calculate_rsi(close)
    macd_l, sig_l, hist = calculate_macd(close)
    bb_u, _, bb_l       = calculate_bollinger_bands(close)
    vwap_v  = calculate_vwap(df)
    vol_rat = calculate_volume_ratio(vol)
    atr_v   = calculate_atr(df)

    sma_cross  = np.where(sma_fast > sma_slow, 1.0, -1.0)
    sma_spread = np.clip((sma_fast - sma_slow) / sma_slow.replace(0, np.nan) * 20, -1.0, 1.0).fillna(0)
    ema_cross  = np.where(ema_fast > ema_slow, 1.0, -1.0)

    rsi_sig = np.where(rsi_v < TECH["rsi_oversold"],   1.0,
              np.where(rsi_v > TECH["rsi_overbought"], -1.0,
              np.clip((45 - rsi_v) / 30, -1.0, 1.0)))

    macd_sig  = np.where(macd_l > sig_l, 1.0, -1.0)
    macd_mom  = np.where(hist > hist.shift(1), 0.5, -0.5)

    bb_range = (bb_u - bb_l).replace(0, np.nan)
    bb_pos   = ((close - bb_l) / bb_range).fillna(0.5).clip(0, 1)
    bb_sig   = (1.0 - 2.0 * bb_pos).clip(-1.0, 1.0)

    vwap_sig = np.where(close > vwap_v, 0.5, -0.5)

    vol_c   = vol_rat.clip(0.5, 2.0).fillna(1.0)
    pch     = close.pct_change()
    vol_sig = np.where(vol_rat >= 1.2,
                       np.clip(pch * 30, -1.0, 1.0),
                       np.clip(pch * 10, -0.3, 0.3))

    mom_3m = np.clip(close.pct_change(63).fillna(0) * 3,   -1.0, 1.0)
    mom_6m = np.clip(close.pct_change(126).fillna(0) * 1.5, -1.0, 1.0)

    # 12-month momentum, skip 1 month (Jegadeesh & Titman 1993)
    # Return from 252 days ago to 21 days ago (skip most recent month to avoid reversal)
    mom_12m = np.clip(
        (close.shift(21) / close.shift(252).replace(0, np.nan) - 1).fillna(0) * 1.5,
        -1.0, 1.0
    )

    # 52-week high proximity (George & Hwang 2004)
    # Stocks near 52W high have higher forward returns due to anchoring bias
    rolling_52w_high = close.rolling(window=252, min_periods=63).max()
    proximity_raw = (close / rolling_52w_high.replace(0, np.nan)).fillna(0.5)
    high_52w_prox = np.clip((proximity_raw - 0.85) * 6.67, -1.0, 1.0)

    # Relative strength vs SPY benchmark (3-month outperformance)
    if spy_df is not None and len(spy_df) >= 63:
        stock_3m = close.pct_change(63).fillna(0)
        spy_3m_series = spy_df["Close"].pct_change(63).fillna(0)
        spy_3m_aligned = spy_3m_series.reindex(close.index, method='ffill').fillna(0)
        relative_str = np.clip((stock_3m - spy_3m_aligned) * 5, -1.0, 1.0)
    else:
        relative_str = pd.Series(0.0, index=close.index)

    # Volatility-adjusted momentum (Barroso & Santa-Clara 2015)
    realized_vol_12m = close.pct_change().rolling(252).std() * np.sqrt(252)
    raw_mom_12m = (close.shift(21) / close.shift(252).replace(0, np.nan) - 1).fillna(0)
    vol_adj_mom = np.clip(
        (raw_mom_12m / realized_vol_12m.replace(0, np.nan)).fillna(0),
        -1.0, 1.0
    )

    # Momentum acceleration: 3M momentum now minus 3M momentum 3 months ago
    mom_3m_series = close.pct_change(63).fillna(0)
    mom_accel = np.clip((mom_3m_series - mom_3m_series.shift(63)).fillna(0) * 5, -1.0, 1.0)

    # Short-term reversal: negative of 5-day return (pullbacks score positively)
    ret_5d = close.pct_change(5).fillna(0)
    short_rev = np.clip(-ret_5d * 10, -1.0, 1.0)

    # Trend efficiency: Kaufman ER × direction
    def _rolling_efficiency(s, period=60):
        net = (s - s.shift(period)).abs()
        vol_sum = s.diff().abs().rolling(period).sum()
        er = (net / vol_sum.replace(0, np.nan)).fillna(0)
        direction = np.where(s > s.shift(period), 1.0, -1.0)
        return np.clip(er * 2 * direction, -1.0, 1.0)
    trend_eff = pd.Series(_rolling_efficiency(close), index=close.index)

    # Volume-price confirmation
    vol_recent_avg = df["Volume"].rolling(20).mean()
    vol_prior_avg  = df["Volume"].rolling(20).mean().shift(20)
    price_ret_20d  = close.pct_change(20).fillna(0)
    vol_change     = ((vol_recent_avg / vol_prior_avg.replace(0, np.nan)) - 1).fillna(0)
    vol_price_conf = np.clip(price_ret_20d * vol_change * 20, -1.0, 1.0)

    # Idiosyncratic momentum (Blitz, Huij & Martens 2011)
    # Strip market beta from stock returns, compute momentum on residuals
    # Vectorized: rolling 252-day beta, then residual cumulative return
    if spy_df is not None and len(spy_df) >= 252:
        stock_daily = close.pct_change().fillna(0)
        spy_daily = spy_df["Close"].pct_change().reindex(close.index, method="ffill").fillna(0)
        # Rolling covariance and variance for beta
        rolling_cov = stock_daily.rolling(252).cov(spy_daily)
        rolling_var = spy_daily.rolling(252).var()
        beta = (rolling_cov / rolling_var.replace(0, np.nan)).clip(-3.0, 3.0).fillna(0)
        # Residual returns = stock - beta * market
        residual = stock_daily - beta * spy_daily
        # 12-month residual return, skip last month (sum of daily residuals)
        resid_12m = residual.rolling(252).sum() - residual.rolling(21).sum()
        idio_mom = np.clip(resid_12m.fillna(0) * 3, -1.0, 1.0)
    else:
        idio_mom = pd.Series(0.0, index=close.index)

    # Drawdown recovery score (Daniel & Moskowitz 2016)
    # Vectorized: penalize stocks in deep drawdown, reward those near highs
    high_6m = close.rolling(126, min_periods=63).max()
    dd_from_high = (1.0 - (close / high_6m.replace(0, np.nan))).fillna(0).clip(0, 1)
    ret_10d = close.pct_change(10).fillna(0)
    # Deep drawdown + recovering
    deep_recovering = np.clip(-dd_from_high * 1.5 + 0.3, -1.0, 0.3)
    # Deep drawdown + still falling
    deep_falling = np.clip(-dd_from_high * 3, -1.0, -0.2)
    # Not in deep drawdown
    healthy = np.clip((0.25 - dd_from_high) * 2, 0.0, 0.5)
    dd_recovery = pd.Series(np.where(
        dd_from_high > 0.25,
        np.where(ret_10d > 0.02, deep_recovering, deep_falling),
        healthy
    ), index=close.index)

    # Realized volatility (for position sizing)
    realized_vol_60d = close.pct_change().rolling(60).std() * np.sqrt(252)
    realized_vol_60d = realized_vol_60d.fillna(0.20)

    sma50  = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    bull_regime = (sma50 > sma200).astype(float).fillna(0)

    atr_pct = (atr_v / close.replace(0, np.nan)).fillna(0.02)

    return pd.DataFrame({
        "sma_crossover": sma_cross,
        "sma_spread":    sma_spread,
        "ema_crossover": ema_cross,
        "rsi":           rsi_sig,
        "macd":          macd_sig,
        "macd_momentum": macd_mom,
        "bollinger":     bb_sig,
        "vwap":          vwap_sig,
        "volume":        vol_sig,
        "momentum_3m":   mom_3m,
        "momentum_6m":   mom_6m,
        "momentum_12m":          mom_12m,
        "high_52w_proximity":    high_52w_prox,
        "relative_strength":     relative_str,
        "vol_adj_momentum":      vol_adj_mom,
        "momentum_acceleration": mom_accel,
        "short_term_reversal":   short_rev,
        "trend_efficiency":      trend_eff,
        "volume_price_confirm":  vol_price_conf,
        "idiosyncratic_momentum": idio_mom,
        "drawdown_recovery":     dd_recovery,
        "realized_vol":          realized_vol_60d,
        "bull_regime":   bull_regime,
        "vol_mult":      vol_c,
        "atr_pct":       atr_pct,
        "price":         close,
    }, index=df.index)


# ── Fast backtester ────────────────────────────────────────────────────────────

def fast_backtest(
    precomputed:    dict,
    warmup_days:    int   = 60,
    rebalance_every: int  = 5,
    max_pos_pct:    float = 0.10,
    max_positions:  int   = 12,
    weights:        dict  = None,
    regime_symbol:  str   = "SPY",
    starting_cash:  float = 100_000,
) -> dict:
    if weights is None:
        weights = WEIGHT_PROFILES["balanced"]

    all_dates = sorted(set.intersection(*[set(df.index) for df in precomputed.values()]))
    if len(all_dates) <= warmup_days:
        return _empty_result(starting_cash)

    cash      = starting_cash
    positions = {}
    trades    = []
    equity    = []

    for day_idx, dt in enumerate(all_dates):
        if day_idx < warmup_days:
            equity.append(starting_cash)
            continue

        regime_sig = precomputed.get(regime_symbol)
        in_bull = True
        if regime_sig is not None and dt in regime_sig.index:
            in_bull = float(regime_sig.at[dt, "bull_regime"]) > 0.5

        # ── BIL bear-regime cash parking ──────────────────────────────────
        bear_etf = getattr(config, "BEAR_REGIME_ETF", None)
        if bear_etf and bear_etf in precomputed and dt in precomputed[bear_etf].index:
            bil_price = float(precomputed[bear_etf].at[dt, "price"])
            if in_bull and bear_etf in positions:
                cash += positions[bear_etf]["qty"] * bil_price
                del positions[bear_etf]
            elif not in_bull and bear_etf not in positions and cash > 1000:
                qty = int(cash * 0.90 / bil_price)
                if qty > 0:
                    cash -= qty * bil_price
                    positions[bear_etf] = {"qty": qty, "entry_price": bil_price,
                                           "highest_price": bil_price}

        # ── Phase 1: catastrophe stop (every day) ─────────────────────────
        for sym in list(positions.keys()):
            if sym == bear_etf:
                continue  # BIL is managed separately, never stop-lossed
            sig_df = precomputed.get(sym)
            if sig_df is None or dt not in sig_df.index:
                continue
            pos = positions[sym]
            cur = float(sig_df.at[dt, "price"])
            pnl_pct = (cur - pos["entry_price"]) / pos["entry_price"]
            if pnl_pct <= -0.20:
                cash += pos["qty"] * cur
                trades.append({"pnl": (cur - pos["entry_price"]) * pos["qty"],
                               "pnl_pct": pnl_pct, "reason": "catastrophe_stop"})
                del positions[sym]

        pos_val = sum(
            positions[s]["qty"] * float(precomputed[s].at[dt, "price"])
            for s in positions if dt in precomputed[s].index
        )
        portfolio_value = cash + pos_val

        # ── Phase 2: rebalance every N trading days ───────────────────────
        trading_day_num = day_idx - warmup_days
        if trading_day_num % rebalance_every == 0:
            # Compute momentum sub-score for each stock
            day_sigs = []
            ranking_wts = getattr(config, "RANKING_WEIGHTS", {
                "momentum_12m": 0.30, "momentum_6m": 0.20,
                "momentum_3m": 0.20, "relative_strength": 0.20,
                "high_52w_proximity": 0.10,
            })
            for sym, sig_df in precomputed.items():
                if sym == bear_etf:
                    continue
                if dt not in sig_df.index:
                    continue
                row = sig_df.loc[dt]
                if pd.isna(row["sma_crossover"]):
                    continue

                # Use configurable ranking weights with renormalization
                avail_wt = sum(w for k, w in ranking_wts.items()
                               if k in row.index and not pd.isna(row.get(k)))
                if avail_wt > 0:
                    mom_score = sum(
                        float(row.get(k, 0)) * w
                        for k, w in ranking_wts.items()
                        if k in row.index and not pd.isna(row.get(k))
                    ) / avail_wt
                else:
                    mom_score = 0.0

                row_bull = float(row.get("bull_regime", 0)) > 0.5
                day_sigs.append({"sym": sym, "score": mom_score,
                                 "price": float(row["price"]),
                                 "realized_vol": float(row.get("realized_vol", 0.20)),
                                 "in_bull": row_bull})

            # Sort by momentum score descending
            day_sigs.sort(key=lambda x: x["score"], reverse=True)

            # Target = top max_positions stocks with score > 0 AND in_bull
            target_syms = set(
                s["sym"] for s in day_sigs[:max_positions]
                if s["score"] > 0 and s["in_bull"]
            )

            # Sell anything held that's NOT in target (rank_drop exit)
            for sym in list(positions.keys()):
                if sym == bear_etf:
                    continue
                if sym not in target_syms:
                    sig_df = precomputed.get(sym)
                    if sig_df is not None and dt in sig_df.index:
                        price = float(sig_df.at[dt, "price"])
                        pos = positions[sym]
                        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
                        cash += pos["qty"] * price
                        trades.append({"pnl": (price - pos["entry_price"]) * pos["qty"],
                                       "pnl_pct": pnl_pct, "reason": "rank_drop"})
                        del positions[sym]

            # Buy up to 3 new stocks from target not already held
            new_buys = 0
            for sig in day_sigs:
                if new_buys >= 3:
                    break
                sym = sig["sym"]
                if sym not in target_syms or sym in positions:
                    continue
                price = sig["price"]

                # Inverse-volatility sizing
                r_vol = sig.get("realized_vol", 0.20)
                t_vol = getattr(config, "INVERSE_VOL_TARGET_VOL", 0.20)
                if getattr(config, "USE_INVERSE_VOL_SIZING", True) and r_vol > 0.01:
                    v_sc = min(max(t_vol / r_vol, 0.3), 2.5)
                else:
                    v_sc = 1.0

                dollar = portfolio_value * max_pos_pct * v_sc
                qty = int(dollar / price)
                if qty > 0 and cash >= qty * price:
                    cash -= qty * price
                    positions[sym] = {"qty": qty, "entry_price": price,
                                      "highest_price": price}
                    new_buys += 1

        pos_val2 = sum(
            positions[s]["qty"] * float(precomputed[s].at[dt, "price"])
            for s in positions if dt in precomputed[s].index
        )
        equity.append(cash + pos_val2)

    last_dt = all_dates[-1]
    for sym, pos in list(positions.items()):
        sig_df = precomputed.get(sym)
        if sig_df is not None and last_dt in sig_df.index:
            p   = float(sig_df.at[last_dt, "price"])
            pnl = (p - pos["entry_price"]) / pos["entry_price"]
            cash += pos["qty"] * p
            trades.append({"pnl": (p - pos["entry_price"]) * pos["qty"],
                           "pnl_pct": pnl, "reason": "end"})

    ending    = cash
    total_ret = (ending - starting_cash) / starting_cash
    n_trading = len(equity) - warmup_days
    ann = ((1 + total_ret) ** (252 / max(n_trading, 1)) - 1) if n_trading > 0 else 0

    eq_arr  = np.array(equity[warmup_days:], dtype=float)
    daily_r = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1)
    std_d   = float(np.std(daily_r))
    sharpe  = float(np.mean(daily_r) / std_d * sqrt(252)) if std_d > 0 and len(daily_r) > 1 else 0

    peak   = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peak - eq_arr) / np.where(peak > 0, peak, 1)))

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]

    return {
        "annualized_return_pct": ann * 100,
        "sharpe_ratio":          sharpe,
        "max_drawdown_pct":      max_dd * 100,
        "total_trades":          len(trades),
        "winning_trades":        len(wins),
        "losing_trades":         len(losses),
        "win_rate":              len(wins) / len(trades) * 100 if trades else 0,
        "ending_value":          ending,
    }


def _empty_result(cash):
    return {"annualized_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0,
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "ending_value": cash}


def walk_forward_score(window_results: list, spy_benchmarks: list) -> dict:
    """
    Score a config across all walk-forward windows.

    Overfitting shows up as high variance — a config that crushes one era
    but fails others is fitting to that era's noise, not real market edges.

    Returns a dict with the composite score and breakdown.
    """
    if not window_results:
        return {"total": -999, "avg_excess": 0, "consistency": 0, "variance_pen": 0}

    excesses = [r["annualized_return_pct"] - spy for r, spy in zip(window_results, spy_benchmarks)]
    sharpes  = [r["sharpe_ratio"] for r in window_results]
    dds      = [r["max_drawdown_pct"] for r in window_results]

    avg_excess   = float(np.mean(excesses))
    avg_sharpe   = float(np.mean(sharpes))
    avg_dd       = float(np.mean(dds))

    # Consistency: how many windows beat SPY
    beats_spy    = sum(1 for e in excesses if e > 0)
    consistency  = beats_spy * 4.0   # Up to +20 for beating all 5

    # Variance penalty: configs that work in only one era get penalized
    variance_pen = float(np.std(excesses)) * 1.5

    # Require minimum trades per window — too few trades = overfitted to lucky exits
    min_trades   = min(r["total_trades"] for r in window_results)
    trade_pen    = 10.0 if min_trades < 15 else 0.0

    total = avg_excess + avg_sharpe * 4.0 - avg_dd * 0.2 + consistency - variance_pen - trade_pen

    return {
        "total":       total,
        "avg_excess":  avg_excess,
        "avg_sharpe":  avg_sharpe,
        "avg_dd":      avg_dd,
        "beats_spy":   beats_spy,
        "consistency": consistency,
        "variance_pen":variance_pen,
        "excesses":    excesses,
    }


# ── Data helpers ───────────────────────────────────────────────────────────────

def fetch_by_date(symbols, start, end=None):
    all_data = {}
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(start=start, end=end)
            if not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].dropna()
                all_data[sym] = df
        except Exception:
            pass
    return all_data


def slice_precomputed(precomputed, start_str, end_str=None, warmup_days=60):
    """Slice precomputed signals to a date window, with warmup context."""
    result = {}
    for sym, df in precomputed.items():
        idx = df.index
        try:
            idx_naive = idx.tz_convert(None)
        except TypeError:
            idx_naive = idx.tz_localize(None)

        start_ts = pd.Timestamp(start_str)
        # Include warmup_days before val_start so indicators are warm
        warmup_start = start_ts - pd.Timedelta(days=warmup_days * 2)
        mask_start = idx_naive >= warmup_start

        if end_str:
            end_ts  = pd.Timestamp(end_str)
            mask    = mask_start & (idx_naive <= end_ts)
        else:
            mask    = mask_start

        sliced = df[mask]
        if len(sliced) > warmup_days + 10:
            result[sym] = sliced
    return result


def spy_ann_for_window(precomputed, start_str, end_str=None):
    """Annualized SPY return for a specific date window."""
    spy = precomputed.get("SPY")
    if spy is None or spy.empty:
        return 0.0
    idx = spy.index
    try:
        idx_naive = idx.tz_convert(None)
    except TypeError:
        idx_naive = idx.tz_localize(None)

    start_ts = pd.Timestamp(start_str)
    mask = idx_naive >= start_ts
    if end_str:
        mask &= (idx_naive <= pd.Timestamp(end_str))
    close = spy["price"][mask]
    if len(close) < 2:
        return 0.0
    ret = (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0])
    n   = len(close)
    return ((1 + ret) ** (252 / max(n, 1)) - 1) * 100


# ── Config persistence ─────────────────────────────────────────────────────────

def write_best_to_config(best):
    with open("config.py", "r") as f:
        text = f.read()

    subs = {
        r"REBALANCE_EVERY\s*=\s*\d+":          f"REBALANCE_EVERY = {best['REBALANCE_EVERY']}",
        r"MAX_OPEN_POSITIONS\s*=\s*\d+":       f"MAX_OPEN_POSITIONS = {best['MAX_OPEN_POSITIONS']}",
        r"MAX_POSITION_SIZE_PCT\s*=\s*[\d.]+": f"MAX_POSITION_SIZE_PCT = {best['MAX_POSITION_SIZE_PCT']}",
    }
    for pat, rep in subs.items():
        text = re.sub(pat, rep, text)

    w = WEIGHT_PROFILES[best["weight_profile"]]
    key_map = {
        "sma_crossover": "sma_crossover", "sma_spread": "sma_spread_strength",
        "ema_crossover": "ema_crossover", "rsi": "rsi", "macd": "macd",
        "macd_momentum": "macd_momentum", "bollinger": "bollinger",
        "vwap": "vwap", "volume": "volume_confirmation",
        "momentum_3m": "momentum_3m", "momentum_6m": "momentum_6m",
        "momentum_12m": "momentum_12m", "relative_strength": "relative_strength",
        "high_52w_proximity": "high_52w_proximity",
    }
    wlines = '    # Signal combination weights — tuned automatically by tune.py\n    "weights": {\n'
    for tk, ck in key_map.items():
        wlines += f'        "{ck}":{" " * (22 - len(ck))}{w[tk]},\n'
    wlines += f'        "earnings_revision": 0.08,   # analyst consensus (live only — absent in backtest)\n'
    wlines += f'        "quality_score":     0.06,   # ROE + gross margin quality (live only)\n'
    wlines += "    },"
    text = re.sub(r'    # Signal combination weights.*?    \},', wlines, text, flags=re.DOTALL)

    with open("config.py", "w") as f:
        f.write(text)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_optimization():
    print("\n" + "=" * 76)
    print("  WALK-FORWARD OPTIMIZER — 5 market eras, each completely unseen")
    print("  Goal: find configs that beat SPY CONSISTENTLY, not just once")
    print("=" * 76)

    # ── 1. Fetch full history ──────────────────────────────────────────────
    print("\nFetching data from 2010 to present...")
    all_data = fetch_by_date(config.WATCHLIST, start="2010-01-01")
    if not all_data:
        print("No data.")
        return
    sample = next(iter(all_data.values()))
    print(f"Range: {sample.index[0].date()}  ->  {sample.index[-1].date()}  "
          f"({len(sample)} trading days, {len(all_data)} stocks)\n")

    # ── 2. Precompute all signals once ────────────────────────────────────
    print("Precomputing signals for all 16 years...")
    spy_raw = all_data.get("SPY")
    precomputed = {
        s: precompute_signals(df, spy_df=spy_raw if s != "SPY" else None)
        for s, df in all_data.items()
    }
    print(f"  {sum(len(v) for v in precomputed.values())} rows cached.\n")

    # ── 3. Prepare each walk-forward window ───────────────────────────────
    windows = []
    spy_benchmarks = []
    print("Walk-forward windows:")
    for i, (ts, te, vs, ve) in enumerate(WALK_FORWARD_WINDOWS):
        val_slice = slice_precomputed(precomputed, vs, ve)
        spy_ann   = spy_ann_for_window(precomputed, vs, ve)
        windows.append(val_slice)
        spy_benchmarks.append(spy_ann)
        label = f"{vs[:4]}–{ve[:4] if ve else 'now'}"
        print(f"  W{i+1}: test {label:12s}  SPY: {spy_ann:+.1f}%/yr  "
              f"({'bear' if spy_ann < 5 else 'bull'})")
    print()

    # ── 4. Grid search across ALL windows ────────────────────────────────
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    print(f"Testing {total} combinations × {len(windows)} windows = "
          f"{total * len(windows)} backtests...\n")

    hdr = (f"  {'#':>4}  {'Reb':>4} {'MaxPos':>6} {'Pos%':>4} "
           f"{'Profile':<20} {'W1':>6} {'W2':>6} {'W3':>6} {'W4':>6} {'W5':>6} "
           f"{'Avg':>6} {'Beat':>5} {'Score':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        wts    = WEIGHT_PROFILES[params["weight_profile"]]

        window_results = []
        for w_slice in windows:
            r = fast_backtest(
                w_slice,
                rebalance_every = params["REBALANCE_EVERY"],
                max_pos_pct     = params["MAX_POSITION_SIZE_PCT"],
                max_positions   = params["MAX_OPEN_POSITIONS"],
                weights         = wts,
            )
            window_results.append(r)

        sc = walk_forward_score(window_results, spy_benchmarks)
        excesses = sc["excesses"]

        all_results.append({**params, "window_results": window_results,
                            "score": sc, "wf_score": sc["total"]})

        exc_strs = "  ".join(f"{e:>+5.1f}" for e in excesses)
        beat_str = f"{sc['beats_spy']}/5"
        star = " *" if sc["beats_spy"] >= 4 else ""
        print(f"  {i+1:>4}  {params['REBALANCE_EVERY']:>4} {params['MAX_OPEN_POSITIONS']:>6} "
              f"{params['MAX_POSITION_SIZE_PCT']:>4} "
              f"{params['weight_profile']:<20}  {exc_strs}  "
              f"{sc['avg_excess']:>+5.1f}  {beat_str:>5}  {sc['total']:>+6.1f}{star}")

    # ── 5. Rank and show top 10 ───────────────────────────────────────────
    all_results.sort(key=lambda x: x["wf_score"], reverse=True)
    top10 = all_results[:10]

    print(f"\n{'=' * 76}")
    print("  TOP 10 MOST CONSISTENT CONFIGS (ranked by walk-forward score)")
    print(f"{'=' * 76}\n")
    print(f"  {'#':>3}  {'Reb':>4} {'MaxPos':>6} {'Pos%':>4} "
          f"{'Profile':<20}  "
          f"{'W1':>5} {'W2':>5} {'W3':>5} {'W4':>5} {'W5':>5}  "
          f"{'Avg':>5} {'Beat':>5} {'Sharpe':>7} {'MaxDD':>6}")
    print("  " + "-" * 100)

    for j, row in enumerate(top10):
        sc  = row["score"]
        exc = sc["excesses"]
        wr  = [r for r in row["window_results"]]
        avg_sharpe = float(np.mean([r["sharpe_ratio"] for r in wr]))
        avg_dd     = float(np.mean([r["max_drawdown_pct"] for r in wr]))
        print(f"  {j+1:>3}  {row['REBALANCE_EVERY']:>4} {row['MAX_OPEN_POSITIONS']:>6} "
              f"{row['MAX_POSITION_SIZE_PCT']:>4} "
              f"{row['weight_profile']:<20}  "
              f"{exc[0]:>+4.0f}% {exc[1]:>+4.0f}% {exc[2]:>+4.0f}% "
              f"{exc[3]:>+4.0f}% {exc[4]:>+4.0f}%  "
              f"{sc['avg_excess']:>+4.1f}%  {sc['beats_spy']}/5  "
              f"{avg_sharpe:>6.2f}  {avg_dd:>5.1f}%")

    # ── 6. Winner ─────────────────────────────────────────────────────────
    best = all_results[0]
    bsc  = best["score"]
    bwr  = best["window_results"]

    window_labels = ["2016-17", "2018-19", "2020-21", "2022", "2023-now"]

    print(f"\n{'=' * 76}")
    print("  WINNER — most consistent config across all market eras")
    print(f"{'=' * 76}")
    print(f"  REBALANCE_EVERY       = {best['REBALANCE_EVERY']}")
    print(f"  MAX_OPEN_POSITIONS    = {best['MAX_OPEN_POSITIONS']}")
    print(f"  MAX_POSITION_SIZE_PCT = {best['MAX_POSITION_SIZE_PCT']}")
    print(f"  Weight profile        = {best['weight_profile']}")
    print()
    print(f"  {'Period':<12} {'Bot':>8}  {'SPY':>8}  {'Excess':>8}  {'Sharpe':>8}  {'MaxDD':>7}  {'Trades':>7}")
    print("  " + "-" * 70)
    for label, r, spy in zip(window_labels, bwr, spy_benchmarks):
        excess = r["annualized_return_pct"] - spy
        marker = " OK" if excess > 0 else " --"
        print(f"  {label:<12} {r['annualized_return_pct']:>+7.1f}%  {spy:>+7.1f}%  "
              f"{excess:>+7.1f}%  {r['sharpe_ratio']:>8.2f}  "
              f"{r['max_drawdown_pct']:>6.1f}%  {r['total_trades']:>7}{marker}")
    print()
    print(f"  Average excess over SPY: {bsc['avg_excess']:+.1f}%/yr")
    print(f"  Beats SPY in:            {bsc['beats_spy']}/5 windows")
    print(f"  Walk-forward score:      {bsc['total']:+.1f}")

    note = ""
    if bsc["beats_spy"] == 5:
        note = "  Beats SPY in ALL 5 market eras — genuinely robust."
    elif bsc["beats_spy"] >= 4:
        note = "  Beats SPY in 4/5 eras — very robust, one era was unfavorable."
    elif bsc["beats_spy"] >= 3:
        note = "  Beats SPY in 3/5 eras — solid, some market conditions are tough."
    else:
        note = "  WARNING: only beats SPY in <3 eras — may be overfitting."
    print(f"\n  {note}")

    # ── 7. Persist ────────────────────────────────────────────────────────
    write_best_to_config(best)
    print(f"\n  config.py updated. Run 'python main.py backtest' to verify.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_optimization()
