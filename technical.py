"""
TECHNICAL.PY — The chart-reading brain.

Calculates technical indicators from OHLCV price data and produces
a single score from -1.0 (strong sell) to +1.0 (strong buy).

NEW: ADX (Average Directional Index) measures trend strength 0-100.
When ADX is low (< 18) the market is choppy and signals are less
reliable — we dampen the score. When ADX is high (> 30) the trend
is strong and we give signals more weight.
"""

import pandas as pd
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)
TECH = config.TECHNICAL


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = None) -> pd.Series:
    if period is None:
        period = TECH["rsi_period"]
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series) -> tuple:
    ema_fast    = calculate_ema(series, TECH["macd_fast"])
    ema_slow    = calculate_ema(series, TECH["macd_slow"])
    macd_line   = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, TECH["macd_signal"])
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(series: pd.Series) -> tuple:
    middle = calculate_sma(series, TECH["bb_period"])
    std    = series.rolling(window=TECH["bb_period"]).std()
    upper  = middle + (TECH["bb_std"] * std)
    lower  = middle - (TECH["bb_std"] * std)
    return upper, middle, lower


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return vwap


def calculate_volume_ratio(volume: pd.Series, period: int = None) -> pd.Series:
    if period is None:
        period = TECH["volume_avg_period"]
    avg_vol = volume.rolling(window=period).mean()
    return volume / avg_vol


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Average Directional Index — measures trend STRENGTH (not direction).

    0–18:  No trend / choppy — signals are noisy, reduce confidence.
    18–25: Weak trend forming.
    25–40: Strong trend — signals are more reliable.
    40+:   Very strong trend (common during breakouts / crashes).

    Formula follows the original Wilder (1978) smoothing convention.
    """
    if period is None:
        period = TECH.get("adx_period", 14)

    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement (only one fires per bar — the larger move wins)
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing (EWM with alpha = 1/period)
    alpha     = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s= minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di   = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
    minus_di  = 100 * minus_dm_s / atr_s.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.fillna(0)


def calculate_efficiency_ratio(series: pd.Series, period: int = 60) -> float:
    """
    Kaufman Efficiency Ratio — measures trend smoothness.
    |net_change| / sum(|daily_changes|)
    1.0 = perfectly smooth trend, 0.0 = pure noise.
    """
    if len(series) < period:
        return 0.0
    net_change = abs(float(series.iloc[-1] - series.iloc[-period]))
    sum_changes = float(series.diff().abs().iloc[-period:].sum())
    if sum_changes == 0:
        return 0.0
    return net_change / sum_changes


def analyze(df: pd.DataFrame, benchmark_df: pd.DataFrame = None,
            live_fundamentals: dict = None,
            sector_momentum: float = None) -> dict:
    """
    Run all technical indicators and return a single trading signal.

    Args:
        df:           OHLCV DataFrame for the stock being analyzed.
        benchmark_df: Optional OHLCV DataFrame for a benchmark (e.g. SPY).
                      Used to compute relative strength — how much the stock
                      outperforms / underperforms the market.

    Returns:
        {
          "score":      float -1.0 to +1.0
          "signals":    dict of individual indicator values
          "indicators": dict of raw values for logging / debugging
        }
    """
    if len(df) < TECH["sma_slow"] + 10:
        logger.warning("Not enough data for full technical analysis")
        return {"score": 0.0, "signals": {}, "indicators": {}}

    close  = df["Close"]
    latest = close.iloc[-1]
    signals = {}

    # ── 1. SMA Crossover ────────────────────────────────────────────
    sma_fast = calculate_sma(close, TECH["sma_fast"])
    sma_slow = calculate_sma(close, TECH["sma_slow"])
    signals["sma_crossover"]       = 1.0 if sma_fast.iloc[-1] > sma_slow.iloc[-1] else -1.0
    sma_spread                     = (sma_fast.iloc[-1] - sma_slow.iloc[-1]) / sma_slow.iloc[-1]
    signals["sma_spread_strength"] = float(np.clip(sma_spread * 20, -1.0, 1.0))

    # ── 2. EMA Crossover ────────────────────────────────────────────
    ema_fast = calculate_ema(close, TECH["ema_fast"])
    ema_slow = calculate_ema(close, TECH["ema_slow"])
    signals["ema_crossover"] = 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1.0

    # ── 3. RSI — momentum-oriented ──────────────────────────────────
    # Classic use: buy oversold (<30), sell overbought (>70) = mean-reversion.
    # For a momentum strategy, high RSI is CONFIRMATION of trend strength:
    #   RSI 70–80 = momentum sweet spot (strong, not yet euphoric)
    #   RSI 50    = neutral
    #   RSI 30    = weak trend
    # This aligns with 12M momentum, relative strength, and 52W-high signals.
    rsi       = calculate_rsi(close)
    rsi_value = rsi.iloc[-1]
    signals["rsi"] = float(np.clip((rsi_value - 50.0) / 25.0, -1.0, 1.0))

    # ── 4. MACD ─────────────────────────────────────────────────────
    macd_line, signal_line, histogram = calculate_macd(close)
    signals["macd"] = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1.0
    if len(histogram) >= 2:
        signals["macd_momentum"] = 0.5 if histogram.iloc[-1] > histogram.iloc[-2] else -0.5

    # ── 5. Bollinger Bands — momentum-oriented ──────────────────────
    # Classic use: buy at lower band, sell at upper band = mean-reversion.
    # For momentum: price in upper half of band = trend intact (bullish);
    # price below middle band = trend broken (bearish).
    # Upper band touch can indicate a breakout continuation in strong trends.
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    if bb_range > 0:
        bb_position       = (latest - bb_lower.iloc[-1]) / bb_range
        signals["bollinger"] = float(np.clip(2.0 * bb_position - 1.0, -1.0, 1.0))
    else:
        signals["bollinger"] = 0.0

    # ── 6. Volume Confirmation ──────────────────────────────────────
    vol_ratio       = calculate_volume_ratio(df["Volume"])
    vol_ratio_latest= float(vol_ratio.iloc[-1]) if not np.isnan(vol_ratio.iloc[-1]) else 1.0
    volume_multiplier = float(np.clip(vol_ratio_latest, 0.5, 2.0))

    if len(close) >= 2:
        price_change_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        if vol_ratio_latest >= 1.2:
            signals["volume_confirmation"] = float(np.clip(price_change_pct * 30, -1.0, 1.0))
        else:
            signals["volume_confirmation"] = float(np.clip(price_change_pct * 10, -0.3, 0.3))
    else:
        signals["volume_confirmation"] = 0.0

    # ── 7. VWAP ─────────────────────────────────────────────────────
    vwap = calculate_vwap(df)
    signals["vwap"] = 0.5 if latest > vwap.iloc[-1] else -0.5

    # ── 8. Medium-term Momentum ─────────────────────────────────────
    roc_63  = close.pct_change(63)
    roc_126 = close.pct_change(126)
    signals["momentum_3m"] = float(np.clip(roc_63.iloc[-1]  * 3,   -1.0, 1.0)) if not pd.isna(roc_63.iloc[-1])  else 0.0
    signals["momentum_6m"] = float(np.clip(roc_126.iloc[-1] * 1.5, -1.0, 1.0)) if not pd.isna(roc_126.iloc[-1]) else 0.0

    # ── 9. ATR (for position sizing) ────────────────────────────────
    atr     = calculate_atr(df)
    atr_pct = float(atr.iloc[-1] / latest) if latest > 0 else 0.02

    # ── 10. ADX — trend strength filter ────────────────────────────
    adx_series = calculate_adx(df)
    adx_value  = float(adx_series.iloc[-1])
    adx_min    = TECH.get("adx_min_trend", 18)

    # ── 11. 12-Month Momentum (skip last month) ──────────────────────
    # Per Jegadeesh & Titman (1993): 12-month trailing return predicts
    # 3–6 month forward return. Skipping the last month avoids short-term
    # mean-reversion that would otherwise dilute the signal.
    if len(close) >= 252:
        roc_12m = float(close.iloc[-21] / close.iloc[-252] - 1)  # 12M, skip 1M
        signals["momentum_12m"] = float(np.clip(roc_12m * 1.5, -1.0, 1.0))
    # else: omitted — weight sum adjusts automatically

    # ── 12. Relative Strength vs Benchmark (SPY) ────────────────────
    # Buy market leaders, not laggards. Only long stocks outperforming
    # the S&P 500 on a 3-month basis.
    if (benchmark_df is not None and
            len(benchmark_df) >= 63 and len(close) >= 63):
        bench_close = benchmark_df["Close"]
        stock_3m = float(close.iloc[-1] / close.iloc[-63] - 1)
        bench_3m = float(bench_close.iloc[-1] / bench_close.iloc[-63] - 1)
        rs_delta  = stock_3m - bench_3m          # positive = outperforming
        signals["relative_strength"] = float(np.clip(rs_delta * 5, -1.0, 1.0))
    # else: omitted — weight sum adjusts automatically

    # ── 13. 52-Week High Proximity ───────────────────────────────────
    # George & Hwang (2004): stocks near their 52-week high have higher
    # forward returns because anchoring bias delays full price discovery.
    lookback_52w = min(252, len(close))
    if lookback_52w >= 63:
        high_52w  = float(close.iloc[-lookback_52w:].max())
        if high_52w > 0:
            proximity = latest / high_52w   # 1.0 = at the 52-week high
            # >0.95 → bullish (+1), <0.70 → bearish (-1), linear between
            signals["high_52w_proximity"] = float(np.clip((proximity - 0.85) * 6.67, -1.0, 1.0))

    # ── 13b. Sector Momentum ─────────────────────────────────────
    # Moskowitz & Grinblatt (1999): sector momentum is a distinct anomaly.
    # Stocks in outperforming sectors have higher forward returns.
    # sector_momentum is pre-computed by the caller (backtester/live loop)
    # as the average 6M return of all stocks in the same sector.
    if sector_momentum is not None:
        signals["sector_momentum"] = float(np.clip(sector_momentum * 3, -1.0, 1.0))

    # ── 14. Analyst Revision & Quality (live trading only) ──────────────
    # These require current fundamental data — unavailable for historical backtesting.
    # When live_fundamentals is None (backtest mode), signals are simply absent
    # and weight normalization redistributes their weight to other signals.
    if live_fundamentals:
        if "earnings_revision" in live_fundamentals:
            signals["earnings_revision"] = float(live_fundamentals["earnings_revision"])
        if "quality_score" in live_fundamentals:
            signals["quality_score"] = float(live_fundamentals["quality_score"])

    # ── 15a. Idiosyncratic Momentum ────────────────────────────────
    # Blitz, Huij & Martens (2011): momentum computed on market-residual
    # returns is more predictive and less crash-prone than raw momentum.
    # Strip out market (SPY) beta, compute 12M momentum on residuals.
    if benchmark_df is not None and len(benchmark_df) >= 252 and len(close) >= 252:
        stock_rets = close.pct_change().iloc[-252:]
        bench_rets = benchmark_df["Close"].pct_change().reindex(close.index, method="ffill").iloc[-252:]
        # Simple beta = cov / var (no look-ahead: uses trailing 252d)
        valid_mask = stock_rets.notna() & bench_rets.notna()
        if valid_mask.sum() >= 60:
            sr = stock_rets[valid_mask].values
            br = bench_rets[valid_mask].values
            beta = np.cov(sr, br)[0, 1] / max(np.var(br), 1e-10)
            beta = np.clip(beta, -3.0, 3.0)
            residual_rets = sr - beta * br
            idio_mom = float(np.sum(residual_rets[:-21]))  # skip last month
            signals["idiosyncratic_momentum"] = float(np.clip(idio_mom * 3, -1.0, 1.0))

    # ── 15b. Volatility-Adjusted Momentum ──────────────────────────
    # Barroso & Santa-Clara (2015): scaling momentum by inverse volatility
    # produces higher Sharpe ratios and dramatically reduces crash risk.
    # This is the single most robust improvement to raw momentum.
    if len(close) >= 252:
        raw_mom = float(close.iloc[-21] / close.iloc[-252] - 1)
        realized_vol_12m = float(close.pct_change().iloc[-252:].std() * np.sqrt(252))
        if realized_vol_12m > 0.01:
            info_ratio = raw_mom / realized_vol_12m
            signals["vol_adj_momentum"] = float(np.clip(info_ratio, -1.0, 1.0))

    # ── 16. Momentum Acceleration ─────────────────────────────────
    # Is momentum getting stronger or fading? Accelerating momentum
    # predicts continuation; decelerating momentum precedes reversals.
    if len(close) >= 126:
        mom_3m_now  = float(close.iloc[-1]  / close.iloc[-63]  - 1)
        mom_3m_prev = float(close.iloc[-63] / close.iloc[-126] - 1)
        accel = mom_3m_now - mom_3m_prev
        signals["momentum_acceleration"] = float(np.clip(accel * 5, -1.0, 1.0))

    # ── 17. Short-Term Reversal (5-day) ───────────────────────────
    # Within uptrending momentum stocks, short-term pullbacks offer
    # better entries (Jegadeesh 1990). Negative 5-day return = buying opp.
    # Weight is low so this only acts as a tiebreaker among similar stocks.
    if len(close) >= 10:
        ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1)
        signals["short_term_reversal"] = float(np.clip(-ret_5d * 10, -1.0, 1.0))

    # ── 18. Trend Efficiency (Kaufman ER) ─────────────────────────
    # Smooth trends are more persistent than noisy ones with the same
    # total return. High efficiency = straight-line move, not zigzag.
    if len(close) >= 60:
        efficiency = calculate_efficiency_ratio(close, 60)
        direction = 1.0 if close.iloc[-1] > close.iloc[-60] else -1.0
        signals["trend_efficiency"] = float(np.clip(efficiency * 2 * direction, -1.0, 1.0))

    # ── 19. Volume-Price Confirmation ─────────────────────────────
    # Blume, Easley, O'Hara (1994): volume carries information.
    # When price rises on increasing volume, the trend is more reliable.
    if len(df) >= 40:
        price_ret_20d = float(close.iloc[-1] / close.iloc[-20] - 1)
        vol_recent = float(df["Volume"].iloc[-20:].mean())
        vol_prior  = float(df["Volume"].iloc[-40:-20].mean())
        if vol_prior > 0:
            vol_change = (vol_recent / vol_prior) - 1.0
            # Positive when price direction and volume direction agree
            confirmation = price_ret_20d * vol_change
            signals["volume_price_confirm"] = float(np.clip(confirmation * 20, -1.0, 1.0))

    # ── 20. Drawdown Recovery Score ──────────────────────────────
    # Daniel & Moskowitz (2016): stocks in deep drawdown from recent
    # highs are momentum crash candidates. Penalize stocks >25% off
    # their 6-month high that haven't started recovering.
    if len(close) >= 126:
        high_6m = float(close.iloc[-126:].max())
        if high_6m > 0:
            dd_from_6m_high = 1.0 - (latest / high_6m)
            if dd_from_6m_high > 0.25:
                # Deep drawdown: penalize. Recovering stocks (positive 10d) penalized less.
                ret_10d = float(close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
                if ret_10d > 0.02:
                    # Recovering from crash — mild penalty
                    signals["drawdown_recovery"] = float(np.clip(-dd_from_6m_high * 1.5 + 0.3, -1.0, 0.3))
                else:
                    # Still falling — strong penalty
                    signals["drawdown_recovery"] = float(np.clip(-dd_from_6m_high * 3, -1.0, -0.2))
            else:
                # Not in deep drawdown — neutral to slightly positive
                signals["drawdown_recovery"] = float(np.clip((0.25 - dd_from_6m_high) * 2, 0.0, 0.5))

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE SCORE
    # ═══════════════════════════════════════════════════════════════
    weights = TECH.get("weights", {
        "sma_crossover": 0.08, "sma_spread_strength": 0.05,
        "ema_crossover": 0.08, "rsi": 0.05, "macd": 0.08,
        "macd_momentum": 0.04, "bollinger": 0.03, "vwap": 0.03,
        "volume_confirmation": 0.03, "momentum_3m": 0.12,
        "momentum_6m": 0.10, "momentum_12m": 0.14,
        "relative_strength": 0.10, "high_52w_proximity": 0.07,
    })

    # Normalise so missing signals (insufficient history) don't distort the score
    available_weight = sum(weights[k] for k in weights if k in signals)
    if available_weight > 0:
        raw_score = sum(signals[k] * weights[k] for k in weights if k in signals)
        raw_score /= available_weight   # re-normalise to [-1, 1] scale
    else:
        raw_score = 0.0

    # Volume amplification — high volume confirms direction
    raw_score *= min(volume_multiplier, 1.5)

    # ADX trend-strength scaling using a sigmoid S-curve.
    # Smoother than the original linear transition — avoids sharp
    # signal jumps near the adx_min threshold.
    #
    #   adx = 0  → scalar ≈ 0.35 (strongly dampen choppy-market noise)
    #   adx = 18 → scalar ≈ 0.78 (neutral/threshold)
    #   adx = 28 → scalar ≈ 1.00 (full signal)
    #   adx = 40 → scalar ≈ 1.15 (strong trend amplification)
    #   adx = 50 → scalar ≈ 1.18 (caps out near 1.20)
    x          = (adx_value - adx_min) / 8.0
    adx_scalar = 0.35 + (1.0 / (1.0 + np.exp(-x))) * 0.85

    raw_score *= adx_scalar

    final_score = float(np.clip(raw_score, -1.0, 1.0))

    # Realized volatility for inverse-vol position sizing
    realized_vol_60d = float(close.pct_change().iloc[-min(60, len(close)-1):].std() * np.sqrt(252)) if len(close) > 10 else 0.20

    indicators = {
        "price":              latest,
        "sma_fast":           sma_fast.iloc[-1],
        "sma_slow":           sma_slow.iloc[-1],
        "ema_fast":           ema_fast.iloc[-1],
        "ema_slow":           ema_slow.iloc[-1],
        "rsi":                rsi_value,
        "macd":               macd_line.iloc[-1],
        "macd_signal":        signal_line.iloc[-1],
        "bb_upper":           bb_upper.iloc[-1],
        "bb_lower":           bb_lower.iloc[-1],
        "vwap":               vwap.iloc[-1],
        "volume_ratio":       vol_ratio_latest,
        "atr":                atr.iloc[-1],
        "atr_pct":            atr_pct,
        "adx":                adx_value,
        "realized_vol":       realized_vol_60d,
        "momentum_3m":        signals.get("momentum_3m", 0.0),
        "momentum_6m":        signals.get("momentum_6m", 0.0),
        "momentum_12m":       signals.get("momentum_12m", 0.0),
        "relative_strength":  signals.get("relative_strength", 0.0),
        "high_52w_proximity": signals.get("high_52w_proximity", 0.0),
        "vol_adj_momentum":   signals.get("vol_adj_momentum", 0.0),
        "momentum_acceleration": signals.get("momentum_acceleration", 0.0),
        "short_term_reversal": signals.get("short_term_reversal", 0.0),
        "trend_efficiency":   signals.get("trend_efficiency", 0.0),
        "volume_price_confirm": signals.get("volume_price_confirm", 0.0),
        "earnings_revision":  signals.get("earnings_revision", None),
        "quality_score":      signals.get("quality_score", None),
        "idiosyncratic_momentum": signals.get("idiosyncratic_momentum", 0.0),
        "sector_momentum":    signals.get("sector_momentum", 0.0),
        "drawdown_recovery":  signals.get("drawdown_recovery", 0.0),
    }

    return {"score": final_score, "signals": signals, "indicators": indicators}
