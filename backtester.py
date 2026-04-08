"""
BACKTESTER.PY — Test the strategy against history BEFORE risking real money.

Walk-forward approach: the bot only ever sees data up to the current
simulated day — no look-ahead bias.

Cross-sectional momentum strategy: always hold the top N stocks by momentum
score, rotating out rank-droppers on each rebalance day. This follows the
academic momentum literature rather than a signal-threshold approach.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

import config
from technical import analyze as technical_analyze
from technical import calculate_sma
from data_fetcher import fetch_vix_history, fetch_yield_curve_history

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol:      str
    entry_date:  str
    entry_price: float
    exit_date:   str  = ""
    exit_price:  float = 0.0
    qty:         int   = 0
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    exit_reason: str  = ""


@dataclass
class BacktestResult:
    starting_cash:        float
    ending_value:         float
    total_return_pct:     float
    annualized_return_pct:float
    sharpe_ratio:         float
    max_drawdown_pct:     float
    total_trades:         int
    winning_trades:       int
    losing_trades:        int
    win_rate:             float
    profit_factor:        float
    avg_win_pct:          float
    avg_loss_pct:         float
    largest_win_pct:      float
    largest_loss_pct:     float
    spy_total_return_pct: float = 0.0
    spy_annualized_pct:   float = 0.0
    trades:               list  = field(default_factory=list)
    equity_curve:         list  = field(default_factory=list)
    daily_returns:        list  = field(default_factory=list)


def run_backtest(
    all_price_data: dict,
    starting_cash:  float = None,
    warmup_days:    int   = 60,
    verbose:        bool  = True,
) -> BacktestResult:
    if starting_cash is None:
        starting_cash = config.BACKTEST_STARTING_CASH

    cash          = starting_cash
    positions     = {}
    all_trades    = []
    equity_curve  = []
    daily_returns = []
    # Re-entry cooldown: maps symbol → earliest day_idx it can be re-bought
    reentry_cooldown = {}
    REENTRY_COOLDOWN_DAYS = getattr(config, "REENTRY_COOLDOWN_DAYS", 0)

    REBALANCE_EVERY = getattr(config, 'REBALANCE_EVERY', 5)

    # Portfolio drawdown scaling — reduces sizing during drawdowns
    peak_equity = starting_cash
    dd_scalar   = 1.0

    # Trailing stop / ATR stop config
    TRAILING_STOP_PCT   = getattr(config, "TRAILING_STOP_PCT", 0.15)
    ATR_STOP_MULTIPLIER = getattr(config, "ATR_STOP_MULTIPLIER", 3.0)
    ATR_STOP_MIN_PCT    = getattr(config, "ATR_STOP_MIN_PCT", 0.06)
    ATR_STOP_MAX_PCT    = getattr(config, "ATR_STOP_MAX_PCT", 0.15)

    # Build sorted common date list
    all_dates = sorted(set.union(*[
        set(df.index.strftime("%Y-%m-%d").tolist())
        for df in all_price_data.values()
    ]))

    # Pre-build date→row-index lookup for each symbol (avoids slow strftime
    # masking on every day × every position during the main loop).
    _date_idx = {}
    for sym, df in all_price_data.items():
        date_strs = df.index.strftime("%Y-%m-%d")
        _date_idx[sym] = {d: i for i, d in enumerate(date_strs)}

    # ── SPY buy-and-hold benchmark ───────────────────────────────────
    # Buy SPY at the first post-warmup close; sell at the last close.
    spy_bh_shares = 0.0
    spy_bh_cost   = 0.0
    spy_first_close = None

    # ── Pre-compute SPY market regime per date ───────────────────────
    # Three-tier regime:
    #   FULL BULL:  SMA50 > SMA200 → fully invested
    #   RECOVERY:   Price > SMA50 but SMA50 < SMA200 → partially invested
    #               (catches the rally BEFORE the golden cross completes)
    #   BEAR:       Price < SMA50 and SMA50 < SMA200 → park in BIL
    spy_bull_regime = {}     # True = full bull OR recovery
    spy_recovery_regime = {} # True = recovery only (partial allocation)
    spy_early_recovery = {}  # True = bear market bounce (V-recovery detection)
    # High-volatility guard: skip new long entries when 10-day SPY realized
    # vol exceeds 40% annualized (≈ VIX 40+, extreme stress only).
    HIGH_VOL_THRESHOLD = 0.40
    spy_high_vol = {}
    if "SPY" in all_price_data:
        spy_close  = all_price_data["SPY"]["Close"]
        spy_sma50  = spy_close.rolling(50).mean()
        spy_sma200 = spy_close.rolling(200).mean()
        spy_vol_10 = spy_close.pct_change().rolling(10).std() * np.sqrt(252)
        # Track SPY peak for drawdown-based early recovery detection
        spy_peak = spy_close.expanding().max()
        spy_dd_from_peak = (spy_peak - spy_close) / spy_peak
        spy_ret_10d = spy_close.pct_change(10)
        spy_ret_21d = spy_close.pct_change(21)
        for dt_idx in range(len(spy_close)):
            dk    = all_price_data["SPY"].index[dt_idx].strftime("%Y-%m-%d")
            s50   = spy_sma50.iloc[dt_idx]
            s200  = spy_sma200.iloc[dt_idx]
            price = float(spy_close.iloc[dt_idx])
            vol   = spy_vol_10.iloc[dt_idx]

            # Staged recovery detection: catches V-bounces at two speeds.
            # Stage 1 (micro): 5-day bounce ≥2% from ≥10% drawdown → 55% allocation
            #   Catches the first-week pop that drives most of the recovery alpha.
            # Stage 2 (early): 10-day bounce ≥1.5% AND positive 21d → 75% allocation
            #   Confirms the recovery is sustained, not a dead-cat bounce.
            dd = spy_dd_from_peak.iloc[dt_idx] if not pd.isna(spy_dd_from_peak.iloc[dt_idx]) else 0
            r10 = spy_ret_10d.iloc[dt_idx] if dt_idx >= 10 and not pd.isna(spy_ret_10d.iloc[dt_idx]) else 0
            r21 = spy_ret_21d.iloc[dt_idx] if dt_idx >= 21 and not pd.isna(spy_ret_21d.iloc[dt_idx]) else 0
            spy_early_recovery[dk] = bool(dd > 0.12 and r10 > 0.015 and r21 > 0)

            if pd.isna(s200) or pd.isna(s50):
                spy_bull_regime[dk] = True
                spy_recovery_regime[dk] = False
            elif s50 > s200:
                # Full bull: golden cross intact
                spy_bull_regime[dk] = True
                spy_recovery_regime[dk] = False
            elif price > s50:
                # Recovery: price above SMA50 but golden cross not yet confirmed
                # Allow investing at reduced allocation
                spy_bull_regime[dk] = True
                spy_recovery_regime[dk] = True
            else:
                # Bear: price below SMA50 and death cross
                spy_bull_regime[dk] = False
                spy_recovery_regime[dk] = False

            spy_high_vol[dk] = bool(not pd.isna(vol) and vol > HIGH_VOL_THRESHOLD)

    # ── Pre-fetch VIX history for position scaling ───────────────────
    vix_by_date = {}
    try:
        vix_series = fetch_vix_history(days=config.LOOKBACK_DAYS + 30)
        for dt, val in vix_series.items():
            dk = pd.Timestamp(dt).strftime("%Y-%m-%d")
            vix_by_date[dk] = float(val)
        logger.info(f"VIX history loaded: {len(vix_by_date)} days")
    except Exception as e:
        logger.warning(f"VIX history unavailable ({e}) — defaulting to 20.0")

    # ── Pre-fetch yield curve history for regime scaling ─────────────
    yc_by_date = {}
    if getattr(config, "USE_YIELD_CURVE_REGIME", False):
        try:
            yc_series = fetch_yield_curve_history(days=config.LOOKBACK_DAYS + 30)
            for dt, val in yc_series.items():
                dk = pd.Timestamp(dt).strftime("%Y-%m-%d")
                yc_by_date[dk] = float(val)
            logger.info(f"Yield curve history loaded: {len(yc_by_date)} days")
        except Exception as e:
            logger.warning(f"Yield curve history unavailable ({e}) — skipping regime scaling")

    if len(all_dates) <= warmup_days:
        logger.error(f"Not enough data: {len(all_dates)} days, need > {warmup_days}")
        return BacktestResult(
            starting_cash=starting_cash, ending_value=starting_cash,
            total_return_pct=0, annualized_return_pct=0, sharpe_ratio=0,
            max_drawdown_pct=0, total_trades=0, winning_trades=0,
            losing_trades=0, win_rate=0, profit_factor=0,
            avg_win_pct=0, avg_loss_pct=0, largest_win_pct=0, largest_loss_pct=0,
        )

    prev_equity = starting_cash

    if verbose:
        print(f"\n{'='*70}")
        print(f"  BACKTESTING on {len(all_dates)} trading days")
        print(f"  Starting cash: ${starting_cash:,.2f}")
        print(f"  Warmup period: {warmup_days} days")
        print(f"  Stocks: {', '.join(all_price_data.keys())}")
        print(f"{'='*70}\n")

    # ── Main day loop ────────────────────────────────────────────────
    for day_idx, date_str in enumerate(all_dates):
        if day_idx < warmup_days:
            equity_curve.append({"date": date_str, "equity": starting_cash})

            # Initialize SPY B&H on the first warmup day using SPY's close
            if spy_first_close is None and "SPY" in all_price_data:
                spy_df = all_price_data["SPY"]
                mask   = spy_df.index.strftime("%Y-%m-%d") <= date_str
                if mask.any():
                    spy_first_close = float(spy_df["Close"][mask].iloc[-1])
                    spy_bh_shares   = starting_cash / spy_first_close
                    spy_bh_cost     = starting_cash
            continue

        slippage    = getattr(config, "SLIPPAGE_PCT", 0.0005)
        regime_only = getattr(config, "REGIME_ONLY_SYMBOLS", {"SPY", "QQQ", "BIL"})
        bear_etf    = getattr(config, "BEAR_REGIME_ETF", None)
        in_bull_now = spy_bull_regime.get(date_str, True)
        in_recovery = spy_recovery_regime.get(date_str, False)
        in_high_vol = spy_high_vol.get(date_str, False)
        in_early_recovery = spy_early_recovery.get(date_str, False)

        # Helper: fast price lookup using pre-built date index
        def _get_price(sym, dt_str):
            idx = _date_idx.get(sym)
            if idx is None:
                return None
            row_i = idx.get(dt_str)
            if row_i is not None:
                return float(all_price_data[sym]["Close"].iloc[row_i])
            # Fallback: find latest date <= dt_str
            sym_dates = list(idx.keys())
            for d in reversed(sym_dates):
                if d <= dt_str:
                    return float(all_price_data[sym]["Close"].iloc[idx[d]])
            return None

        # ── Phase 1: Daily catastrophe check (20% hard stop) ────────
        for symbol in list(positions.keys()):
            if symbol == bear_etf:
                continue
            current_price = _get_price(symbol, date_str)
            if current_price is None:
                continue
            pos = positions[symbol]
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            if pnl_pct <= -0.20:
                fill_price = current_price * (1 - slippage)
                trade = _close_position(positions, symbol, fill_price, date_str, cash, "catastrophe_stop")
                cash += trade.qty * fill_price
                all_trades.append(trade)
                if verbose:
                    _print_trade(trade, "CATAS STOP")

        # ── Phase 1b: Trailing stop + ATR stop (daily) ──────────────
        for symbol in list(positions.keys()):
            if symbol == bear_etf or symbol == "BIL":
                continue
            if symbol not in positions:
                continue
            current_price = _get_price(symbol, date_str)
            if current_price is None:
                continue
            pos = positions[symbol]

            # Update high-water mark
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price

            # Tiered trailing stop: tighten trail as gains grow larger.
            # Protects more profit on big runners without cutting moderate winners.
            gain_from_entry = (pos["highest_price"] - pos["entry_price"]) / pos["entry_price"]
            trailing_stop_tiers = getattr(config, "TRAILING_STOP_TIERS", [(0.10, TRAILING_STOP_PCT)])
            trailing_stop_dist = None
            for tier_gain, tier_trail in trailing_stop_tiers:
                if gain_from_entry >= tier_gain:
                    trailing_stop_dist = tier_trail
                    break
            if trailing_stop_dist is None:
                continue  # not enough profit to activate any tier

            # --- ATR-based stop distance ---
            atr_stop_dist = trailing_stop_dist  # fallback if not enough data
            full_df = all_price_data.get(symbol)
            if full_df is not None:
                sym_idx = _date_idx.get(symbol, {}).get(date_str)
                if sym_idx is not None and sym_idx >= 14:
                    high = full_df["High"].iloc[sym_idx-13:sym_idx+1]
                    low  = full_df["Low"].iloc[sym_idx-13:sym_idx+1]
                    close_prev = full_df["Close"].iloc[sym_idx-14:sym_idx]
                    tr = pd.concat([
                        high - low,
                        (high - close_prev.values).abs(),
                        (low  - close_prev.values).abs(),
                    ], axis=1).max(axis=1)
                    atr_14 = float(tr.mean())
                    atr_pct = atr_14 / current_price if current_price > 0 else 0
                    atr_stop_dist = max(ATR_STOP_MIN_PCT,
                                        min(atr_pct * ATR_STOP_MULTIPLIER, ATR_STOP_MAX_PCT))

            # Use the tighter of the two stops
            effective_stop_pct = min(trailing_stop_dist, atr_stop_dist)

            # Check if price has dropped more than effective_stop_pct from peak
            drop_from_peak = (pos["highest_price"] - current_price) / pos["highest_price"]
            if drop_from_peak >= effective_stop_pct:
                fill_price = current_price * (1 - slippage)
                trade = _close_position(positions, symbol, fill_price, date_str, cash, "trailing_stop")
                cash += trade.qty * fill_price
                all_trades.append(trade)
                if verbose:
                    _print_trade(trade, "TRAIL STOP")

        # ── Phase 2: Rebalance every N days ─────────────────────────
        trading_day_num = day_idx - warmup_days
        is_rebalance_day = (trading_day_num % REBALANCE_EVERY == 0)

        # ── BIL cash parking (only during extreme vol) ────────────
        if is_rebalance_day and bear_etf and bear_etf in all_price_data:
            bil_price = _get_price(bear_etf, date_str)
            if bil_price is not None:
                # Exit BIL when ANY equity allocation is warranted
                if (in_bull_now or not in_high_vol) and bear_etf in positions:
                    cash += positions[bear_etf]["qty"] * bil_price
                    del positions[bear_etf]
                # Only park in BIL during extreme volatility (fully cash)
                elif in_high_vol and bear_etf not in positions and cash > 1000:
                    qty = int(cash * 0.90 / bil_price)
                    if qty > 0:
                        cash -= qty * bil_price
                        positions[bear_etf] = {
                            "qty": qty, "entry_price": bil_price,
                            "entry_date": date_str, "highest_price": bil_price,
                        }

        if is_rebalance_day:
            # Step 1: Score all tradeable stocks by pure momentum
            # Build SPY slice once for all stocks
            spy_row_i = _date_idx.get("SPY", {}).get(date_str)
            if spy_row_i is not None:
                spy_end = spy_row_i + 1
                spy_start = max(0, spy_end - 300)
                spy_ta = all_price_data["SPY"].iloc[spy_start:spy_end]
            else:
                spy_ta = None

            # Pre-compute sector momentum (Moskowitz & Grinblatt 1999):
            # average 6-month return of all stocks in each sector.
            sector_map = getattr(config, "SECTOR_MAP", {})
            sector_returns = {}  # sector → list of 6M returns
            for symbol, full_df in all_price_data.items():
                if symbol in regime_only:
                    continue
                sym_row_i = _date_idx.get(symbol, {}).get(date_str)
                if sym_row_i is not None and sym_row_i >= 126:
                    ret_6m = float(full_df["Close"].iloc[sym_row_i] / full_df["Close"].iloc[sym_row_i - 126] - 1)
                    sec = sector_map.get(symbol)
                    if sec:
                        sector_returns.setdefault(sec, []).append(ret_6m)
            sector_avg_mom = {sec: np.mean(rets) for sec, rets in sector_returns.items()}

            scored_stocks = []
            for symbol, full_df in all_price_data.items():
                if symbol in regime_only:
                    continue
                if symbol == bear_etf:
                    continue
                sym_row_i = _date_idx.get(symbol, {}).get(date_str)
                if sym_row_i is None or sym_row_i < warmup_days:
                    continue
                current_price = float(full_df["Close"].iloc[sym_row_i])
                # Only pass last 300 rows to technical_analyze.
                ta_end   = sym_row_i + 1
                ta_start = max(0, ta_end - 300)
                ta_slice = full_df.iloc[ta_start:ta_end]
                # Sector momentum for this stock's sector
                sym_sector = sector_map.get(symbol)
                sec_mom = sector_avg_mom.get(sym_sector) if sym_sector else None
                try:
                    tech_result = technical_analyze(ta_slice, benchmark_df=spy_ta,
                                                   sector_momentum=sec_mom)
                except Exception:
                    continue
                tech_signals = tech_result.get("signals", {})
                indicators   = tech_result.get("indicators", {})

                # Score using configurable ranking weights with proper
                # renormalization for missing factors (insufficient history)
                ranking_weights = getattr(config, "RANKING_WEIGHTS", {
                    "momentum_12m": 0.30, "momentum_6m": 0.20,
                    "momentum_3m": 0.20, "relative_strength": 0.20,
                    "high_52w_proximity": 0.10,
                })
                avail_wt = sum(w for k, w in ranking_weights.items() if k in tech_signals)
                if avail_wt > 0:
                    mom_score = sum(
                        tech_signals.get(k, 0) * w
                        for k, w in ranking_weights.items()
                        if k in tech_signals
                    ) / avail_wt
                else:
                    mom_score = 0.0

                scored_stocks.append({
                    "symbol":       symbol,
                    "mom_score":    mom_score,
                    "price":        current_price,
                    "indicators":   indicators,
                })

            # Step 2: Determine target portfolio
            scored_stocks.sort(key=lambda x: x["mom_score"], reverse=True)
            sector_map     = getattr(config, "SECTOR_MAP", {})
            max_per_sector = getattr(config, "MAX_POSITIONS_PER_SECTOR", 99)
            sector_max_override = getattr(config, "SECTOR_MAX_OVERRIDE", {})
            max_positions  = config.MAX_OPEN_POSITIONS

            # Market breadth: % of universe with positive momentum score.
            # Breadth deterioration often leads index declines — reduce
            # exposure when fewer stocks are participating in the rally.
            if scored_stocks:
                breadth = sum(1 for s in scored_stocks if s["mom_score"] > 0) / len(scored_stocks)
            else:
                breadth = 0.5
            breadth_scalar = 1.0
            if breadth < 0.20:
                breadth_scalar = 0.40
            elif breadth < 0.30:
                breadth_scalar = 0.65
            elif breadth < 0.40:
                breadth_scalar = 0.85

            target_symbols = []
            # Adaptive position count based on regime + breadth:
            #   Full bull or recovery: 100% of max positions
            #   Early recovery (bear but bouncing): 70% — catch V-recoveries
            #   Bear (price < SMA50 & death cross): 40% — stay partially invested
            #   High vol (>40% annualized): 0% — extreme stress only
            #   Low breadth: further scale down
            if in_high_vol:
                effective_max = 0
            elif in_bull_now:
                effective_max = int(max_positions * breadth_scalar)
            elif in_early_recovery:
                # V-recovery detected: increase from 40% to 75% to catch bounce
                effective_max = int(max_positions * 0.75 * breadth_scalar)
            else:
                effective_max = int(max_positions * 0.40 * breadth_scalar)

            if effective_max > 0:
                sector_counts = {}
                for s in scored_stocks:
                    if s["mom_score"] <= 0:
                        break
                    if len(target_symbols) >= effective_max:
                        break
                    sym     = s["symbol"]
                    sector  = sector_map.get(sym)
                    if sector:
                        sec_cap = sector_max_override.get(sector, max_per_sector)
                        if sector_counts.get(sector, 0) >= sec_cap:
                            continue
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    target_symbols.append(sym)

            target_set = set(target_symbols)

            # Build a wider "hold set" for ranking hysteresis: existing positions
            # stay unless they drop below RANK_SELL_THRESHOLD rank.
            # This reduces churn from small rank fluctuations near the cutoff.
            rank_sell_threshold = getattr(config, "RANK_SELL_THRESHOLD", 0)
            if rank_sell_threshold > 0:
                hold_symbols = set()
                hold_sector_counts = {}
                for s in scored_stocks[:rank_sell_threshold]:
                    if s["mom_score"] <= 0:
                        break
                    sym = s["symbol"]
                    sector = sector_map.get(sym)
                    if sector:
                        sec_cap = sector_max_override.get(sector, max_per_sector)
                        if hold_sector_counts.get(sector, 0) >= sec_cap:
                            continue
                        hold_sector_counts[sector] = hold_sector_counts.get(sector, 0) + 1
                    hold_symbols.add(sym)
            else:
                hold_symbols = target_set

            # Step 3: Sell rank-droppers (using hysteresis threshold)
            for symbol in list(positions.keys()):
                if symbol == bear_etf:
                    continue
                if symbol not in hold_symbols:
                    current_price = _get_price(symbol, date_str)
                    if current_price is None:
                        continue
                    fill_price = current_price * (1 - slippage)
                    trade = _close_position(positions, symbol, fill_price, date_str, cash, "rank_drop")
                    cash += trade.qty * fill_price
                    all_trades.append(trade)
                    # Set re-entry cooldown after rank_drop exits
                    if REENTRY_COOLDOWN_DAYS > 0:
                        reentry_cooldown[symbol] = day_idx + REENTRY_COOLDOWN_DAYS
                    if verbose:
                        _print_trade(trade, "RANK DROP")

            # Step 4: Buy new additions
            max_new_per_rebal = getattr(config, "MAX_NEW_POSITIONS_PER_DAY", 10)
            max_sector_exposure = getattr(config, "MAX_SECTOR_EXPOSURE_PCT", 0.30)
            new_positions_today = 0

            for sym_info in scored_stocks:
                if new_positions_today >= max_new_per_rebal:
                    break
                symbol = sym_info["symbol"]
                if symbol not in target_set:
                    continue
                if symbol in positions:
                    continue

                # Re-entry cooldown: skip stocks recently sold via rank_drop
                if REENTRY_COOLDOWN_DAYS > 0 and reentry_cooldown.get(symbol, 0) > day_idx:
                    continue

                current_price = sym_info["price"]

                # Compute current portfolio value for sizing and sector checks
                portfolio_value = cash + sum(
                    p["qty"] * (_get_price(s, date_str) or 0)
                    for s, p in positions.items()
                )

                # Sector dollar-weighted exposure check
                sym_sector = sector_map.get(symbol)
                if sym_sector and portfolio_value > 0:
                    sector_dollar_exposure = 0.0
                    for s, p in positions.items():
                        if sector_map.get(s) == sym_sector:
                            s_price = _get_price(s, date_str)
                            if s_price:
                                sector_dollar_exposure += p["qty"] * s_price
                    # Estimate new position size to check if it would breach limit
                    est_new_dollars = portfolio_value * config.MAX_POSITION_SIZE_PCT
                    if (sector_dollar_exposure + est_new_dollars) / portfolio_value > max_sector_exposure:
                        continue  # skip — would exceed sector exposure limit

                # VIX-based position scaling
                vix_now = vix_by_date.get(date_str, 20.0)
                vix_mult = 1.0
                for vix_thresh, mult in sorted(
                    getattr(config, "VIX_SCALE_LEVELS", [(999, 1.0)]),
                    key=lambda x: x[0]
                ):
                    if vix_now < vix_thresh:
                        vix_mult = mult
                        break

                # Yield curve scaling
                yc_spread    = yc_by_date.get(date_str)
                yc_threshold = getattr(config, "YIELD_CURVE_BEARISH_THRESHOLD", -0.25)
                yc_mult      = 0.70 if (yc_spread is not None and yc_spread < yc_threshold) else 1.0

                # Inverse-volatility sizing: volatile stocks get smaller
                # positions so each contributes roughly equal risk.
                sym_indicators = sym_info.get("indicators", {})
                realized_vol = sym_indicators.get("realized_vol", 0.20)
                target_vol   = getattr(config, "INVERSE_VOL_TARGET_VOL", 0.20)
                if getattr(config, "USE_INVERSE_VOL_SIZING", True) and realized_vol > 0.01:
                    vol_scalar = target_vol / realized_vol
                    vol_scalar = min(max(vol_scalar, 0.3), 2.5)
                else:
                    vol_scalar = 1.0

                pos_pct = min(
                    config.MAX_POSITION_SIZE_PCT * vix_mult * yc_mult * vol_scalar * dd_scalar,
                    1.0 / max(config.MAX_OPEN_POSITIONS, 1)
                )

                dollar_amount = portfolio_value * pos_pct
                buy_price     = current_price * (1 + slippage)
                qty           = int(dollar_amount / buy_price)

                if qty > 0 and cash >= qty * buy_price:
                    cash -= qty * buy_price
                    positions[symbol] = {
                        "qty":         qty,
                        "entry_price": buy_price,
                        "entry_date":  date_str,
                        "highest_price": buy_price,
                    }
                    new_positions_today += 1
                    if verbose:
                        print(f"  {date_str} | BUY  {qty:>4} x {symbol:<5} "
                              f"@ ${buy_price:>8.2f} | mom: {sym_info['mom_score']:+.3f} | "
                              f"cost: ${qty * buy_price:>10,.2f}")

        # ── End-of-day equity ────────────────────────────────────────
        position_value = 0
        for sym, pos in positions.items():
            p = _get_price(sym, date_str)
            if p is not None:
                position_value += pos["qty"] * p

        equity = cash + position_value
        equity_curve.append({"date": date_str, "equity": equity})

        if prev_equity > 0:
            daily_returns.append((equity - prev_equity) / prev_equity)
        prev_equity = equity

        # Update drawdown scaling for next day's buys
        if equity > peak_equity:
            peak_equity = equity
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        dd_scalar = 1.0
        for dd_thresh, dd_mult in sorted(
            getattr(config, "DRAWDOWN_SCALE_LEVELS", [(1.0, 1.0)]),
            key=lambda x: x[0],
        ):
            if current_dd < dd_thresh:
                dd_scalar = dd_mult
                break

    # ── Close remaining positions at end ────────────────────────────
    last_date = all_dates[-1]
    for symbol in list(positions.keys()):
        if symbol in all_price_data:
            last_price = float(all_price_data[symbol]["Close"].iloc[-1])
            trade = _close_position(positions, symbol, last_price, last_date, cash, "backtest_end")
            cash += trade.qty * last_price
            all_trades.append(trade)

    # ── Compute metrics ──────────────────────────────────────────────
    ending_value = cash
    total_return = (ending_value - starting_cash) / starting_cash
    trading_days = len(all_dates) - warmup_days

    annualized   = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

    if daily_returns and len(daily_returns) > 1:
        avg_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe    = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
    else:
        sharpe = 0

    equity_values = [e["equity"] for e in equity_curve]
    peak, max_dd  = equity_values[0], 0.0
    for val in equity_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    wins          = [t for t in all_trades if t.pnl > 0]
    losses        = [t for t in all_trades if t.pnl < 0]
    win_rate      = len(wins) / len(all_trades) if all_trades else 0
    gross_profit  = sum(t.pnl for t in wins)
    gross_loss    = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win       = np.mean([t.pnl_pct for t in wins])   if wins   else 0
    avg_loss      = np.mean([t.pnl_pct for t in losses]) if losses else 0
    largest_win   = max((t.pnl_pct for t in wins),   default=0)
    largest_loss  = min((t.pnl_pct for t in losses), default=0)

    # ── SPY buy-and-hold benchmark ───────────────────────────────────
    spy_total_return = spy_ann = 0.0
    if spy_first_close and "SPY" in all_price_data:
        spy_last_close  = float(all_price_data["SPY"]["Close"].iloc[-1])
        spy_total_return= (spy_last_close - spy_first_close) / spy_first_close * 100
        n_active        = max(trading_days, 1)
        spy_ann         = ((1 + spy_total_return / 100) ** (252 / n_active) - 1) * 100

    result = BacktestResult(
        starting_cash        =starting_cash,
        ending_value         =ending_value,
        total_return_pct     =total_return * 100,
        annualized_return_pct=annualized   * 100,
        sharpe_ratio         =sharpe,
        max_drawdown_pct     =max_dd       * 100,
        total_trades         =len(all_trades),
        winning_trades       =len(wins),
        losing_trades        =len(losses),
        win_rate             =win_rate     * 100,
        profit_factor        =profit_factor,
        avg_win_pct          =avg_win      * 100,
        avg_loss_pct         =avg_loss     * 100,
        largest_win_pct      =largest_win  * 100,
        largest_loss_pct     =largest_loss * 100,
        spy_total_return_pct =spy_total_return,
        spy_annualized_pct   =spy_ann,
        trades               =all_trades,
        equity_curve         =equity_curve,
        daily_returns        =daily_returns,
    )

    if verbose:
        print_results(result)

    return result


def _close_position(
    positions: dict, symbol: str, exit_price: float,
    exit_date: str, cash: float, reason: str
) -> BacktestTrade:
    pos     = positions.pop(symbol)
    pnl     = (exit_price - pos["entry_price"]) * pos["qty"]
    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
    return BacktestTrade(
        symbol=symbol, entry_date=pos["entry_date"],
        entry_price=pos["entry_price"], exit_date=exit_date,
        exit_price=exit_price, qty=pos["qty"],
        pnl=pnl, pnl_pct=pnl_pct, exit_reason=reason,
    )


def _print_trade(trade: BacktestTrade, label: str):
    marker = "[W]" if trade.pnl >= 0 else "[L]"
    print(f"  {trade.exit_date} | {label:<12} {trade.qty:>4} x {trade.symbol:<5} "
          f"@ ${trade.exit_price:>8.2f} | P&L: ${trade.pnl:>+10,.2f} "
          f"({trade.pnl_pct:>+6.1%}) {marker}")


def print_results(result: BacktestResult):
    excess    = result.total_return_pct - result.spy_total_return_pct
    ann_excess= result.annualized_return_pct - result.spy_annualized_pct
    beat_icon = "+" if excess >= 0 else "-"

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Starting Capital:    ${result.starting_cash:>12,.2f}")
    print(f"  Ending Value:        ${result.ending_value:>12,.2f}")
    print(f"  {'─'*37}")
    print(f"  {'Metric':<22} {'Bot':>10}  {'SPY B&H':>10}  {'Excess':>8}")
    print(f"  {'─'*55}")
    print(f"  {'Total Return':<22} {result.total_return_pct:>+9.2f}%  "
          f"{result.spy_total_return_pct:>+9.2f}%  {excess:>+7.2f}%")
    print(f"  {'Annualized Return':<22} {result.annualized_return_pct:>+9.2f}%  "
          f"{result.spy_annualized_pct:>+9.2f}%  {ann_excess:>+7.2f}%")
    print(f"  {'─'*55}")
    print(f"  {'Sharpe Ratio':<22} {result.sharpe_ratio:>10.2f}")
    print(f"  {'Max Drawdown':<22} {result.max_drawdown_pct:>9.2f}%")
    print(f"  {'─'*55}")
    print(f"  {'Total Trades':<22} {result.total_trades:>10}")
    print(f"  {'Winning Trades':<22} {result.winning_trades:>10}")
    print(f"  {'Losing Trades':<22} {result.losing_trades:>10}")
    print(f"  {'Win Rate':<22} {result.win_rate:>9.1f}%")
    print(f"  {'Profit Factor':<22} {result.profit_factor:>10.2f}")
    print(f"  {'─'*55}")
    print(f"  {'Avg Win':<22} {result.avg_win_pct:>+9.2f}%")
    print(f"  {'Avg Loss':<22} {result.avg_loss_pct:>+9.2f}%")
    print(f"  {'Largest Win':<22} {result.largest_win_pct:>+9.2f}%")
    print(f"  {'Largest Loss':<22} {result.largest_loss_pct:>+9.2f}%")
    print(f"{'='*70}\n")

    if result.sharpe_ratio >= 1.5:
        verdict = "STRONG — Sharpe > 1.5 is excellent"
    elif result.sharpe_ratio >= 1.0:
        verdict = "GOOD — Sharpe > 1.0 beats most funds"
    elif result.sharpe_ratio >= 0.5:
        verdict = "DECENT — room for improvement"
    else:
        verdict = "WEAK — consider running: python tune.py"
    print(f"  Verdict: {verdict}")

    alpha_str = f"{ann_excess:+.1f}%/yr vs SPY"
    if ann_excess > 2:
        print(f"  Alpha:   {alpha_str}  (outperforming)")
    elif ann_excess > -2:
        print(f"  Alpha:   {alpha_str}  (roughly in line)")
    else:
        print(f"  Alpha:   {alpha_str}  (underperforming — try: python tune.py)")
    print()


def plot_equity_curve(result: BacktestResult, spy_data: pd.DataFrame = None,
                      output_path: str = "equity_curve.png"):
    """
    Save an equity curve chart comparing the bot vs SPY buy-and-hold.
    Requires matplotlib — skips gracefully if not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart. "
                       "Install with: pip install matplotlib")
        return

    dates  = [pd.Timestamp(e["date"]) for e in result.equity_curve]
    equity = [e["equity"]             for e in result.equity_curve]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bot equity
    ax.plot(dates, equity, label="Bot", color="#2196F3", linewidth=2)

    # SPY buy-and-hold — normalise index to tz-naive for comparison
    if spy_data is not None and not spy_data.empty and len(dates) > 0:
        spy_close = spy_data["Close"].copy()
        try:
            spy_idx = spy_close.index.tz_convert(None)
        except TypeError:
            spy_idx = spy_close.index.tz_localize(None)
        spy_close.index = spy_idx
        spy_start = float(spy_close.iloc[0])
        spy_values = []
        for d in dates:
            d_naive = d.tz_localize(None) if d.tzinfo is not None else d
            subset  = spy_close.loc[spy_close.index <= d_naive]
            val     = result.starting_cash * float(subset.iloc[-1]) / spy_start if len(subset) > 0 else result.starting_cash
            spy_values.append(val)
        ax.plot(dates, spy_values, label="SPY B&H", color="#FF9800",
                linewidth=1.5, linestyle="--", alpha=0.85)

    # Starting capital reference line
    ax.axhline(result.starting_cash, color="#888", linewidth=0.8,
               linestyle=":", label="Starting capital")

    # Shade drawdown periods
    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    dd_pct = (peak - eq_arr) / peak
    in_dd  = False
    dd_start = None
    for i, (d, dp) in enumerate(zip(dates, dd_pct)):
        if dp > 0.02 and not in_dd:
            in_dd    = True
            dd_start = d
        elif dp <= 0.01 and in_dd:
            ax.axvspan(dd_start, d, alpha=0.07, color="red")
            in_dd = False
    if in_dd and dd_start:
        ax.axvspan(dd_start, dates[-1], alpha=0.07, color="red")

    ax.set_title(f"Bot vs SPY Buy-and-Hold\n"
                 f"Bot: {result.total_return_pct:+.1f}%  |  "
                 f"SPY B&H: {result.spy_total_return_pct:+.1f}%  |  "
                 f"Sharpe: {result.sharpe_ratio:.2f}  |  "
                 f"Max DD: {result.max_drawdown_pct:.1f}%",
                 fontsize=11)
    ax.set_ylabel("Portfolio Value ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved to: {output_path}")
