"""
20-year validation backtest using precomputed signals.

Uses tune.py's precompute_signals() for O(N) signal computation, then
runs a full-featured backtest (trailing stops, sector caps, drawdown
scaling, breadth, inverse-vol sizing) with O(1) signal lookups.
"""
import sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import config
from data_fetcher import fetch_all_price_data, fetch_vix_history
from tune import precompute_signals

DAYS_20Y = 365 * 20
WARMUP = 252
STARTING_CASH = 100_000

print(f"Fetching 20 years of data for {len(config.WATCHLIST)} symbols...")
sys.stdout.flush()

t0 = time.time()
all_data = fetch_all_price_data(config.WATCHLIST, days=DAYS_20Y)
print(f"Fetch done in {time.time()-t0:.0f}s — {len(all_data)} symbols")
for sym in ["SPY", "AAPL", "NVDA", "CRWD"]:
    if sym in all_data:
        df = all_data[sym]
        print(f"  {sym}: {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} days)")
sys.stdout.flush()

# ── Precompute all signals once ─────────────────────────────────────
print("\nPrecomputing signals...")
sys.stdout.flush()
t0 = time.time()
spy_raw = all_data.get("SPY")
precomputed = {}
for sym, df in all_data.items():
    try:
        precomputed[sym] = precompute_signals(df, spy_df=spy_raw if sym != "SPY" else None)
    except Exception:
        pass
print(f"Precomputed in {time.time()-t0:.0f}s — {sum(len(v) for v in precomputed.values())} rows")
sys.stdout.flush()

# ── Fetch VIX history ──────────────────────────────────────────────
vix_by_date = {}
try:
    vix_s = fetch_vix_history(days=DAYS_20Y + 60)
    for dt, val in vix_s.items():
        vix_by_date[pd.Timestamp(dt).strftime("%Y-%m-%d")] = float(val)
except Exception:
    pass

# ── Pre-compute SPY regime ─────────────────────────────────────────
spy_bull = {}
spy_high_vol = {}
spy_early_recovery = {}
spy_micro_recovery = {}
if "SPY" in precomputed:
    spy_df = precomputed["SPY"]
    vol_10 = spy_df["price"].pct_change().rolling(10).std() * np.sqrt(252)
    spy_peak = spy_df["price"].expanding().max()
    spy_dd = (spy_peak - spy_df["price"]) / spy_peak
    spy_ret5  = spy_df["price"].pct_change(5)
    spy_ret10 = spy_df["price"].pct_change(10)
    spy_ret21 = spy_df["price"].pct_change(21)
    for dt in spy_df.index:
        dk = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        spy_bull[dk] = float(spy_df.at[dt, "bull_regime"]) > 0.5
        v = vol_10.get(dt, np.nan) if dt in vol_10.index else np.nan
        spy_high_vol[dk] = bool(not pd.isna(v) and v > 0.40)
        dd_val = spy_dd.get(dt, 0) if dt in spy_dd.index else 0
        r5  = spy_ret5.get(dt, 0)  if dt in spy_ret5.index  else 0
        r10 = spy_ret10.get(dt, 0) if dt in spy_ret10.index else 0
        r21 = spy_ret21.get(dt, 0) if dt in spy_ret21.index else 0
        if pd.isna(dd_val): dd_val = 0
        if pd.isna(r5): r5 = 0
        if pd.isna(r10): r10 = 0
        if pd.isna(r21): r21 = 0
        spy_micro_recovery[dk] = bool(dd_val > 0.10 and r5 > 0.02)
        spy_early_recovery[dk] = bool(dd_val > 0.12 and r10 > 0.015 and r21 > 0)

# ── Build date list from SPY (broadest coverage) ──────────────────
# Use SPY's dates, not intersection — newer IPOs shouldn't shrink the range.
# Each stock is only scored on dates where it has data.
if "SPY" in precomputed:
    all_dates = sorted(precomputed["SPY"].index)
else:
    all_dates = sorted(set.union(*[set(df.index) for df in precomputed.values()]))
print(f"Date range: {len(all_dates)} trading days")
sys.stdout.flush()

# ── Config ─────────────────────────────────────────────────────────
REBALANCE_EVERY    = getattr(config, "REBALANCE_EVERY", 5)
MAX_POSITIONS      = config.MAX_OPEN_POSITIONS
MAX_POS_PCT        = config.MAX_POSITION_SIZE_PCT
SLIPPAGE           = getattr(config, "SLIPPAGE_PCT", 0.0005)
TRAILING_TIERS     = getattr(config, "TRAILING_STOP_TIERS", [(0.10, 0.25)])
ATR_MULT           = getattr(config, "ATR_STOP_MULTIPLIER", 3.0)
ATR_MIN            = getattr(config, "ATR_STOP_MIN_PCT", 0.08)
ATR_MAX            = getattr(config, "ATR_STOP_MAX_PCT", 0.20)
SECTOR_MAP         = getattr(config, "SECTOR_MAP", {})
MAX_PER_SECTOR     = getattr(config, "MAX_POSITIONS_PER_SECTOR", 5)
SECTOR_OVERRIDE    = getattr(config, "SECTOR_MAX_OVERRIDE", {})
RANK_SELL          = getattr(config, "RANK_SELL_THRESHOLD", 24)
COOLDOWN           = getattr(config, "REENTRY_COOLDOWN_DAYS", 12)
MAX_NEW_PER_DAY    = getattr(config, "MAX_NEW_POSITIONS_PER_DAY", 4)
MAX_SECTOR_EXP     = getattr(config, "MAX_SECTOR_EXPOSURE_PCT", 0.40)
REGIME_ONLY        = getattr(config, "REGIME_ONLY_SYMBOLS", {"SPY", "QQQ", "BIL"})
BEAR_ETF           = getattr(config, "BEAR_REGIME_ETF", "BIL")
ranking_weights    = getattr(config, "RANKING_WEIGHTS", {})
VIX_LEVELS         = getattr(config, "VIX_SCALE_LEVELS", [(999, 1.0)])
DD_LEVELS          = getattr(config, "DRAWDOWN_SCALE_LEVELS", [(1.0, 1.0)])

# ── Main backtest loop ─────────────────────────────────────────────
print(f"\nRunning backtest: {len(all_dates)} days, rebal every {REBALANCE_EVERY}d...")
sys.stdout.flush()
t0 = time.time()

cash = STARTING_CASH
positions = {}
trades = []
equity_curve = []
reentry_cooldown = {}
peak_equity = STARTING_CASH
dd_scalar = 1.0

for day_idx, dt in enumerate(all_dates):
    dk = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]

    if day_idx < WARMUP:
        equity_curve.append({"date": dk, "equity": STARTING_CASH})
        continue

    in_bull = spy_bull.get(dk, True)
    in_hvol = spy_high_vol.get(dk, False)
    in_early_rec = spy_early_recovery.get(dk, False)

    # ── Phase 1: Catastrophe stop (daily) ──────────────────────────
    for sym in list(positions.keys()):
        if sym == BEAR_ETF:
            continue
        sig_df = precomputed.get(sym)
        if sig_df is None or dt not in sig_df.index:
            continue
        cur = float(sig_df.at[dt, "price"])
        pos = positions[sym]
        pnl_pct = (cur - pos["entry_price"]) / pos["entry_price"]
        if pnl_pct <= -0.20:
            fill = cur * (1 - SLIPPAGE)
            cash += pos["qty"] * fill
            trades.append({"pnl": (fill - pos["entry_price"]) * pos["qty"],
                           "pnl_pct": (fill - pos["entry_price"]) / pos["entry_price"]})
            del positions[sym]

    # ── Phase 1b: Trailing stop (daily) ────────────────────────────
    for sym in list(positions.keys()):
        if sym == BEAR_ETF or sym not in positions:
            continue
        sig_df = precomputed.get(sym)
        if sig_df is None or dt not in sig_df.index:
            continue
        cur = float(sig_df.at[dt, "price"])
        pos = positions[sym]
        if cur > pos["highest_price"]:
            pos["highest_price"] = cur

        gain = (pos["highest_price"] - pos["entry_price"]) / pos["entry_price"]
        trail_dist = None
        for tier_gain, tier_trail in TRAILING_TIERS:
            if gain >= tier_gain:
                trail_dist = tier_trail
                break
        if trail_dist is None:
            continue

        # ATR stop
        atr_pct_val = float(sig_df.at[dt, "atr_pct"]) if "atr_pct" in sig_df.columns else 0.02
        atr_stop = max(ATR_MIN, min(atr_pct_val * ATR_MULT, ATR_MAX))
        eff_stop = min(trail_dist, atr_stop)

        drop = (pos["highest_price"] - cur) / pos["highest_price"]
        if drop >= eff_stop:
            fill = cur * (1 - SLIPPAGE)
            cash += pos["qty"] * fill
            trades.append({"pnl": (fill - pos["entry_price"]) * pos["qty"],
                           "pnl_pct": (fill - pos["entry_price"]) / pos["entry_price"]})
            del positions[sym]

    # ── Phase 2: Rebalance ─────────────────────────────────────────
    tday = day_idx - WARMUP
    if tday % REBALANCE_EVERY == 0:
        # BIL parking
        if BEAR_ETF and BEAR_ETF in precomputed and dt in precomputed[BEAR_ETF].index:
            bil_p = float(precomputed[BEAR_ETF].at[dt, "price"])
            if (in_bull or not in_hvol) and BEAR_ETF in positions:
                cash += positions[BEAR_ETF]["qty"] * bil_p
                del positions[BEAR_ETF]
            elif in_hvol and BEAR_ETF not in positions and cash > 1000:
                qty = int(cash * 0.90 / bil_p)
                if qty > 0:
                    cash -= qty * bil_p
                    positions[BEAR_ETF] = {"qty": qty, "entry_price": bil_p,
                                           "highest_price": bil_p}

        # Compute sector momentum (Moskowitz & Grinblatt 1999)
        sector_avg_mom = {}
        for sym, sig_df in precomputed.items():
            if sym in REGIME_ONLY or sym == BEAR_ETF:
                continue
            if dt not in sig_df.index:
                continue
            sec = SECTOR_MAP.get(sym)
            if sec and "momentum_6m" in sig_df.columns:
                m6 = sig_df.at[dt, "momentum_6m"]
                if not pd.isna(m6):
                    sector_avg_mom.setdefault(sec, []).append(float(m6))
        sector_avg_mom = {s: np.mean(v) for s, v in sector_avg_mom.items()}

        # Score all stocks
        scored = []
        for sym, sig_df in precomputed.items():
            if sym in REGIME_ONLY or sym == BEAR_ETF:
                continue
            if dt not in sig_df.index:
                continue
            row = sig_df.loc[dt]
            if pd.isna(row.get("sma_crossover", np.nan)):
                continue

            # Inject sector momentum into row for scoring
            sec = SECTOR_MAP.get(sym)
            sec_mom_val = sector_avg_mom.get(sec, 0.0) if sec else 0.0

            avail_wt = sum(w for k, w in ranking_weights.items()
                           if (k in row.index and not pd.isna(row.get(k))) or k == "sector_momentum")
            if avail_wt > 0:
                score = 0.0
                for k, w in ranking_weights.items():
                    if k == "sector_momentum":
                        score += float(np.clip(sec_mom_val * 3, -1.0, 1.0)) * w
                    elif k in row.index and not pd.isna(row.get(k)):
                        score += float(row.get(k, 0)) * w
                    else:
                        avail_wt -= w  # remove missing factor weight
                score = score / avail_wt if avail_wt > 0 else 0.0
            else:
                score = 0.0

            scored.append({
                "sym": sym, "score": score,
                "price": float(row["price"]),
                "realized_vol": float(row.get("realized_vol", 0.20)),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Breadth
        breadth = sum(1 for s in scored if s["score"] > 0) / len(scored) if scored else 0.5
        b_sc = 1.0
        if breadth < 0.20: b_sc = 0.40
        elif breadth < 0.30: b_sc = 0.65
        elif breadth < 0.40: b_sc = 0.85

        if in_hvol:
            eff_max = 0
        elif in_bull:
            eff_max = int(MAX_POSITIONS * b_sc)
        elif in_early_rec:
            # V-recovery: increase from 40% to 75% to catch bounce
            eff_max = int(MAX_POSITIONS * 0.75 * b_sc)
        else:
            eff_max = int(MAX_POSITIONS * 0.40 * b_sc)

        # Target portfolio with sector caps
        target = []
        sector_counts = {}
        for s in scored:
            if s["score"] <= 0:
                break
            if len(target) >= eff_max:
                break
            sec = SECTOR_MAP.get(s["sym"])
            if sec:
                cap = SECTOR_OVERRIDE.get(sec, MAX_PER_SECTOR)
                if sector_counts.get(sec, 0) >= cap:
                    continue
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            target.append(s["sym"])
        target_set = set(target)

        # Hysteresis hold set
        if RANK_SELL > 0:
            hold = set()
            h_sec = {}
            for s in scored[:RANK_SELL]:
                if s["score"] <= 0:
                    break
                sec = SECTOR_MAP.get(s["sym"])
                if sec:
                    cap = SECTOR_OVERRIDE.get(sec, MAX_PER_SECTOR)
                    if h_sec.get(sec, 0) >= cap:
                        continue
                    h_sec[sec] = h_sec.get(sec, 0) + 1
                hold.add(s["sym"])
        else:
            hold = target_set

        # Sell rank-droppers
        for sym in list(positions.keys()):
            if sym == BEAR_ETF:
                continue
            if sym not in hold:
                sig_df = precomputed.get(sym)
                if sig_df is None or dt not in sig_df.index:
                    continue
                p = float(sig_df.at[dt, "price"]) * (1 - SLIPPAGE)
                pos = positions[sym]
                cash += pos["qty"] * p
                trades.append({"pnl": (p - pos["entry_price"]) * pos["qty"],
                               "pnl_pct": (p - pos["entry_price"]) / pos["entry_price"]})
                del positions[sym]
                if COOLDOWN > 0:
                    reentry_cooldown[sym] = day_idx + COOLDOWN

        # Buy new
        new_buys = 0
        for si in scored:
            if new_buys >= MAX_NEW_PER_DAY:
                break
            sym = si["sym"]
            if sym not in target_set or sym in positions:
                continue
            if COOLDOWN > 0 and reentry_cooldown.get(sym, 0) > day_idx:
                continue

            price = si["price"]
            pv = cash + sum(
                positions[s]["qty"] * float(precomputed[s].at[dt, "price"])
                for s in positions if s in precomputed and dt in precomputed[s].index
            )

            # Sector exposure check
            sec = SECTOR_MAP.get(sym)
            if sec and pv > 0:
                sec_exp = sum(
                    positions[s]["qty"] * float(precomputed[s].at[dt, "price"])
                    for s in positions
                    if SECTOR_MAP.get(s) == sec and s in precomputed and dt in precomputed[s].index
                )
                if (sec_exp + pv * MAX_POS_PCT) / pv > MAX_SECTOR_EXP:
                    continue

            # VIX scaling
            vix_now = vix_by_date.get(dk, 20.0)
            vix_m = 1.0
            for vt, vm in sorted(VIX_LEVELS, key=lambda x: x[0]):
                if vix_now < vt:
                    vix_m = vm
                    break

            # Inverse-vol sizing
            rv = si["realized_vol"]
            tv = getattr(config, "INVERSE_VOL_TARGET_VOL", 0.20)
            if rv > 0.01:
                vs = min(max(tv / rv, 0.3), 2.5)
            else:
                vs = 1.0

            pos_pct = min(MAX_POS_PCT * vix_m * vs * dd_scalar,
                          1.0 / max(MAX_POSITIONS, 1))
            dollar = pv * pos_pct
            buy_p = price * (1 + SLIPPAGE)
            qty = int(dollar / buy_p)
            if qty > 0 and cash >= qty * buy_p:
                cash -= qty * buy_p
                positions[sym] = {"qty": qty, "entry_price": buy_p,
                                  "highest_price": buy_p}
                new_buys += 1

    # ── End-of-day equity ──────────────────────────────────────────
    pos_val = sum(
        positions[s]["qty"] * float(precomputed[s].at[dt, "price"])
        for s in positions if s in precomputed and dt in precomputed[s].index
    )
    equity = cash + pos_val
    equity_curve.append({"date": dk, "equity": equity})

    if equity > peak_equity:
        peak_equity = equity
    cur_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
    dd_scalar = 1.0
    for ddt, ddm in sorted(DD_LEVELS, key=lambda x: x[0]):
        if cur_dd < ddt:
            dd_scalar = ddm
            break

elapsed = time.time() - t0
print(f"Backtest done in {elapsed:.0f}s")
sys.stdout.flush()

# ── Close remaining positions ──────────────────────────────────────
last_dt = all_dates[-1]
for sym, pos in list(positions.items()):
    if sym in precomputed and last_dt in precomputed[sym].index:
        p = float(precomputed[sym].at[last_dt, "price"])
        trades.append({"pnl": (p - pos["entry_price"]) * pos["qty"],
                       "pnl_pct": (p - pos["entry_price"]) / pos["entry_price"]})
        cash += pos["qty"] * p
positions.clear()

# ── Compute metrics ────────────────────────────────────────────────
ending = cash
total_ret = (ending - STARTING_CASH) / STARTING_CASH
n_trading = len(equity_curve) - WARMUP
ann = ((1 + total_ret) ** (252 / max(n_trading, 1)) - 1) if n_trading > 0 else 0

eq_arr = np.array([e["equity"] for e in equity_curve[WARMUP:]], dtype=float)
daily_r = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1)
std_d = float(np.std(daily_r))
sharpe = float(np.mean(daily_r) / std_d * np.sqrt(252)) if std_d > 0 else 0

peak = np.maximum.accumulate(eq_arr)
max_dd = float(np.max((peak - eq_arr) / np.where(peak > 0, peak, 1)))

wins = [t for t in trades if t["pnl"] > 0]
losses = [t for t in trades if t["pnl"] < 0]
win_rate = len(wins) / len(trades) * 100 if trades else 0
gross_profit = sum(t["pnl"] for t in wins)
gross_loss = abs(sum(t["pnl"] for t in losses))
pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

# SPY benchmark
spy_first = float(precomputed["SPY"].at[all_dates[WARMUP], "price"]) if "SPY" in precomputed else 0
spy_last = float(precomputed["SPY"].at[all_dates[-1], "price"]) if "SPY" in precomputed else 0
spy_ret = (spy_last / spy_first - 1) if spy_first > 0 else 0
spy_ann = ((1 + spy_ret) ** (252 / max(n_trading, 1)) - 1) * 100

years = n_trading / 252
excess = ann * 100 - spy_ann

print(f"\n{'='*70}")
print(f"  20-YEAR VALIDATION BACKTEST")
print(f"{'='*70}")
print(f"  Period:              {equity_curve[WARMUP]['date']} -> {equity_curve[-1]['date']}")
print(f"  Duration:            {years:.1f} years ({n_trading} trading days)")
print(f"  Starting Capital:    ${STARTING_CASH:>12,.2f}")
print(f"  Ending Value:        ${ending:>12,.2f}")
print(f"  {'─'*55}")
print(f"  {'Metric':<22} {'Bot':>10}  {'SPY B&H':>10}  {'Excess':>8}")
print(f"  {'─'*55}")
print(f"  {'Total Return':<22} {total_ret*100:>+9.1f}%  {spy_ret*100:>+9.1f}%  {total_ret*100 - spy_ret*100:>+7.1f}%")
print(f"  {'Annualized Return':<22} {ann*100:>+9.2f}%  {spy_ann:>+9.2f}%  {excess:>+7.2f}%")
print(f"  {'─'*55}")
print(f"  {'Sharpe Ratio':<22} {sharpe:>10.2f}")
print(f"  {'Max Drawdown':<22} {max_dd*100:>9.2f}%")
print(f"  {'─'*55}")
print(f"  {'Total Trades':<22} {len(trades):>10}")
print(f"  {'Trades/Year':<22} {len(trades)/max(years,1):>10.0f}")
print(f"  {'Win Rate':<22} {win_rate:>9.1f}%")
print(f"  {'Profit Factor':<22} {pf:>10.2f}")
avg_win = np.mean([t["pnl_pct"] for t in wins]) * 100 if wins else 0
avg_loss = np.mean([t["pnl_pct"] for t in losses]) * 100 if losses else 0
print(f"  {'Avg Win':<22} {avg_win:>+9.2f}%")
print(f"  {'Avg Loss':<22} {avg_loss:>+9.2f}%")
print(f"{'='*70}\n")

# Year-by-year
print(f"  Year-by-year returns:")
print(f"  {'Year':<6} {'Bot':>8}  {'SPY':>8}  {'Excess':>8}")
print(f"  {'─'*35}")
eq = equity_curve
prev_val = eq[WARMUP]["equity"]
prev_year = eq[WARMUP]["date"][:4]
spy_prev = spy_first

for e in eq[WARMUP:]:
    yr = e["date"][:4]
    if yr != prev_year:
        yr_ret = (e["equity"] / prev_val - 1) * 100
        # SPY return for this year
        spy_p = 0
        for s_e in eq[WARMUP:]:
            if s_e["date"][:4] == yr and "SPY" in precomputed:
                spy_idx_list = [i for i in precomputed["SPY"].index
                                if i.strftime("%Y-%m-%d") <= s_e["date"]]
                if spy_idx_list:
                    spy_p = float(precomputed["SPY"].at[spy_idx_list[-1], "price"])
                break
        if spy_p > 0 and spy_prev > 0:
            spy_yr = (spy_p / spy_prev - 1) * 100
        else:
            spy_yr = 0
        print(f"    {prev_year}  {yr_ret:>+7.1f}%  {spy_yr:>+7.1f}%  {yr_ret - spy_yr:>+7.1f}%")
        prev_val = e["equity"]
        spy_prev = spy_p if spy_p > 0 else spy_prev
        prev_year = yr

# Final partial year
final = eq[-1]
yr_ret = (final["equity"] / prev_val - 1) * 100
print(f"    {prev_year}  {yr_ret:>+7.1f}%  {'--':>8}  {'--':>8}  [partial]")

print()
sys.stdout.flush()
