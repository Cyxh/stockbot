"""
MAIN.PY — Entry point for the trading bot.

USAGE:
    python main.py backtest     Run backtest on historical data (+ saves equity_curve.png)
    python main.py scan         One-time scan — shows what the bot would do NOW
    python main.py live         Start live/paper trading loop
    python main.py tune         Walk-forward optimizer (finds best parameters)

Start with 'backtest', then 'tune' if results are weak, then 'scan', then 'live'.
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Fix Windows console UTF-8 so emoji characters don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

ET = ZoneInfo("America/New_York")

import config
import data_fetcher
from strategy import generate_signal
from backtester import run_backtest, plot_equity_curve
from trader import Trader
from report import generate_daily_report

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger("main")


# ── Market calendar helpers ──────────────────────────────────────────────────

def _is_market_holiday(dt: datetime) -> bool:
    """
    Returns True if dt is a US market holiday (NYSE closed).
    Uses pandas-market-calendars if installed; otherwise returns False
    (weekend check in the caller still applies).
    """
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(
            start_date=dt.strftime("%Y-%m-%d"),
            end_date  =dt.strftime("%Y-%m-%d"),
        )
        return sched.empty  # Empty schedule = closed that day
    except Exception:
        return False  # Can't determine — assume open


def _next_market_open(now: datetime) -> datetime:
    """Return the next 9:30 AM ET open day (skipping weekends and holidays)."""
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= candidate:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5 or _is_market_holiday(candidate):
        candidate += timedelta(days=1)
    return candidate


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_backtest(save_chart: bool = True):
    print("\n" + "="*70)
    print("  STOCK BOT — BACKTEST MODE")
    print(f"  Testing on {config.LOOKBACK_DAYS} days of historical data")
    print("="*70)

    print("\n📥 Fetching historical price data...")
    all_data = data_fetcher.fetch_all_price_data(config.WATCHLIST, days=config.LOOKBACK_DAYS)

    if not all_data:
        print("❌ No data fetched. Check your internet connection.")
        return

    print(f"✅ Got data for {len(all_data)} stocks\n")

    result = run_backtest(all_data, verbose=True)

    # Trade breakdown
    if result.trades:
        print(f"\n{'─'*70}")
        print(f"  TRADE LOG ({len(result.trades)} trades)")
        print(f"{'─'*70}")
        for t in result.trades:
            emoji = "🟢" if t.pnl >= 0 else "🔴"
            print(f"  {emoji} {t.symbol:<8} | "
                  f"In: {t.entry_date[:10]} @ ${t.entry_price:.2f} → "
                  f"Out: {t.exit_date[:10]} @ ${t.exit_price:.2f} | "
                  f"P&L: ${t.pnl:+,.2f} ({t.pnl_pct:+.1%}) | "
                  f"{t.exit_reason}")

    # Save chart
    if save_chart:
        spy_df = all_data.get("SPY", pd.DataFrame())
        plot_equity_curve(result, spy_data=spy_df, output_path="equity_curve.png")

    return result


def cmd_scan():
    print("\n" + "="*70)
    print("  STOCK BOT — MARKET SCAN")
    print(f"  Scanning {len(config.WATCHLIST)} stocks...")
    print("="*70 + "\n")

    all_prices = data_fetcher.fetch_all_price_data()
    all_news   = data_fetcher.fetch_all_news()
    spy_data   = all_prices.get("SPY")

    signals = []
    for symbol in config.WATCHLIST:
        if symbol not in all_prices:
            continue
        signal = generate_signal(
            symbol=symbol,
            price_data=all_prices[symbol],
            articles=all_news.get(symbol, []),
            benchmark_data=spy_data if symbol != "SPY" else None,
        )
        signals.append(signal)

    signals.sort(key=lambda s: abs(s.combined_score), reverse=True)

    print(f"{'Symbol':<7} {'Action':<6} {'Confidence':>10} "
          f"{'Tech':>8} {'Sent':>8} {'Combined':>9}")
    print("─" * 60)

    for sig in signals:
        color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig.action, "⚪")
        print(f"  {color} {sig.symbol:<5} {sig.action:<6} "
              f"{sig.confidence:>9.1%} "
              f"{sig.technical_score:>+7.3f} "
              f"{sig.sentiment_score:>+7.3f} "
              f"{sig.combined_score:>+8.3f}")

    actionable = [s for s in signals if s.action != "HOLD"]
    if actionable:
        print(f"\n{'─'*60}")
        print("  ACTIONABLE SIGNALS:\n")
        for sig in actionable:
            print(f"  {sig.symbol} — {sig.action}")
            for line in sig.reasoning:
                print(f"    {line}")
            print()
    else:
        print("\n  No actionable signals right now. All HOLD.")


def cmd_live():
    trader = Trader()
    mode   = "PAPER" if config.PAPER_TRADING else "⚠️  LIVE"

    print("\n" + "="*70)
    print(f"  STOCK BOT — {mode} TRADING MODE")
    print(f"  Portfolio: ${trader.get_portfolio_value():,.2f}")
    print(f"  Watching: {', '.join(config.WATCHLIST)}")
    print("="*70)

    if not config.PAPER_TRADING:
        print("\n  ⚠️  REAL MONEY MODE — Ctrl+C to abort\n")
        for i in range(10, 0, -1):
            print(f"  Starting in {i}...", end="\r")
            time.sleep(1)

    scan_interval   = 300  # 5 minutes
    cooldown_symbols= {}   # symbol -> datetime when cooldown expires
    cached_news     = {}   # reuse news between scans (NewsAPI free = 100 req/day)
    last_news_fetch = None # datetime of last news fetch
    report_sent_today = None  # date when today's report was sent

    while True:
        try:
            now = datetime.now(ET)

            # Skip weekends
            if now.weekday() >= 5:
                next_open  = _next_market_open(now)
                sleep_secs = max((next_open - now).total_seconds(), 60)
                logger.info(f"Weekend. Sleeping until {next_open.strftime('%a %H:%M ET')} "
                            f"({sleep_secs/3600:.1f} hrs)...")
                time.sleep(min(sleep_secs, 3600))
                continue

            # Skip market holidays
            if _is_market_holiday(now):
                next_open  = _next_market_open(now)
                sleep_secs = max((next_open - now).total_seconds(), 60)
                logger.info(f"Market holiday. Next open: {next_open.strftime('%a %b %d %H:%M ET')}")
                time.sleep(min(sleep_secs, 3600))
                continue

            # Market hours: 9:30–16:00 ET
            market_open  = (now.hour > 9) or (now.hour == 9 and now.minute >= 30)
            market_close = now.hour >= 16

            if not market_open or market_close:
                # Generate daily report after market close (once per day)
                if market_close and report_sent_today != now.date():
                    try:
                        generate_daily_report(trader)
                        report_sent_today = now.date()
                    except Exception as e:
                        logger.error(f"Failed to generate report: {e}")

                next_open  = _next_market_open(now)
                sleep_secs = min((next_open - now).total_seconds(), scan_interval)
                logger.info(f"Market closed ({now.strftime('%H:%M ET')}). "
                            f"Next open: {next_open.strftime('%a %H:%M ET')}.")
                time.sleep(max(sleep_secs, 60))
                continue

            logger.info("Running market scan...")

            live_days = getattr(config, "LIVE_LOOKBACK_DAYS", 365)
            all_prices = data_fetcher.fetch_all_price_data(days=live_days)

            # Refresh news at most once per hour (NewsAPI free tier = 100 req/day)
            if last_news_fetch is None or (now - last_news_fetch).total_seconds() > 3600:
                cached_news = data_fetcher.fetch_all_news()
                last_news_fetch = now
            all_news = cached_news

            trader.check_stop_loss_take_profit()

            # Update cooldown set from trader's recent stop-loss exits
            for record in trader.get_trade_log():
                if record.get("reason") == "stop_loss":
                    sym = record.get("symbol")
                    ts  = record.get("timestamp", "")
                    if sym and ts:
                        try:
                            exit_dt = datetime.fromisoformat(ts)
                            cooldown_until = exit_dt + timedelta(
                                days=getattr(config, "COOLDOWN_DAYS", 0)
                            )
                            cooldown_symbols[sym] = cooldown_until
                        except Exception:
                            pass

            active_cooldowns = {
                s for s, until in cooldown_symbols.items()
                if datetime.now() < until
            }

            # SPY market regime
            in_bull_regime = True
            if "SPY" in all_prices:
                spy_close = all_prices["SPY"]["Close"]
                if len(spy_close) >= 200:
                    sma50  = spy_close.rolling(50).mean().iloc[-1]
                    sma200 = spy_close.rolling(200).mean().iloc[-1]
                    if not (pd.isna(sma50) or pd.isna(sma200)):
                        in_bull_regime = bool(sma50 > sma200)

            # Bear-regime BIL parking
            bear_etf = getattr(config, "BEAR_REGIME_ETF", None)
            if bear_etf and bear_etf in all_prices:
                positions_now   = trader.get_positions()
                bil_in_portfolio= bear_etf in positions_now
                if in_bull_regime and bil_in_portfolio:
                    trader.sell(bear_etf, reason="regime_bull")
                    logger.info("Sold BIL: market regime turned bull")
                elif not in_bull_regime and not bil_in_portfolio:
                    account_cash = float(trader.client.get_account().cash) if trader.client else 0
                    if account_cash > 1000:
                        bil_price    = float(all_prices[bear_etf]["Close"].iloc[-1])
                        dollar_amount= account_cash * 0.90
                        trader.buy(bear_etf, dollar_amount, bil_price)
                        logger.info(f"Bought BIL: bear regime, parking ${dollar_amount:,.0f}")

            portfolio_value = trader.get_portfolio_value()
            positions       = trader.get_positions()

            spy_data    = all_prices.get("SPY")
            regime_only = getattr(config, "REGIME_ONLY_SYMBOLS", {"SPY", "QQQ", "BIL"})

            # ── Cross-sectional ranking: score all stocks first ──────
            # This aligns live trading with the backtester — buy the BEST
            # stocks, not the first ones alphabetically in the watchlist.

            # Pre-compute sector momentum (Moskowitz & Grinblatt 1999)
            sector_map = getattr(config, "SECTOR_MAP", {})
            sector_returns = {}
            for sym in config.WATCHLIST:
                if sym in regime_only or sym not in all_prices:
                    continue
                df = all_prices[sym]
                if len(df) >= 126:
                    ret_6m = float(df["Close"].iloc[-1] / df["Close"].iloc[-126] - 1)
                    sec = sector_map.get(sym)
                    if sec:
                        sector_returns.setdefault(sec, []).append(ret_6m)
            import numpy as _np
            sector_avg_mom = {s: _np.mean(v) for s, v in sector_returns.items()}

            all_signals = []
            for symbol in config.WATCHLIST:
                if symbol not in all_prices:
                    continue
                if symbol in regime_only:
                    continue

                # Sector momentum for this stock
                sym_sector = sector_map.get(symbol)
                sec_mom = sector_avg_mom.get(sym_sector) if sym_sector else None

                signal = generate_signal(
                    symbol=symbol,
                    price_data=all_prices[symbol],
                    articles=all_news.get(symbol, []),
                    portfolio_value=portfolio_value,
                    current_positions=positions,
                    in_bull_regime=in_bull_regime,
                    cooldown_symbols=active_cooldowns,
                    benchmark_data=spy_data if symbol != "SPY" else None,
                    sector_momentum=sec_mom,
                )
                all_signals.append(signal)

            # Process SELLs first (free up capital), then BUYs ranked by strength
            for signal in all_signals:
                if signal.action == "SELL":
                    trader.sell(signal.symbol, reason="strategy_signal")
                    logger.info(f"Executed SELL for {signal.symbol}")
                    for line in signal.reasoning:
                        logger.info(f"  {line}")
                elif signal.action == "PUT":
                    current_price = float(all_prices[signal.symbol]["Close"].iloc[-1])
                    dollar_amount = portfolio_value * config.OPTIONS["max_position_pct"]
                    trader.buy_option(signal.symbol, "put", dollar_amount, current_price)
                    logger.info(f"Executed PUT for {signal.symbol}")

            # Sort BUYs by combined_score descending — strongest signals first
            buy_signals = sorted(
                [s for s in all_signals if s.action in ("BUY", "CALL")],
                key=lambda s: s.combined_score,
                reverse=True,
            )
            for signal in buy_signals:
                current_price = float(all_prices[signal.symbol]["Close"].iloc[-1])
                if signal.action == "BUY":
                    dollar_amount = portfolio_value * signal.position_size_pct
                    trader.buy(signal.symbol, dollar_amount, current_price)
                    logger.info(f"Executed BUY for {signal.symbol}")
                    for line in signal.reasoning:
                        logger.info(f"  {line}")
                elif signal.action == "CALL":
                    dollar_amount = portfolio_value * config.OPTIONS["max_position_pct"]
                    trader.buy_option(signal.symbol, "call", dollar_amount, current_price)
                    logger.info(f"Executed CALL for {signal.symbol}")

            logger.info(f"Scan complete. Next scan in {scan_interval}s. "
                        f"Portfolio: ${trader.get_portfolio_value():,.2f}")
            time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
            print(f"Final portfolio: ${trader.get_portfolio_value():,.2f}")
            print(f"Trades executed: {len(trader.get_trade_log())}")
            break

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)


def cmd_tune():
    """Run the walk-forward optimizer to find the best parameters."""
    print("\n" + "="*70)
    print("  STOCK BOT — WALK-FORWARD OPTIMIZER")
    print("  Tests parameter combinations across 5 separate market eras.")
    print("  Finds the config that beats SPY CONSISTENTLY — not just once.")
    print("="*70 + "\n")
    import tune
    tune.run_optimization()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Commands:")
        print("  python main.py backtest    Test on historical data")
        print("  python main.py scan        One-time market scan")
        print("  python main.py live        Start live/paper trading")
        print("  python main.py tune        Walk-forward optimizer")
        return

    command = sys.argv[1].lower()

    if command == "backtest":
        cmd_backtest()
    elif command == "scan":
        cmd_scan()
    elif command in ("live", "paper"):
        if command == "paper":
            config.PAPER_TRADING = True
        cmd_live()
    elif command == "tune":
        cmd_tune()
    else:
        print(f"Unknown command: {command}")
        print("Use: backtest, scan, paper, live, or tune")


if __name__ == "__main__":
    main()
