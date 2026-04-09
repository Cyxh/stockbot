"""
REPORT.PY — Daily performance report, pushed to GitHub.

Generates a markdown summary of portfolio state, trades, and P&L,
then commits it to the repo so progress can be tracked over time.

Runs automatically at market close from main.py.
"""

import logging
import os
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def generate_daily_report(trader) -> str | None:
    """Generate a markdown report and push it to GitHub.

    Returns the file path on success, None on failure.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    now = datetime.now(ET)
    date_str = now.strftime("%Y-%m-%d")
    filepath = os.path.join(REPORT_DIR, f"{date_str}.md")

    try:
        account = trader.client.get_account() if trader.client else None
    except Exception:
        account = None

    # Portfolio summary
    equity = float(account.equity) if account else 0
    cash = float(account.cash) if account else 0
    buying_power = float(account.buying_power) if account else 0

    positions = trader.get_positions()
    trade_log = trader.get_trade_log()

    # Today's trades
    today_trades = [
        t for t in trade_log
        if t.get("timestamp", "").startswith(date_str)
    ]

    # Build report
    lines = [
        f"# Daily Report — {date_str}",
        "",
        "## Portfolio Summary",
        f"- **Equity:** ${equity:,.2f}",
        f"- **Cash:** ${cash:,.2f}",
        f"- **Buying Power:** ${buying_power:,.2f}",
        f"- **Open Positions:** {len(positions)}",
        f"- **Trades Today:** {len(today_trades)}",
        "",
    ]

    # Positions table
    if positions:
        lines.append("## Open Positions")
        lines.append("| Symbol | Qty | Avg Price | Current | P&L % |")
        lines.append("|--------|-----|-----------|---------|-------|")
        for sym, pos in sorted(positions.items()):
            lines.append(
                f"| {sym} | {pos.get('qty', 0):.0f} "
                f"| ${pos.get('avg_price', 0):.2f} "
                f"| ${pos.get('current_price', 0):.2f} "
                f"| {pos.get('unrealized_pl_pct', 0):.1%} |"
            )
        lines.append("")

    # Today's trades
    if today_trades:
        lines.append("## Trades Today")
        lines.append("| Time | Action | Symbol | Qty | Price | Status |")
        lines.append("|------|--------|--------|-----|-------|--------|")
        for t in today_trades:
            ts = t.get("timestamp", "")
            time_str = ts[11:16] if len(ts) > 16 else ts
            lines.append(
                f"| {time_str} | {t.get('action', '')} "
                f"| {t.get('symbol', '')} | {t.get('qty', '-')} "
                f"| ${t.get('price', 0):.2f} | {t.get('status', '')} |"
            )
        lines.append("")
    else:
        lines.append("## Trades Today")
        lines.append("No trades executed today.")
        lines.append("")

    lines.append(f"*Generated at {now.strftime('%H:%M ET')}*")

    report_text = "\n".join(lines)

    # Write report
    with open(filepath, "w") as f:
        f.write(report_text)

    # Push to GitHub
    try:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["git", "add", filepath], cwd=project_dir, check=True,
                        capture_output=True, timeout=30)
        subprocess.run(
            ["git", "commit", "-m", f"Daily report {date_str}"],
            cwd=project_dir, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(["git", "push"], cwd=project_dir, check=True,
                        capture_output=True, timeout=60)
        logger.info(f"Daily report pushed to GitHub: {date_str}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not push report to GitHub: {e.stderr}")
    except Exception as e:
        logger.warning(f"Git push failed: {e}")

    return filepath
