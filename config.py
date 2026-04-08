"""
CONFIG.PY — Central configuration for the trading bot.

HOW TO SET UP:
1. Copy .env.example to .env
2. Fill in your Alpaca and NewsAPI keys in .env
3. Never commit .env — it's in .gitignore
"""

import os

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on real env vars or defaults

# =============================================================================
# ALPACA BROKERAGE CREDENTIALS  (set in .env, not here)
# =============================================================================
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY",    "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

# Paper trading = fake money (safe default). Set PAPER_TRADING=false in .env for real money.
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() != "false"

# Alpaca endpoints
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL  = "https://api.alpaca.markets"

# =============================================================================
# NEWS API CREDENTIALS  (set in .env, not here)
# =============================================================================
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")

# =============================================================================
# STOCK WATCHLIST
# Broad sector coverage enables cross-sectional momentum:
# the bot naturally rotates into whatever sector is leading.
# =============================================================================
WATCHLIST = [
    # ── Mega-cap Tech ──────────────────────────────────────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO",
    # ── High-Growth Tech ───────────────────────────────────────────────
    "AMD", "TSLA", "CRM", "NOW", "PANW", "CRWD", "FICO",
    "NET", "DDOG", "FTNT", "ADBE", "INTU",
    # ── Semiconductors ────────────────────────────────────────────────
    "AMAT", "KLAC", "LRCX", "MU", "TSM", "ASML",
    # ── Healthcare ─────────────────────────────────────────────────────
    "UNH", "LLY", "ABBV", "ISRG", "DXCM", "ELV",
    "CI", "HUM", "IDXX", "VEEV", "REGN",
    # ── Financials ─────────────────────────────────────────────────────
    "JPM", "V", "GS", "MS", "BLK", "SPGI",
    "MCO", "AXP", "ICE", "CME",
    # ── Consumer Discretionary ─────────────────────────────────────────
    "COST", "WMT", "HD", "NKE", "SBUX", "MCD",
    "LOW", "BKNG", "TGT", "MELI",
    # ── Consumer Staples ───────────────────────────────────────────────
    "PG", "KO", "PEP",
    # ── Industrials ────────────────────────────────────────────────────
    "CAT", "DE", "HON", "RTX", "LMT", "GE",
    "ITW", "PH", "ODFL", "URI",
    # ── Materials ──────────────────────────────────────────────────────
    "LIN", "APD", "SHW", "ECL",
    # ── Real Estate ────────────────────────────────────────────────────
    "AMT", "EQIX", "PLD",
    # ── Energy ─────────────────────────────────────────────────────────
    "XOM", "CVX", "SLB", "OXY", "EOG", "PSX", "MPC",
    # ── ETFs (regime filter + cash parking — never traded for alpha) ───
    "SPY", "QQQ", "BIL",
]

# When the regime filter blocks stock buying, idle cash is parked here
BEAR_REGIME_ETF = "BIL"

# ETFs used only for regime detection / cash parking — never traded for alpha.
# Trading SPY/QQQ directly adds pure market beta with zero alpha.
REGIME_ONLY_SYMBOLS = {"SPY", "QQQ", "BIL"}

# Maximum new long positions opened on a single day.
# Prevents pile-ins at correlated market peaks.
MAX_NEW_POSITIONS_PER_DAY = 4

# Minimum holding period (trading days) before a position can be sold via
# rank-drop rebalancing. Stops still fire normally for protection.
# Prevents noise-driven churn on positions held for only one rebalance cycle.
# Ranking hysteresis: once a stock enters the portfolio (top MAX_OPEN_POSITIONS),
# it stays unless it drops below this rank. Reduces whipsaw from small rank changes.
# E.g., with 16 max positions and sell rank 24, a stock enters at rank 16 but only
# exits when it drops to rank 25+. Set to 0 to disable.
RANK_SELL_THRESHOLD = 24

# Re-entry cooldown: after a rank_drop exit, don't re-buy the same stock
# for this many trading days. Prevents churn on borderline stocks.
REENTRY_COOLDOWN_DAYS = 12

# Sector map — used for concentration limits.
# Portfolio theory: limiting intra-sector exposure reduces drawdown when
# entire sectors sell off together (e.g. July 2024 tech correction,
# April 2025 energy/industrials tariff shock).
SECTOR_MAP = {
    # Tech
    "AAPL": "tech",  "MSFT": "tech",  "GOOGL": "tech", "AMZN": "tech",
    "NVDA": "tech",  "META": "tech",  "AVGO": "tech",  "AMD":  "tech",
    "TSLA": "tech",  "CRM":  "tech",  "NOW":  "tech",  "PANW": "tech",
    "CRWD": "tech",  "FICO": "tech",  "NET":  "tech",  "DDOG": "tech",
    "FTNT": "tech",  "ADBE": "tech",  "INTU": "tech",
    # Semiconductors (separate sector — different cycle from software)
    "AMAT": "semis", "KLAC": "semis", "LRCX": "semis", "MU":   "semis",
    "TSM":  "semis", "ASML": "semis",
    # Healthcare
    "UNH":  "healthcare", "LLY":  "healthcare", "ABBV": "healthcare",
    "ISRG": "healthcare", "DXCM": "healthcare", "ELV":  "healthcare",
    "CI":   "healthcare", "HUM":  "healthcare", "IDXX": "healthcare",
    "VEEV": "healthcare", "REGN": "healthcare",
    # Financials
    "JPM":  "financials", "V":    "financials", "GS":   "financials",
    "MS":   "financials", "BLK":  "financials", "SPGI": "financials",
    "MCO":  "financials", "AXP":  "financials", "ICE":  "financials",
    "CME":  "financials",
    # Consumer Discretionary
    "COST": "consumer",   "WMT":  "consumer",   "HD":   "consumer",
    "NKE":  "consumer",   "SBUX": "consumer",   "MCD":  "consumer",
    "LOW":  "consumer",   "BKNG": "consumer",   "TGT":  "consumer",
    "MELI": "consumer",
    # Consumer Staples
    "PG":   "staples",    "KO":   "staples",    "PEP":  "staples",
    # Industrials
    "CAT":  "industrials","DE":   "industrials","HON":  "industrials",
    "RTX":  "industrials","LMT":  "industrials","GE":   "industrials",
    "ITW":  "industrials","PH":   "industrials","ODFL": "industrials",
    "URI":  "industrials",
    # Materials
    "LIN":  "materials",  "APD":  "materials",  "SHW":  "materials",
    "ECL":  "materials",
    # Real Estate
    "AMT":  "realestate", "EQIX": "realestate", "PLD":  "realestate",
    # Energy
    "XOM":  "energy",     "CVX":  "energy",     "SLB":  "energy",
    "OXY":  "energy",     "EOG":  "energy",     "PSX":  "energy",
    "MPC":  "energy",
}

# Maximum open positions in any single sector.
# Limits correlated drawdowns when an entire sector rotates out.
# Default sector cap — overridden per-sector below.
MAX_POSITIONS_PER_SECTOR = 5

# Tiered sector caps: defensive sectors get fewer slots since momentum
# signal strength is weaker there. Growth/cyclical sectors keep full cap.
SECTOR_MAX_OVERRIDE = {
    "healthcare": 2,
    "staples":    1,
    "materials":  2,
    "realestate": 2,
}

# Max % of total portfolio value in any single sector (dollar-weighted limit).
# Prevents tech from becoming 80% of the portfolio even with 4-position count limit.
MAX_SECTOR_EXPOSURE_PCT = 0.40

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
# 730 calendar days ≈ 500 trading days — gives 12M momentum a full year of
# firing history even after the 60-day warmup period.
LOOKBACK_DAYS           = 730
BACKTEST_STARTING_CASH  = 100_000

# Cross-sectional rebalance frequency (trading days).
# Every N days: re-rank all stocks, rotate out rank-droppers, buy new leaders.
# 5 = weekly rebalance (matches academic monthly momentum at shorter scale).
REBALANCE_EVERY = 3

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Maximum % of portfolio per single trade.
# Reduced to 10% (from 15%) — with 10 positions the portfolio is fully deployed.
MAX_POSITION_SIZE_PCT = 0.15

# Maximum number of concurrent open positions.
# 15 maximizes opportunities across the expanded 83-stock universe.
MAX_OPEN_POSITIONS = 16

# Hard stop-loss: used only as a reference for options and config compatibility.
# Stock exits are handled by the trailing stop (TRAILING_STOP_PCT) from entry.
STOP_LOSS_PCT = 0.08

# Take-profit (used only for options — stocks use trailing stop instead)
TAKE_PROFIT_PCT = 0.12

# Trailing stop for stock positions: only fires to protect large gains.
# Tiered system: tighter trail on moderate winners, wider on big runners.
TRAILING_STOP_PCT = 0.25  # fallback / max trail

# Tiered trailing stop thresholds:
# (min_gain_from_entry, trail_pct) — checked in order, first match wins.
# Wider trails let momentum winners run longer — post-exit analysis showed
# stocks averaged +5.5% in the 20 days after being stopped out.
TRAILING_STOP_TIERS = [
    (0.35, 0.18),  # big runner (35%+ gain): 18% trail to lock in profit
    (0.10, 0.25),  # moderate winner (10%+ gain): 25% trail gives room to run
]

# Minimum signal confidence to enter a trade (0.0–1.0)
# 0.44 — slightly lower than 0.48 to re-enter faster after crashes.
MIN_CONFIDENCE = 0.45

# Cooldown: don't re-enter a symbol this many trading days after a stop-loss exit.
# Reduced from 15 → 7 days — don't miss fast post-crash recoveries.
COOLDOWN_DAYS = 0

# Per-side slippage/commission estimate (0.05% = 0.1% round-trip).
# Applied in backtesting to produce realistic results.
SLIPPAGE_PCT = 0.0005

# =============================================================================
# ATR-BASED STOP-LOSS
# Uses the stock's own volatility (Average True Range) to set stop distance.
# More volatile stocks get wider stops; stable stocks get tighter stops.
# This prevents stable stocks from being stopped out by noise while also
# preventing volatile stocks from running massive losses before stopping.
# =============================================================================
ATR_STOP_MULTIPLIER = 3.0   # Stop at 3.0× ATR below entry price
ATR_STOP_MIN_PCT    = 0.08  # Never tighter than 8% (momentum stocks need room)
ATR_STOP_MAX_PCT    = 0.20  # Never wider than 20%

# =============================================================================
# VIX-BASED POSITION SCALING
# Reduces position sizes as market fear (VIX) rises.
# Each tuple: (vix_threshold, position_size_multiplier)
# When VIX is below the threshold, that multiplier applies.
# =============================================================================
VIX_SCALE_LEVELS = [
    (15,  1.00),  # VIX < 15:  calm market,  full position size
    (20,  0.80),  # VIX 15-20: mild concern, 80% size
    (25,  0.60),  # VIX 20-25: elevated,     60% size
    (35,  0.40),  # VIX 25-35: high fear,    40% size
    (999, 0.20),  # VIX > 35:  extreme fear, 20% size (near-cash)
]

# =============================================================================
# EARNINGS BLACKOUT
# Avoid opening new positions within this many days of an earnings announcement.
# Earnings cause unpredictable overnight gaps that stop-losses can't protect against.
# =============================================================================
EARNINGS_BLACKOUT_DAYS = 5

# =============================================================================
# YIELD CURVE REGIME FILTER
# The 10Y-3M Treasury spread is a proven recession predictor.
# When inverted (negative), reduce new position sizes by 30%.
# =============================================================================
USE_YIELD_CURVE_REGIME = True
YIELD_CURVE_BEARISH_THRESHOLD = -0.25  # Spread below this = macro warning

# =============================================================================
# TECHNICAL INDICATOR SETTINGS
# =============================================================================
TECHNICAL = {
    "sma_fast": 10,
    "sma_slow": 50,
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "volume_avg_period": 20,
    "adx_period": 14,         # ADX period — measures trend strength 0-100
    "adx_min_trend": 18,      # ADX below this = choppy/ranging market (dampen signals)
    # Signal combination weights — tuned automatically by tune.py
    # Factor diversification reduces overfitting: 10 factors average out
    # noise better than 5 concentrated ones.
    "weights": {
        "sma_crossover":         0.00,
        "sma_spread_strength":   0.00,
        "ema_crossover":         0.00,
        "rsi":                   0.00,
        "macd":                  0.00,
        "macd_momentum":         0.00,
        "bollinger":             0.00,
        "vwap":                  0.00,
        "volume_confirmation":   0.00,
        "momentum_12m":          0.15,  # strongest single predictor (JT 1993)
        "momentum_6m":           0.10,
        "momentum_3m":           0.10,
        "vol_adj_momentum":      0.10,  # momentum/vol (Barroso & SC 2015)
        "idiosyncratic_momentum":0.08,  # market-residual momentum (Blitz+ 2011)
        "relative_strength":     0.08,
        "sector_momentum":       0.05,  # sector-level momentum (M&G 1999)
        "momentum_acceleration": 0.06,  # trend strengthening vs fading
        "high_52w_proximity":    0.05,  # anchoring bias (George & Hwang 2004)
        "trend_efficiency":      0.05,  # smooth trends persist longer
        "drawdown_recovery":     0.04,  # penalize momentum crash candidates
        "short_term_reversal":   0.04,  # buy pullbacks in uptrends
        "volume_price_confirm":  0.03,  # volume confirms direction
        "earnings_revision":     0.04,  # analyst consensus (live only)
        "quality_score":         0.02,  # ROE + margins (live only)
    },
}

# =============================================================================
# CROSS-SECTIONAL RANKING WEIGHTS (used by backtester for stock selection)
# These must match the backtest-available factors in TECHNICAL weights above.
# Weights are renormalized at runtime when factors are missing (insufficient data).
# =============================================================================
RANKING_WEIGHTS = {
    "momentum_12m":          0.18,   # strongest predictor of forward returns
    "vol_adj_momentum":      0.12,   # risk-adjusted momentum — more robust OOS
    "idiosyncratic_momentum":0.08,   # market-residual momentum (Blitz+ 2011)
    "relative_strength":     0.10,   # outperformance vs S&P 500
    "momentum_6m":           0.10,
    "momentum_3m":           0.08,
    "sector_momentum":       0.06,   # sector-level momentum (M&G 1999)
    "high_52w_proximity":    0.06,   # anchoring bias
    "momentum_acceleration": 0.06,   # trend strengthening
    "trend_efficiency":      0.05,   # smooth > noisy trends
    "short_term_reversal":   0.05,   # buy the dip in uptrends
    "volume_price_confirm":  0.06,   # volume confirms direction
}

# =============================================================================
# INVERSE-VOLATILITY POSITION SIZING
# Size positions by inverse of realized volatility so each position
# contributes roughly equal risk. Volatile stocks get smaller positions.
# This is the standard risk-parity approach used by most quant funds.
# =============================================================================
USE_INVERSE_VOL_SIZING = True
INVERSE_VOL_TARGET_VOL = 0.20   # Target 20% annualized vol per position

# =============================================================================
# PORTFOLIO DRAWDOWN SCALING
# Reduces new position sizes when the portfolio is in drawdown.
# Prevents compounding losses during regime transitions.
# Each tuple: (drawdown_threshold, position_size_multiplier)
# =============================================================================
DRAWDOWN_SCALE_LEVELS = [
    (0.08, 1.00),   # <8% drawdown: full size (normal market noise)
    (0.15, 0.80),   # 8-15% drawdown: 80% size
    (0.22, 0.50),   # 15-22% drawdown: 50% size
    (0.30, 0.25),   # 22-30% drawdown: 25% size
    (1.00, 0.10),   # >30% drawdown: 10% (near-cash)
]

# =============================================================================
# SENTIMENT SETTINGS
# =============================================================================
SENTIMENT = {
    "news_lookback_hours": 48,
    "min_articles": 3,
    "bullish_threshold":  0.15,
    "bearish_threshold": -0.15,
    # Exponential decay: articles older than this many hours get half-weight.
    # Keeps today's headlines more influential than yesterday's.
    "recency_halflife_hours": 24,
}

# =============================================================================
# OPTIONS SETTINGS
# =============================================================================
OPTIONS = {
    "enabled":              True,
    "target_dte":           30,
    "exit_dte":             5,
    "stop_loss_pct":        0.50,
    "take_profit_pct":      1.00,
    "use_puts_on_bearish":  True,
    "use_calls_on_bullish": False,
    "max_position_pct":     0.02,
}

# =============================================================================
# LOGGING
# =============================================================================
LOG_FILE  = "bot_trades.log"
LOG_LEVEL = "INFO"
