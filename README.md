# 🤖 Algorithmic Stock Trading Bot

A Python trading bot that combines **technical analysis** (chart patterns, indicators) with **news sentiment analysis** to make data-driven trading decisions.

## ⚡ Quick Start (5 minutes)

### 1. Install Python 3.10+
Download from [python.org](https://python.org) if you don't have it.

### 2. Install dependencies
```bash
cd stock_bot
pip install -r requirements.txt
```

### 3. Run your first backtest (no API keys needed!)
```bash
python main.py backtest
```
This tests the strategy against 180 days of real historical data using Yahoo Finance (free, no account needed).

### 4. Set up API keys (for live trading + news)

**Alpaca (free brokerage with API):**
1. Go to [alpaca.markets](https://alpaca.markets) and create a free account
2. Navigate to Paper Trading → API Keys
3. Copy your API Key and Secret Key into `config.py`

**NewsAPI (free news data):**
1. Go to [newsapi.org](https://newsapi.org) and create a free account
2. Copy your API key into `config.py`

### 5. Run a live scan
```bash
python main.py scan
```
This shows what the bot would trade RIGHT NOW without placing any orders.

### 6. Start paper trading
```bash
python main.py live
```
This runs the bot with Alpaca's paper (fake money) account. Leave it running during market hours.

---

## 📁 Project Structure

| File | What it does |
|------|-------------|
| `config.py` | All settings: API keys, risk limits, stock watchlist |
| `data_fetcher.py` | Pulls price data (Yahoo Finance) and news (NewsAPI) |
| `technical.py` | Calculates chart indicators (SMA, RSI, MACD, Bollinger, etc.) |
| `sentiment.py` | Scores news headlines as bullish or bearish |
| `strategy.py` | Combines both analyses into BUY/SELL/HOLD decisions |
| `trader.py` | Executes orders through Alpaca brokerage |
| `backtester.py` | Tests the strategy against 180 days of historical data |
| `main.py` | Entry point — run this with `backtest`, `scan`, or `live` |

---

## 🧠 How the Bot Thinks

### Technical Analysis (60% weight)
The bot reads price charts using 7 indicators:

| Indicator | What it measures | Signal |
|-----------|-----------------|--------|
| SMA Crossover | Trend direction | Fast above slow = bullish |
| EMA Crossover | Trend (faster reaction) | Fast above slow = bullish |
| RSI | Momentum (0-100) | Below 30 = buy, above 70 = sell |
| MACD | Trend momentum | MACD above signal line = bullish |
| Bollinger Bands | Volatility envelope | Price at lower band = potential buy |
| Volume | Trade confirmation | High volume confirms moves |
| VWAP | Institutional price | Price above VWAP = bullish |

### Sentiment Analysis (40% weight)
The bot reads news headlines and scores them:
- Uses VADER sentiment analysis (tuned for social/financial text)
- Custom financial keyword dictionary (400+ terms)
- Requires multiple articles to form an opinion (avoids reacting to one headline)
- Agreement among articles boosts confidence

### Decision Rules
- Bot only trades when **both brains agree** in the same direction
- Minimum confidence threshold of 60% (configurable)
- Maximum 6 positions at once
- Maximum 5% of portfolio in any single stock
- Automatic stop-loss at -3% and take-profit at +7%

---

## ⚙️ Configuration Guide

### Adjusting Risk (in `config.py`)

| Setting | Default | Conservative | Aggressive |
|---------|---------|-------------|------------|
| `MAX_POSITION_SIZE_PCT` | 0.05 (5%) | 0.03 (3%) | 0.08 (8%) |
| `MAX_OPEN_POSITIONS` | 6 | 4 | 8 |
| `STOP_LOSS_PCT` | 0.03 (3%) | 0.02 (2%) | 0.05 (5%) |
| `TAKE_PROFIT_PCT` | 0.07 (7%) | 0.05 (5%) | 0.10 (10%) |
| `MIN_CONFIDENCE` | 0.6 | 0.7 | 0.5 |

### Changing the Stock Watchlist
Edit the `WATCHLIST` in `config.py`. Tips:
- Stick to **liquid stocks** (high daily volume) — they have tighter spreads
- 8-15 stocks is a good range — enough diversity without too much noise
- Include 1-2 ETFs (SPY, QQQ) as market barometers

---

## 📊 Understanding Backtest Results

| Metric | What it means | Good values |
|--------|--------------|-------------|
| Total Return | Your profit/loss % | Positive and > S&P 500 |
| Sharpe Ratio | Return per unit of risk | > 1.0 is good, > 2.0 is great |
| Max Drawdown | Worst peak-to-trough drop | < 15% is manageable |
| Win Rate | % of trades that profit | > 50% with good profit factor |
| Profit Factor | Gross wins / Gross losses | > 1.5 is solid |

---

## 🚀 Going Live (Real Money)

When you're ready to trade real money:

1. **Paper trade for at least 1-3 months** to validate performance
2. In `config.py`, set `PAPER_TRADING = False`
3. Use your **live** API keys from Alpaca (not paper keys)
4. Start small — consider starting with a modest amount
5. Monitor daily — don't just set it and forget it

---

## ⚠️ Important Disclaimers

- **This is not financial advice.** This bot is a tool for educational and experimental purposes.
- **Past performance does not guarantee future results.** Backtests always look better than live trading.
- **You can lose money.** Algorithmic trading carries significant risk.
- **Start with paper trading.** Always validate with fake money first.
- **The bot is not infallible.** Market conditions change, black swan events happen.

---

## 🔧 Troubleshooting

**"No data fetched"** — Check your internet connection. Yahoo Finance is free but can be rate-limited.

**"vaderSentiment not installed"** — Run `pip install vaderSentiment`. The bot will still work without it using keyword-only sentiment.

**"Alpaca not configured"** — The backtest works without Alpaca. You only need it for live/paper trading.

**Backtest shows negative returns** — Try adjusting `MIN_CONFIDENCE` higher (e.g., 0.7) or changing the watchlist. Not every strategy works on every set of stocks.
