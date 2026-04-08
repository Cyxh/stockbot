"""
DATA_FETCHER.PY — Pulls all the raw data the bot needs.

Two types of data:
1. PRICE DATA — Historical OHLCV (Open, High, Low, Close, Volume) from Yahoo Finance
2. NEWS DATA — Headlines about each stock from NewsAPI

Think of this as the bot's "eyes" — it sees the market through this data.
"""

import datetime
import logging
import os
import pickle
import time
import requests
import yfinance as yf
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Data cache ──────────────────────────────────────────────────────────────
# Saves fetched price data to disk so backtests don't re-download every run.
# Cache expires after CACHE_MAX_AGE_HOURS hours.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
CACHE_MAX_AGE_HOURS = 4  # refetch after this many hours


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def _cache_get(key: str):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    if age_hours > CACHE_MAX_AGE_HOURS:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_set(key: str, data):
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.debug(f"Cache write failed for {key}: {e}")


def fetch_price_data(symbol: str, days: int = None) -> pd.DataFrame:
    """
    Fetch historical price data for a stock.

    Args:
        symbol: Stock ticker like "AAPL"
        days:   How many calendar days of history (default: config.LOOKBACK_DAYS)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Indexed by date.
    """
    if days is None:
        days = config.LOOKBACK_DAYS

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)

    # Check cache first
    cache_key = f"price_{symbol}_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info(f"  Cached {len(cached)} trading days for {symbol}")
        return cached

    logger.info(f"Fetching {days} days of price data for {symbol}...")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No price data returned for {symbol}")
            return pd.DataFrame()

        # Keep only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Drop any rows with missing data
        df.dropna(inplace=True)

        logger.info(f"  Got {len(df)} trading days for {symbol}")
        _cache_set(cache_key, df)
        return df

    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_all_price_data(symbols: list = None, days: int = None) -> dict:
    """
    Fetch price data for all stocks in the watchlist.

    Returns:
        Dictionary: { "AAPL": DataFrame, "MSFT": DataFrame, ... }
    """
    if symbols is None:
        symbols = config.WATCHLIST
    if days is None:
        days = config.LOOKBACK_DAYS

    all_data = {}
    for symbol in symbols:
        cache_key = f"price_{symbol}_{days}"
        is_cached = _cache_get(cache_key) is not None
        df = fetch_price_data(symbol, days)
        if not df.empty:
            all_data[symbol] = df
        if not is_cached:
            time.sleep(0.3)  # Be polite to Yahoo's servers (skip on cache hit)

    logger.info(f"Fetched price data for {len(all_data)}/{len(symbols)} symbols")
    return all_data


def fetch_news(symbol: str, days_back: int = 7) -> list:
    """
    Fetch recent news headlines for a stock from NewsAPI.

    Args:
        symbol:    Stock ticker like "AAPL"
        days_back: How many days of news to fetch

    Returns:
        List of dicts: [{"title": "...", "description": "...",
                         "published": datetime, "source": "..."}]
    """
    if config.NEWS_API_KEY == "YOUR_NEWSAPI_KEY":
        logger.warning("NewsAPI key not set — skipping news fetch. "
                        "Get a free key at https://newsapi.org")
        return []

    # Map tickers to company names for better news search results
    company_names = {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
        "AMZN": "Amazon", "NVDA": "Nvidia", "META": "Meta",
        "TSLA": "Tesla", "AMD": "AMD", "AVGO": "Broadcom",
        "CRM": "Salesforce", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
        "ABBV": "AbbVie", "JPM": "JPMorgan", "V": "Visa",
        "GS": "Goldman Sachs", "COST": "Costco", "WMT": "Walmart",
        "HD": "Home Depot", "XOM": "ExxonMobil", "CVX": "Chevron",
        "CAT": "Caterpillar", "DE": "Deere", "SPY": "S&P 500",
        "QQQ": "Nasdaq",
        "NOW": "ServiceNow", "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike",
        "FICO": "Fair Isaac FICO", "NET": "Cloudflare", "DDOG": "Datadog",
        "FTNT": "Fortinet", "ADBE": "Adobe", "INTU": "Intuit",
        "AMAT": "Applied Materials", "KLAC": "KLA Corporation", "LRCX": "Lam Research",
        "MU": "Micron Technology", "TSM": "Taiwan Semiconductor", "ASML": "ASML",
        "ISRG": "Intuitive Surgical", "DXCM": "DexCom", "ELV": "Elevance Health",
        "CI": "Cigna", "HUM": "Humana", "IDXX": "IDEXX Laboratories",
        "VEEV": "Veeva Systems", "REGN": "Regeneron",
        "MS": "Morgan Stanley", "BLK": "BlackRock", "SPGI": "S&P Global",
        "MCO": "Moodys", "AXP": "American Express", "ICE": "Intercontinental Exchange",
        "CME": "CME Group",
        "NKE": "Nike", "SBUX": "Starbucks", "MCD": "McDonalds",
        "LOW": "Lowes", "BKNG": "Booking Holdings", "TGT": "Target", "MELI": "MercadoLibre",
        "PG": "Procter Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
        "HON": "Honeywell", "RTX": "RTX Raytheon", "LMT": "Lockheed Martin",
        "GE": "GE Aerospace", "ITW": "Illinois Tool Works", "PH": "Parker Hannifin",
        "ODFL": "Old Dominion Freight", "URI": "United Rentals",
        "LIN": "Linde", "APD": "Air Products", "SHW": "Sherwin-Williams", "ECL": "Ecolab",
        "AMT": "American Tower", "EQIX": "Equinix", "PLD": "Prologis",
        "SLB": "SLB Schlumberger", "OXY": "Occidental Petroleum",
        "EOG": "EOG Resources", "PSX": "Phillips 66", "MPC": "Marathon Petroleum",
    }
    query = company_names.get(symbol, symbol) + " stock"

    from_date = (datetime.datetime.now() -
                 datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 30,
        "apiKey": config.NEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "published": art.get("publishedAt", ""),
                "source": art.get("source", {}).get("name", "Unknown"),
            })

        logger.info(f"  Got {len(articles)} news articles for {symbol}")
        return articles

    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []


def fetch_all_news(symbols: list = None) -> dict:
    """
    Fetch news for all stocks in the watchlist.

    Returns:
        Dictionary: { "AAPL": [articles], "MSFT": [articles], ... }
    """
    if symbols is None:
        symbols = config.WATCHLIST

    all_news = {}
    for symbol in symbols:
        articles = fetch_news(symbol)
        all_news[symbol] = articles
        time.sleep(0.2)  # Rate limit courtesy

    return all_news


def get_current_price(symbol: str) -> float | None:
    """
    Get the most recent closing price for a stock.
    Used during live trading to check current prices.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
    return None


def fetch_fundamentals(symbol: str) -> dict:
    """
    Fetch live fundamental signals for a stock (analyst sentiment + quality metrics).
    Used only in live trading — not backtesting (no historical fundamental replay).

    Returns:
        Dict with "earnings_revision" and "quality_score" keys, each in [-1.0, 1.0].
        Empty dict on failure (signals will be absent from weighted scoring).
    """
    try:
        ticker = yf.Ticker(symbol)
        result = {}

        # ── Analyst recommendation consensus ──────────────────────────
        # Upward/downward bias in analyst ratings predicts forward returns
        # (Elgers et al. 2001; IC ~0.04-0.06 in academic literature).
        try:
            recs = ticker.recommendations_summary
            if recs is not None and not recs.empty:
                latest = recs.iloc[0]
                upgrades   = float(latest.get("strongBuy", 0) + latest.get("buy", 0))
                downgrades = float(latest.get("sell", 0) + latest.get("strongSell", 0))
                holds      = float(latest.get("hold", 0))
                total      = upgrades + downgrades + holds
                if total > 0:
                    # Net bullish ratio: +1 = unanimous buy, -1 = unanimous sell
                    result["earnings_revision"] = float(
                        max(min((upgrades - downgrades) / total * 2, 1.0), -1.0)
                    )
        except Exception:
            pass

        # ── Quality score: ROE + Gross Margins ────────────────────────
        # High-quality companies (Novy-Marx 2013; Fama-French 2015) outperform.
        # Using ROE (>=20% = excellent) and gross margins (>=40% = excellent).
        try:
            info = ticker.info
            scores = []
            roe = info.get("returnOnEquity")
            if roe is not None and not (isinstance(roe, float) and (roe != roe)):
                # 20%+ ROE = +1, 0% = 0, negative = -1
                scores.append(float(max(min(roe / 0.20, 1.0), -1.0)))
            gross_margin = info.get("grossMargins")
            if gross_margin is not None and not (isinstance(gross_margin, float) and (gross_margin != gross_margin)):
                # 40%+ gross margin = +1, 10% = 0 baseline, negative = poor
                scores.append(float(max(min((gross_margin - 0.10) / 0.30, 1.0), -1.0)))
            if scores:
                result["quality_score"] = float(sum(scores) / len(scores))
        except Exception:
            pass

        return result

    except Exception as e:
        logger.warning(f"Could not fetch fundamentals for {symbol}: {e}")
        return {}


def fetch_vix() -> float:
    """
    Fetch the current CBOE Volatility Index (VIX) value.
    VIX measures implied 30-day volatility of the S&P 500 options market.

    Interpretation:
      < 15:  Low fear — calm, trending market
      15-20: Mild concern
      20-25: Elevated uncertainty
      25-35: High fear — increased hedging activity
      > 35:  Extreme fear / crisis conditions

    Returns:
        Current VIX level as a float. Returns 20.0 (neutral) on failure.
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch VIX: {e}")
    return 20.0  # Neutral default


def fetch_vix_history(days: int = 730) -> pd.Series:
    """
    Fetch historical VIX data for backtesting.
    Returns a Series indexed by date with VIX closing values.
    """
    cache_key = f"vix_history_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info(f"  Cached VIX history: {len(cached)} days")
        return cached

    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days + 60)
        vix = yf.Ticker("^VIX")
        df = vix.history(start=start, end=end)
        if not df.empty:
            series = df["Close"]
            series.index = pd.to_datetime(series.index).tz_localize(None)
            _cache_set(cache_key, series)
            return series
    except Exception as e:
        logger.warning(f"Could not fetch VIX history: {e}")
    return pd.Series(dtype=float)


def get_earnings_dates(symbol: str, lookahead_days: int = 30) -> list:
    """
    Get upcoming earnings announcement dates for a stock.

    Args:
        symbol:         Stock ticker
        lookahead_days: How many days ahead to look

    Returns:
        List of datetime.date objects for upcoming earnings dates.
        Empty list if unavailable.
    """
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None:
            return []

        # calendar is a dict with 'Earnings Date' key (list of timestamps)
        earnings_dates = cal.get("Earnings Date", [])
        if not earnings_dates:
            return []

        today = datetime.date.today()
        cutoff = today + datetime.timedelta(days=lookahead_days)
        upcoming = []
        for dt in earnings_dates:
            try:
                if hasattr(dt, "date"):
                    d = dt.date()
                else:
                    d = pd.Timestamp(dt).date()
                if today <= d <= cutoff:
                    upcoming.append(d)
            except Exception:
                pass
        return upcoming
    except Exception as e:
        logger.debug(f"Could not fetch earnings dates for {symbol}: {e}")
        return []


def is_near_earnings(symbol: str, blackout_days: int = None) -> bool:
    """
    Returns True if the stock has an earnings announcement within
    blackout_days trading days (default from config.EARNINGS_BLACKOUT_DAYS).

    Used to avoid opening new positions before earnings surprises.
    """
    if blackout_days is None:
        blackout_days = getattr(config, "EARNINGS_BLACKOUT_DAYS", 5)

    dates = get_earnings_dates(symbol, lookahead_days=blackout_days + 5)
    if not dates:
        return False

    today = datetime.date.today()
    for d in dates:
        days_away = (d - today).days
        if 0 <= days_away <= blackout_days:
            return True
    return False


def fetch_yield_curve_spread() -> float:
    """
    Fetch the 10-year minus 3-month Treasury yield spread.

    This is the most widely-tracked recession predictor:
    - Positive spread (e.g. +1.5%): Normal — longer-term rates > short-term
    - Near zero: Flattening — economic uncertainty growing
    - Negative (inverted): Recession warning — has preceded every US recession
      since 1955 with ~12-18 month lead time

    Uses yfinance: ^TNX (10-year yield) and ^IRX (13-week T-Bill rate).
    Both are quoted in percent (e.g., 4.5 = 4.5%).

    Returns:
        Spread in percentage points. Returns 1.0 (non-inverted default) on failure.
    """
    try:
        t10_hist = yf.Ticker("^TNX").history(period="5d")["Close"]
        t3m_hist = yf.Ticker("^IRX").history(period="5d")["Close"]

        if t10_hist.empty or t3m_hist.empty:
            return 1.0

        t10 = float(t10_hist.iloc[-1])
        t3m = float(t3m_hist.iloc[-1])
        spread = t10 - t3m
        logger.debug(f"Yield curve: 10Y={t10:.2f}% 3M={t3m:.2f}% spread={spread:+.2f}%")
        return spread
    except Exception as e:
        logger.warning(f"Could not fetch yield curve data: {e}")
        return 1.0  # Default to non-inverted


def fetch_yield_curve_history(days: int = 730) -> pd.Series:
    """
    Fetch historical yield curve spread for backtesting.
    Returns a Series of (10Y - 3M) spread indexed by date.
    """
    cache_key = f"yield_curve_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info(f"  Cached yield curve history: {len(cached)} days")
        return cached

    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days + 60)

        t10 = yf.Ticker("^TNX").history(start=start, end=end)["Close"]
        t3m = yf.Ticker("^IRX").history(start=start, end=end)["Close"]

        if t10.empty or t3m.empty:
            return pd.Series(dtype=float)

        # Align on common dates and compute spread
        spread = (t10 - t3m).dropna()
        spread.index = pd.to_datetime(spread.index).tz_localize(None)
        _cache_set(cache_key, spread)
        return spread
    except Exception as e:
        logger.warning(f"Could not fetch yield curve history: {e}")
        return pd.Series(dtype=float)
