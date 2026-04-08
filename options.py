"""
OPTIONS.PY — Options pricing, contract selection, and simulation.

Enables the bot to trade puts and calls, not just stocks.

WHY OPTIONS?
- PUTS profit when a stock FALLS — so the bot can make money in a bear market
- CALLS give leveraged upside on bullish moves
- Risk is capped: you can only lose the premium you paid

HOW IT WORKS IN BACKTEST:
We simulate option prices using the Black-Scholes formula — the industry-
standard model for option pricing. It takes:
  - Current stock price
  - Strike price (we use at-the-money, so strike = current price)
  - Time to expiration (in years)
  - Risk-free rate (~5% currently)
  - Volatility (calculated from the historical price data we already have)

We reprice the option every simulated day and exit on:
  - Stop-loss: option loses 50% of premium (configurable)
  - Take-profit: option gains 100% of premium (configurable)
  - Time stop: exit when 5 DTE remain (avoids gamma risk / expiration wipeout)

HOW IT WORKS IN LIVE TRADING:
We fetch the real options chain from Yahoo Finance, find the closest
at-the-money contract with ~30 days to expiration, and execute via Alpaca.
"""

import logging
from math import log, sqrt, exp, erfc
from datetime import datetime, date, timedelta

import numpy as np
import yfinance as yf

import config

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05  # ~5% (approximate current T-bill / Fed funds rate)


# ── Black-Scholes Implementation ────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF — uses math.erfc so no scipy dependency needed."""
    return 0.5 * erfc(-x / sqrt(2))


def black_scholes(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str = "call",
    r: float = RISK_FREE_RATE,
) -> float:
    """
    Black-Scholes option pricing formula.

    Args:
        S:           Current underlying (stock) price
        K:           Strike price
        T:           Time to expiration in years (e.g., 30/252 for 30 trading days)
        sigma:       Annualized volatility (e.g., 0.25 = 25% vol)
        option_type: "call" or "put"
        r:           Annual risk-free rate

    Returns:
        Theoretical option premium per share.
        Multiply by 100 for the cost of 1 standard contract.
    """
    if T <= 0:
        # At expiration, option is worth its intrinsic value only
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    sigma = max(sigma, 0.01)  # Avoid division by zero

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return max(price, 0.01)


def historical_vol(price_series, window: int = 20) -> float:
    """
    Calculate annualized historical volatility from a closing price series.
    This is the standard volatility input for Black-Scholes when implied
    volatility is not available (e.g., in backtesting).
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_returns) < window:
        return 0.30  # Default 30% vol if not enough history

    hv = float(log_returns.rolling(window=window).std().iloc[-1]) * sqrt(252)
    return float(np.clip(hv, 0.05, 2.0))  # Clamp to sane range


# ── Backtest Simulation ──────────────────────────────────────────────────────

def simulate_option_entry(
    underlying_price: float,
    option_type: str,
    hist_vol: float,
    target_dte: int = None,
) -> dict:
    """
    Simulate buying an ATM option for backtesting.
    Prices the option using Black-Scholes with historical volatility.

    Returns a contract dict that travels with the position.
    """
    if target_dte is None:
        target_dte = config.OPTIONS["target_dte"]

    T = target_dte / 252
    premium = black_scholes(underlying_price, underlying_price, T, hist_vol, option_type)

    return {
        "option_type": option_type,
        "strike": underlying_price,   # At-the-money
        "premium_paid": premium,
        "dte_at_entry": target_dte,
        "hist_vol": hist_vol,
    }


def simulate_option_value(contract: dict, current_price: float, days_held: int) -> float:
    """
    Calculate current value of an option position using Black-Scholes.
    Called every day in the backtester to mark positions to market.
    """
    remaining_dte = max(contract["dte_at_entry"] - days_held, 0)
    T = remaining_dte / 252
    return black_scholes(
        current_price,
        contract["strike"],
        T,
        contract["hist_vol"],
        contract["option_type"],
    )


# ── Live Contract Selection ──────────────────────────────────────────────────

def get_live_contract(symbol: str, option_type: str, target_dte: int = None) -> dict | None:
    """
    Fetch a real options contract from Yahoo Finance for live trading.
    Finds the ATM contract with expiration closest to target_dte.

    Args:
        symbol:     Stock ticker (e.g., "AAPL")
        option_type: "call" or "put"
        target_dte: Desired days to expiration

    Returns:
        Contract dict with symbol, strike, expiry, ask price, etc.
        Returns None if options unavailable for this ticker.
    """
    if target_dte is None:
        target_dte = config.OPTIONS["target_dte"]

    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            logger.warning(f"No options available for {symbol}")
            return None

        # Find expiration closest to target DTE
        today = date.today()
        target_date = today + timedelta(days=target_dte)
        best_exp = min(
            expirations,
            key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d").date() - target_date).days)
        )

        chain = ticker.option_chain(best_exp)
        contracts = chain.calls if option_type == "call" else chain.puts

        if contracts.empty:
            logger.warning(f"Empty {option_type} chain for {symbol} expiring {best_exp}")
            return None

        # Current stock price
        hist = ticker.history(period="1d")
        if hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])

        # Find closest ATM strike
        contracts = contracts.copy()
        contracts["strike_diff"] = (contracts["strike"] - current_price).abs()
        atm = contracts.loc[contracts["strike_diff"].idxmin()]

        dte = (datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days
        ask = float(atm["ask"]) if float(atm["ask"]) > 0 else float(atm["lastPrice"])

        logger.info(
            f"  Options contract: {atm['contractSymbol']} | "
            f"{option_type.upper()} ${float(atm['strike']):.0f} exp {best_exp} "
            f"({dte} DTE) | ask: ${ask:.2f}"
        )

        return {
            "contract_symbol": atm["contractSymbol"],
            "underlying": symbol,
            "option_type": option_type,
            "strike": float(atm["strike"]),
            "expiration": best_exp,
            "dte": dte,
            "ask": ask,
            "bid": float(atm["bid"]),
            "last": float(atm["lastPrice"]),
            "implied_vol": float(atm["impliedVolatility"]) if float(atm["impliedVolatility"]) > 0 else 0.30,
            "underlying_price": current_price,
        }

    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {e}")
        return None
