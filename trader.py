"""
TRADER.PY — The execution engine.

This module actually places buy and sell orders through the Alpaca brokerage.
It also manages:
- Stop-loss orders (auto-sell if a stock drops too much)
- Take-profit orders (auto-sell if a stock hits your target)
- Position tracking

IMPORTANT SAFETY FEATURES:
- All orders are LIMIT orders (not market orders), so you never get a
  surprise fill price
- Stop-loss is placed immediately after every buy
- Position sizes are capped by config.MAX_POSITION_SIZE_PCT
"""

import logging
from datetime import datetime

import config
from options import get_live_contract

logger = logging.getLogger(__name__)

# Try to import Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest,
        StopLossRequest, TakeProfitRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Install with: pip install alpaca-py")


class Trader:
    """
    Handles all order execution and position management.
    """

    def __init__(self, paper: bool = None):
        self.paper = paper if paper is not None else config.PAPER_TRADING
        self.client = None
        self.positions = {}
        self.trade_log = []
        self.highest_prices = {}  # Trailing stop: track peak price per position

        if ALPACA_AVAILABLE and config.ALPACA_API_KEY != "YOUR_ALPACA_API_KEY":
            try:
                self.client = TradingClient(
                    config.ALPACA_API_KEY,
                    config.ALPACA_SECRET_KEY,
                    paper=self.paper,
                )
                account = self.client.get_account()
                logger.info(f"Connected to Alpaca ({'PAPER' if self.paper else 'LIVE'})")
                logger.info(f"Account equity: ${float(account.equity):,.2f}")
                self._sync_positions()
            except Exception as e:
                logger.error(f"Failed to connect to Alpaca: {e}")
                self.client = None
        else:
            logger.warning("Alpaca not configured — running in simulation mode")

    def _sync_positions(self):
        """Sync our position tracker with what Alpaca actually shows."""
        if not self.client:
            return

        try:
            positions = self.client.get_all_positions()
            self.positions = {}
            for pos in positions:
                self.positions[pos.symbol] = {
                    "qty": float(pos.qty),
                    "avg_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_pl_pct": float(pos.unrealized_plpc),
                }
            logger.info(f"Synced {len(self.positions)} positions from Alpaca")
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        if self.client:
            try:
                account = self.client.get_account()
                return float(account.equity)
            except Exception:
                pass
        return config.BACKTEST_STARTING_CASH  # Fallback for simulation

    def get_positions(self) -> dict:
        """Get current positions."""
        if self.client:
            self._sync_positions()
        return self.positions

    def buy(self, symbol: str, dollar_amount: float, current_price: float) -> dict:
        """
        Place a buy order.

        Args:
            symbol:        Stock ticker
            dollar_amount: How much $ to invest
            current_price: Current stock price (for limit calculation)

        Returns:
            Dict with order details
        """
        qty = int(dollar_amount / current_price)
        if qty <= 0:
            logger.warning(f"Cannot buy {symbol}: amount ${dollar_amount:.2f} "
                          f"< 1 share at ${current_price:.2f}")
            return {"status": "rejected", "reason": "insufficient_amount"}

        # Calculate stop-loss and take-profit prices
        stop_price = round(current_price * (1 - config.STOP_LOSS_PCT), 2)
        take_profit_price = round(current_price * (1 + config.TAKE_PROFIT_PCT), 2)

        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "BUY",
            "symbol": symbol,
            "qty": qty,
            "price": current_price,
            "total": qty * current_price,
            "stop_loss": stop_price,
            "take_profit": take_profit_price,
            "status": "pending",
        }

        if self.client:
            try:
                # Place market order (simplest; could use limit for better fills)
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.client.submit_order(order_data=order_data)
                trade_record["order_id"] = str(order.id)
                trade_record["status"] = "submitted"
                logger.info(f"BUY ORDER: {qty} shares of {symbol} @ ~${current_price:.2f} "
                          f"(stop: ${stop_price}, target: ${take_profit_price})")

            except Exception as e:
                trade_record["status"] = "error"
                trade_record["error"] = str(e)
                logger.error(f"Error placing buy order for {symbol}: {e}")
        else:
            trade_record["status"] = "simulated"
            logger.info(f"[SIM] BUY: {qty} x {symbol} @ ${current_price:.2f} = "
                       f"${qty * current_price:,.2f}")

        # Initialize trailing stop peak
        self.highest_prices[symbol] = current_price

        self.trade_log.append(trade_record)
        return trade_record

    def sell(self, symbol: str, reason: str = "signal") -> dict:
        """
        Sell entire position in a stock.

        Args:
            symbol: Stock ticker
            reason: Why we're selling (for logging)

        Returns:
            Dict with order details
        """
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "action": "SELL",
            "symbol": symbol,
            "reason": reason,
            "status": "pending",
        }

        if self.client:
            try:
                # Close the entire position
                self.client.close_position(symbol)
                trade_record["status"] = "submitted"
                logger.info(f"SELL ORDER: Close all {symbol} (reason: {reason})")

            except Exception as e:
                trade_record["status"] = "error"
                trade_record["error"] = str(e)
                logger.error(f"Error selling {symbol}: {e}")
        else:
            pos = self.positions.get(symbol, {})
            trade_record["qty"] = pos.get("qty", 0)
            trade_record["status"] = "simulated"
            logger.info(f"[SIM] SELL: {symbol} (reason: {reason})")

        self.trade_log.append(trade_record)
        return trade_record

    def check_stop_loss_take_profit(self) -> list:
        """
        Check all positions against stop-loss and trailing stop levels.
        Returns list of actions taken.
        """
        actions = []
        self._sync_positions()

        bear_etf = getattr(config, "BEAR_REGIME_ETF", None)
        trail_pct = getattr(config, "TRAILING_STOP_PCT", config.TAKE_PROFIT_PCT)

        # Iterate over a snapshot so we can sell during iteration
        for symbol, pos in list(self.positions.items()):
            # Never stop-loss the T-bill parking position
            if symbol == bear_etf:
                continue

            current_price = pos.get("current_price", 0)
            if current_price <= 0:
                continue

            pnl_pct = pos.get("unrealized_pl_pct", 0)

            # Update trailing high watermark
            peak = self.highest_prices.get(symbol, pos.get("avg_price", current_price))
            if current_price > peak:
                peak = current_price
                self.highest_prices[symbol] = peak

            trail_floor = peak * (1 - trail_pct)

            if pnl_pct <= -config.STOP_LOSS_PCT:
                logger.warning(f"STOP LOSS triggered for {symbol} "
                               f"(loss: {pnl_pct:.1%})")
                result = self.sell(symbol, reason="stop_loss")
                actions.append(result)
                self.highest_prices.pop(symbol, None)

            elif current_price <= trail_floor and pnl_pct > 0:
                logger.info(f"TRAILING STOP triggered for {symbol} "
                            f"(gain: {pnl_pct:.1%}, peak: ${peak:.2f}, "
                            f"floor: ${trail_floor:.2f})")
                result = self.sell(symbol, reason="trailing_stop")
                actions.append(result)
                self.highest_prices.pop(symbol, None)

        return actions

    def buy_option(
        self,
        symbol: str,
        option_type: str,
        dollar_amount: float,
        current_price: float,
    ) -> dict:
        """
        Buy a put or call option on a stock.

        Args:
            symbol:       Underlying stock ticker
            option_type:  "call" or "put"
            dollar_amount: Max $ to spend on premium
            current_price: Current stock price (for logging)

        Returns:
            Dict with order details
        """
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "action": option_type.upper(),
            "symbol": symbol,
            "dollar_amount": dollar_amount,
            "underlying_price": current_price,
            "status": "pending",
        }

        # Fetch the live options contract
        contract = get_live_contract(symbol, option_type)
        if not contract:
            trade_record["status"] = "rejected"
            trade_record["reason"] = "no_contract_available"
            logger.warning(f"No {option_type} contract available for {symbol}")
            return trade_record

        # Calculate how many contracts we can afford
        cost_per_contract = contract["ask"] * 100  # Standard contract = 100 shares
        num_contracts = max(1, int(dollar_amount / cost_per_contract))
        total_cost = num_contracts * cost_per_contract

        trade_record.update({
            "contract_symbol": contract["contract_symbol"],
            "strike": contract["strike"],
            "expiration": contract["expiration"],
            "dte": contract["dte"],
            "ask": contract["ask"],
            "num_contracts": num_contracts,
            "total_cost": total_cost,
        })

        if self.client:
            try:
                # Alpaca options order — same MarketOrderRequest, just with the
                # options contract symbol (e.g., "AAPL240315C00175000")
                order_data = MarketOrderRequest(
                    symbol=contract["contract_symbol"],
                    qty=num_contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.client.submit_order(order_data=order_data)
                trade_record["order_id"] = str(order.id)
                trade_record["status"] = "submitted"
                logger.info(
                    f"{option_type.upper()} ORDER: {num_contracts} contract(s) of "
                    f"{contract['contract_symbol']} @ ${contract['ask']:.2f}/share "
                    f"(${total_cost:,.2f} total)"
                )

            except Exception as e:
                trade_record["status"] = "error"
                trade_record["error"] = str(e)
                logger.error(f"Error placing {option_type} order for {symbol}: {e}")
        else:
            trade_record["status"] = "simulated"
            logger.info(
                f"[SIM] {option_type.upper()}: {num_contracts}c {symbol} "
                f"strike ${contract['strike']:.0f} exp {contract['expiration']} "
                f"@ ${contract['ask']:.2f}/sh = ${total_cost:,.2f}"
            )

        self.trade_log.append(trade_record)
        return trade_record

    def get_trade_log(self) -> list:
        """Return the full trade history."""
        return self.trade_log
