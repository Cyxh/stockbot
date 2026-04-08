"""
STRATEGY.PY — The decision-making brain.

This is where technical analysis and sentiment analysis get combined into
actual trading decisions: BUY, SELL, or HOLD.

THE DECISION PROCESS:
1. Get the technical score (-1 to +1)
2. Get the sentiment score (-1 to +1)
3. Combine them with configurable weights
4. If the combined score exceeds our confidence threshold → trade
5. Apply risk management rules (position sizing, max positions)

CONSERVATIVE APPROACH:
The bot only trades when BOTH brains agree in the same direction.
A strong technical signal + neutral sentiment = no trade.
This reduces the number of trades but dramatically improves win rate.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import config
from technical import analyze as technical_analyze
from sentiment import analyze as sentiment_analyze
from data_fetcher import fetch_fundamentals, is_near_earnings, fetch_vix

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    A trading signal — the bot's conclusion about what to do with a stock.
    """
    symbol: str
    action: str              # "BUY", "SELL", or "HOLD"
    confidence: float        # 0.0 to 1.0 — how sure the bot is
    technical_score: float   # -1.0 to 1.0
    sentiment_score: float   # -1.0 to 1.0
    combined_score: float    # -1.0 to 1.0
    position_size_pct: float # What % of portfolio to allocate
    reasoning: list = field(default_factory=list)  # Human-readable explanations


def combine_scores(
    technical_score: float,
    sentiment_score: float,
    sentiment_confidence: float,
    tech_weight: float = 0.60,
    sent_weight: float = 0.40,
) -> float:
    """
    Combine technical and sentiment scores into one signal.

    If sentiment data is weak (low confidence), we lean more on technicals.
    If sentiment data is strong, we give it full weight.
    """
    # Adjust sentiment weight by its confidence
    effective_sent_weight = sent_weight * sentiment_confidence
    effective_tech_weight = tech_weight + (sent_weight - effective_sent_weight)

    # Normalize weights to sum to 1
    total_weight = effective_tech_weight + effective_sent_weight
    if total_weight == 0:
        return 0.0

    combined = (
        (technical_score * effective_tech_weight +
         sentiment_score * effective_sent_weight)
        / total_weight
    )
    return combined


def calculate_position_size(
    confidence: float,
    atr_pct: float = 0.02,
    portfolio_value: float = 100_000,
    vix_level: float = 20.0,
    realized_vol: float = 0.20,
) -> float:
    """
    Determine how much money to put into a trade.

    Uses four factors:
    1. Confidence — higher confidence = larger position
    2. Realized volatility — inverse-vol sizing (risk parity)
    3. VIX level — scale down as market fear rises (VIX_SCALE_LEVELS in config)
    4. ATR fallback — if realized vol unavailable, uses ATR
    """
    max_pct = config.MAX_POSITION_SIZE_PCT

    # VIX scaling: reduce position size as market fear rises
    vix_multiplier = 1.0
    for vix_thresh, multiplier in sorted(
        getattr(config, "VIX_SCALE_LEVELS", [(999, 1.0)]),
        key=lambda x: x[0]
    ):
        if vix_level < vix_thresh:
            vix_multiplier = multiplier
            break

    # Base size scales with confidence (0.5 to 1.0 of max)
    base_pct = max_pct * (0.5 + 0.5 * confidence)

    # Inverse-volatility sizing: each position contributes equal risk
    target_vol = getattr(config, "INVERSE_VOL_TARGET_VOL", 0.20)
    if getattr(config, "USE_INVERSE_VOL_SIZING", True) and realized_vol > 0.01:
        vol_scalar = target_vol / realized_vol
        vol_scalar = min(max(vol_scalar, 0.3), 2.5)
    else:
        # Fallback to ATR-based sizing
        vol_scalar = 0.02 / max(atr_pct, 0.005)
        vol_scalar = min(max(vol_scalar, 0.3), 1.5)

    final_pct = base_pct * vol_scalar * vix_multiplier

    # Never exceed max position size
    return min(final_pct, max_pct)


def generate_signal(
    symbol: str,
    price_data,
    articles: list,
    portfolio_value: float = 100_000,
    current_positions: dict = None,
    in_bull_regime: bool = True,
    cooldown_symbols: set = None,
    benchmark_data=None,
    vix_level: float = None,
    sector_momentum: float = None,
) -> Signal:
    """
    The main decision function. Analyzes a stock and returns a Signal.

    Args:
        symbol:           Stock ticker
        price_data:       DataFrame of OHLCV data
        articles:         List of news articles
        portfolio_value:  Current portfolio value
        current_positions: Dict of currently held positions

    Returns:
        Signal object with BUY/SELL/HOLD decision
    """
    if current_positions is None:
        current_positions = {}
    if cooldown_symbols is None:
        cooldown_symbols = set()

    # Fetch live VIX if not provided
    if vix_level is None:
        try:
            vix_level = fetch_vix()
        except Exception:
            vix_level = 20.0

    reasoning = []

    # ── Step 1: Technical Analysis ──────────────────────────────────
    # Pass benchmark_data (SPY) so relative strength can be computed.
    # For SPY itself, benchmark_data should be None to avoid self-comparison.
    bench = benchmark_data if symbol != "SPY" else None
    # Fetch live fundamental signals (analyst revisions + quality metrics).
    # Returns {} if unavailable — technical.analyze() handles missing data gracefully.
    fundamentals = {}
    if symbol not in getattr(config, "REGIME_ONLY_SYMBOLS", set()):
        try:
            fundamentals = fetch_fundamentals(symbol)
        except Exception:
            pass
    tech_result = technical_analyze(price_data, benchmark_df=bench,
                                     live_fundamentals=fundamentals if fundamentals else None,
                                     sector_momentum=sector_momentum)
    tech_score = tech_result["score"]
    indicators = tech_result["indicators"]
    reasoning.append(f"Technical score: {tech_score:+.3f}")

    # Log key indicators
    if indicators:
        adx_val = indicators.get("adx", 0)
        rs_val  = indicators.get("relative_strength", None)
        m12_val = indicators.get("momentum_12m", None)
        h52_val = indicators.get("high_52w_proximity", None)
        reasoning.append(
            f"  RSI: {indicators.get('rsi', 0):.1f} | "
            f"MACD vs Signal: {indicators.get('macd', 0):.3f} vs {indicators.get('macd_signal', 0):.3f} | "
            f"Vol ratio: {indicators.get('volume_ratio', 0):.2f} | "
            f"ADX: {adx_val:.1f}"
        )
        if rs_val is not None:
            reasoning.append(
                f"  RS vs SPY: {rs_val:+.3f} | "
                f"12M mom: {m12_val:+.3f} | "
                f"52W high prox: {h52_val:+.3f}"
            )

    # ── Step 2: Sentiment Analysis ──────────────────────────────────
    sent_result = sentiment_analyze(articles)
    sent_score = sent_result["score"]
    sent_confidence = sent_result["confidence"]
    reasoning.append(
        f"Sentiment score: {sent_score:+.3f} "
        f"({sent_result['num_articles']} articles, "
        f"confidence: {sent_confidence:.2f})"
    )

    # ── Step 3: Combine Scores ──────────────────────────────────────
    combined = combine_scores(tech_score, sent_score, sent_confidence)
    reasoning.append(f"Combined score: {combined:+.3f}")

    # ── Step 4: Determine Action ────────────────────────────────────
    abs_combined = abs(combined)
    confidence = abs_combined  # Confidence = magnitude of signal

    # Alignment bonus: when both brains agree, boost confidence
    if (tech_score > 0 and sent_score > 0) or (tech_score < 0 and sent_score < 0):
        alignment_bonus = 0.1
        confidence = min(confidence + alignment_bonus, 1.0)
        reasoning.append("  ✓ Technical and sentiment AGREE — confidence boosted")
    elif (tech_score > 0.2 and sent_score < -0.2) or (tech_score < -0.2 and sent_score > 0.2):
        confidence *= 0.7  # Conflicting signals → reduce confidence
        reasoning.append("  ✗ Technical and sentiment DISAGREE — confidence reduced")

    # Decision thresholds
    min_conf = config.MIN_CONFIDENCE
    already_holding = symbol in current_positions
    num_positions = len(current_positions)
    opts = config.OPTIONS

    if combined > 0 and confidence >= min_conf and not already_holding:
        # ── BUY / CALL SIGNAL ── (only in bull regime)
        if symbol in cooldown_symbols:
            action = "HOLD"
            reasoning.append(f"  -> HOLD (cooldown active — recent stop-loss on {symbol})")
        elif not in_bull_regime:
            action = "HOLD"
            reasoning.append("  -> HOLD (bull signal blocked: bear market regime)")
        elif num_positions >= config.MAX_OPEN_POSITIONS:
            action = "HOLD"
            reasoning.append(f"  Would BUY but max positions ({config.MAX_OPEN_POSITIONS}) reached")
        else:
            # Earnings blackout: avoid opening within N days of earnings
            blackout_days = getattr(config, "EARNINGS_BLACKOUT_DAYS", 0)
            near_earnings = False
            if blackout_days > 0 and symbol not in getattr(config, "REGIME_ONLY_SYMBOLS", set()):
                try:
                    near_earnings = is_near_earnings(symbol, blackout_days)
                except Exception:
                    near_earnings = False
            if near_earnings:
                action = "HOLD"
                reasoning.append(f"  -> HOLD (earnings blackout: within {blackout_days} days of earnings)")
            elif opts["enabled"] and opts["use_calls_on_bullish"]:
                action = "CALL"
                reasoning.append(f"  -> CALL option (confidence {confidence:.2f} >= threshold {min_conf})")
            else:
                action = "BUY"
                reasoning.append(f"  -> BUY (confidence {confidence:.2f} >= threshold {min_conf})")

    elif combined < 0 and confidence >= min_conf:
        if already_holding:
            # ── SELL existing stock position ──
            action = "SELL"
            reasoning.append(f"  → SELL (confidence {confidence:.2f} >= threshold {min_conf})")
        elif (opts["enabled"] and opts["use_puts_on_bearish"] and
              symbol == "SPY" and not in_bull_regime):
            # ── PUT SIGNAL — SPY puts only in confirmed bear regime ──
            if num_positions >= config.MAX_OPEN_POSITIONS:
                action = "HOLD"
                reasoning.append(f"  Would PUT but max positions ({config.MAX_OPEN_POSITIONS}) reached")
            else:
                action = "PUT"
                reasoning.append(f"  → PUT option (confidence {confidence:.2f} >= threshold {min_conf})")
        else:
            action = "HOLD"
            reasoning.append("  → HOLD (bearish but no actionable position)")

    else:
        action = "HOLD"
        if abs_combined < min_conf:
            reasoning.append(f"  → HOLD (confidence {confidence:.2f} < threshold {min_conf})")
        elif combined > 0 and already_holding:
            reasoning.append("  → HOLD (already in position)")
        else:
            reasoning.append("  → HOLD (no actionable signal)")

    # ── Step 5: Position Sizing ─────────────────────────────────────
    atr_pct = indicators.get("atr_pct", 0.02) if indicators else 0.02
    realized_vol = indicators.get("realized_vol", 0.20) if indicators else 0.20
    pos_size = calculate_position_size(confidence, atr_pct, portfolio_value, vix_level, realized_vol)

    if action == "BUY":
        reasoning.append(f"  Position size: {pos_size:.1%} of portfolio "
                         f"(${portfolio_value * pos_size:,.0f})")

    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        technical_score=tech_score,
        sentiment_score=sent_score,
        combined_score=combined,
        position_size_pct=pos_size,
        reasoning=reasoning,
    )
