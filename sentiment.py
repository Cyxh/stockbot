"""
SENTIMENT.PY — The news-reading brain.

This module reads news headlines about each stock and scores them
on a scale from very negative (-1.0) to very positive (+1.0).

HOW IT WORKS:
1. Takes raw news headlines from data_fetcher
2. Runs them through FinBERT (financial-domain BERT model, ~89% accuracy)
   if the transformers library is available, otherwise falls back to VADER
   (~70% accuracy on financial text), otherwise uses keyword-only analysis
3. Applies a custom financial keyword booster (words like "surge",
   "plummet", "beat earnings" get extra weight) in all cases
4. Averages everything into a single sentiment score per stock

WHY THIS MATTERS:
News drives short-term price movement. If 10 headlines are screaming
"Company X beats earnings expectations!", the stock will likely go up
before technical indicators even react.
"""

import logging
import re
from datetime import datetime

import config

logger = logging.getLogger(__name__)

# Try to import VADER; if not available, we fall back to keyword-only analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed — using keyword-only sentiment. "
                    "Install with: pip install vaderSentiment")

# Try to import FinBERT (transformers library).
# FinBERT is trained on financial news — significantly more accurate than
# VADER (89% vs 70%) for earnings/analyst/market text.
# Install with: pip install transformers torch
# Model auto-downloads (~500MB) on first use.
FINBERT_AVAILABLE = False
_finbert_pipeline = None

def _get_finbert():
    """Lazy-load FinBERT pipeline on first call to avoid startup delay."""
    global _finbert_pipeline, FINBERT_AVAILABLE
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _finbert_pipeline = hf_pipeline(
                "text-classification",
                model="yiyanghkust/finbert-tone",
                return_all_scores=True,
                device=-1,      # CPU (use 0 for CUDA GPU if available)
                truncation=True,
                max_length=512,
            )
        FINBERT_AVAILABLE = True
        logger.info("FinBERT loaded successfully — using financial-domain sentiment model")
        return _finbert_pipeline
    except Exception as e:
        logger.info(f"FinBERT not available ({e}) — falling back to VADER/keywords")
        FINBERT_AVAILABLE = False
        return None


# =============================================================================
# FINANCIAL KEYWORD LEXICON
# These boost or penalize sentiment beyond what generic VADER catches.
# Curated from financial news patterns.
# =============================================================================
BULLISH_KEYWORDS = {
    # Strong bullish
    "surge": 0.6, "surges": 0.6, "soar": 0.6, "soars": 0.6,
    "skyrocket": 0.7, "rally": 0.5, "rallies": 0.5,
    "breakout": 0.5, "all-time high": 0.6, "record high": 0.6,
    "beat expectations": 0.5, "beats expectations": 0.5,
    "beat estimates": 0.5, "beats estimates": 0.5,
    "strong earnings": 0.5, "earnings beat": 0.5,
    "revenue growth": 0.4, "profit growth": 0.4,
    "upgrade": 0.5, "upgraded": 0.5, "buy rating": 0.5,
    "outperform": 0.4, "bullish": 0.5,

    # Moderate bullish
    "gains": 0.3, "rises": 0.3, "climbs": 0.3, "jumps": 0.4,
    "positive": 0.2, "growth": 0.3, "expand": 0.2,
    "innovation": 0.2, "partnership": 0.2, "acquisition": 0.2,
    "dividend": 0.2, "buyback": 0.3, "share repurchase": 0.3,
    "strong demand": 0.3, "upbeat": 0.3, "optimistic": 0.3,
}

BEARISH_KEYWORDS = {
    # Strong bearish
    "crash": -0.7, "crashes": -0.7, "plunge": -0.6, "plunges": -0.6,
    "plummet": -0.6, "tank": -0.5, "tanks": -0.5,
    "collapse": -0.6, "sell-off": -0.5, "selloff": -0.5,
    "miss expectations": -0.5, "misses expectations": -0.5,
    "miss estimates": -0.5, "misses estimates": -0.5,
    "weak earnings": -0.5, "earnings miss": -0.5,
    "revenue decline": -0.4, "profit decline": -0.4,
    "downgrade": -0.5, "downgraded": -0.5, "sell rating": -0.5,
    "underperform": -0.4, "bearish": -0.5,

    # Moderate bearish
    "losses": -0.3, "falls": -0.3, "drops": -0.3, "declines": -0.3,
    "slump": -0.4, "tumble": -0.4, "sinks": -0.4,
    "negative": -0.2, "concerns": -0.2, "worried": -0.2,
    "lawsuit": -0.3, "investigation": -0.3, "fraud": -0.5,
    "layoffs": -0.3, "restructuring": -0.2, "debt": -0.2,
    "recession": -0.4, "inflation": -0.2, "tariff": -0.3,
    "bankruptcy": -0.7, "default": -0.5, "warning": -0.3,
}

ALL_KEYWORDS = {**BULLISH_KEYWORDS, **BEARISH_KEYWORDS}


def _keyword_score(text: str) -> float:
    """
    Score text using our custom financial keyword dictionary.
    Returns a value roughly in [-1, 1].
    """
    text_lower = text.lower()
    scores = []

    for keyword, weight in ALL_KEYWORDS.items():
        if keyword in text_lower:
            scores.append(weight)

    if not scores:
        return 0.0

    # Average of matched keywords, but cap the magnitude
    return max(min(sum(scores) / max(len(scores), 1), 1.0), -1.0)


def _vader_score(text: str) -> float:
    """
    Score text using VADER sentiment analyzer.
    Returns compound score in [-1, 1].
    """
    if not VADER_AVAILABLE:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def _finbert_score(text: str) -> float:
    """
    Score text using FinBERT financial sentiment model.
    Returns compound score in [-1, 1]: positive=bullish, negative=bearish.
    """
    pipe = _get_finbert()
    if pipe is None:
        return 0.0
    try:
        results = pipe(text[:512])[0]  # Truncate to model max length
        # Results: [{"label": "Positive", "score": 0.8}, {"label": "Negative", ...}, ...]
        scores = {r["label"]: r["score"] for r in results}
        positive = scores.get("Positive", 0.0)
        negative = scores.get("Negative", 0.0)
        # Net sentiment: +1 = fully positive, -1 = fully negative
        return float(positive - negative)
    except Exception:
        return 0.0


def score_headline(title: str, description: str = "") -> dict:
    """
    Score a single news headline using the best available model.

    Priority: FinBERT > VADER > keywords-only
    Keywords always contribute 40% weight as a domain-specific booster.

    Args:
        title:       The headline text
        description: Optional article description/snippet

    Returns:
        Dict with "vader_score", "keyword_score", "finbert_score", "combined_score"
    """
    full_text = f"{title} {description}".strip()

    if not full_text:
        return {"vader_score": 0.0, "keyword_score": 0.0,
                "finbert_score": 0.0, "combined_score": 0.0}

    keyword = _keyword_score(full_text)

    # Try FinBERT first (most accurate for financial text)
    finbert = _finbert_score(full_text)
    if FINBERT_AVAILABLE and finbert != 0.0:
        # FinBERT 60% + keywords 40%
        combined = 0.60 * finbert + 0.40 * keyword
        return {
            "vader_score":   0.0,
            "finbert_score": finbert,
            "keyword_score": keyword,
            "combined_score": float(max(min(combined, 1.0), -1.0)),
        }

    # Fall back to VADER
    vader = _vader_score(full_text)
    if VADER_AVAILABLE:
        combined = 0.60 * vader + 0.40 * keyword
    else:
        combined = keyword

    return {
        "vader_score":   vader,
        "finbert_score": 0.0,
        "keyword_score": keyword,
        "combined_score": float(max(min(combined, 1.0), -1.0)),
    }


def _recency_weight(published_str: str, halflife_hours: float) -> float:
    """
    Exponential decay weight based on article age.
    An article published `halflife_hours` ago gets weight 0.5.
    An article published today gets weight ~1.0.
    Very old articles still contribute but with reduced influence.
    """
    if not published_str or halflife_hours <= 0:
        return 1.0
    try:
        pub = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        now = datetime.now(pub.tzinfo)
        age_hours = max((now - pub).total_seconds() / 3600, 0)
        import math
        return math.exp(-math.log(2) * age_hours / halflife_hours)
    except Exception:
        return 1.0


def analyze(articles: list) -> dict:
    """
    Analyze a list of news articles and produce a sentiment signal.

    Args:
        articles: List of dicts with "title", "description", and "published" keys

    Returns:
        {
          "score":         float -1.0 to +1.0
          "num_articles":  int
          "article_scores": list of individual scores
          "confidence":    float 0.0 to 1.0
        }
    """
    if not articles:
        return {"score": 0.0, "num_articles": 0, "article_scores": [], "confidence": 0.0}

    # Log which sentiment model is active (once per session)
    model_name = "FinBERT" if FINBERT_AVAILABLE else ("VADER" if VADER_AVAILABLE else "keywords-only")
    logger.debug(f"Sentiment model in use: {model_name}")

    halflife = config.SENTIMENT.get("recency_halflife_hours", 24)
    article_scores = []

    for article in articles:
        result = score_headline(
            article.get("title", ""),
            article.get("description", ""),
        )
        result["title"]  = article.get("title", "")[:80]
        result["weight"] = _recency_weight(article.get("published", ""), halflife)
        article_scores.append(result)

    # Weighted average (recent articles count more)
    total_weight    = sum(a["weight"] for a in article_scores)
    if total_weight == 0:
        total_weight = 1.0
    avg_score = sum(a["combined_score"] * a["weight"] for a in article_scores) / total_weight

    num_articles = len(article_scores)
    min_articles = config.SENTIMENT["min_articles"]

    # Confidence: article count × directional agreement
    count_confidence = min(num_articles / (min_articles * 2), 1.0)

    combined_scores = [a["combined_score"] for a in article_scores]
    if num_articles > 1:
        positive  = sum(1 for s in combined_scores if s > 0.05)
        negative  = sum(1 for s in combined_scores if s < -0.05)
        dominant  = max(positive, negative)
        agreement = dominant / num_articles
    else:
        agreement = 0.5

    confidence = count_confidence * agreement

    if num_articles < min_articles:
        avg_score *= num_articles / min_articles

    return {
        "score":         max(min(avg_score, 1.0), -1.0),
        "num_articles":  num_articles,
        "article_scores": article_scores,
        "confidence":    confidence,
    }
