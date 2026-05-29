"""News Sentiment Module — Phase 2.

Sources:
  1. NSE corporate announcements (RSS)
  2. Moneycontrol Markets RSS
  3. Economic Times Markets RSS

Scoring:
  - Primary:  VADER (fast, financial-aware, no GPU needed)
  - Fallback: keyword rule-based scorer

Output: sentiment score per symbol per fetch in range [-1.0, +1.0]
  -1.0 = very bearish, 0.0 = neutral, +1.0 = very bullish

Integration:
  - Score < -0.3 → block LONG signals
  - Score >  0.3 → block SHORT signals
  - abs(score) < 0.3 → neutral, allow all signals

Usage:
    from production.data.news_sentiment import NewsSentiment
    ns = NewsSentiment()
    score = ns.score("NIFTY50")       # -1.0 to +1.0
    ns.should_block(signal_direction, "NIFTY50")  # True/False
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

IST         = ZoneInfo("Asia/Kolkata")
CACHE_FILE  = Path("data/.news_cache.json")
CACHE_TTL_S = 300   # refresh every 5 minutes

# RSS feeds
_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
]

# Symbol → search keywords
_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "NIFTY50":     ["nifty", "sensex", "market", "index", "equity"],
    "NIFTYBANK":   ["bank nifty", "banknifty", "banking", "bank index"],
    "RELIANCE":    ["reliance", "ril"],
    "HDFCBANK":    ["hdfc bank", "hdfcbank"],
    "ICICIBANK":   ["icici bank", "icicibank"],
    "SBIN":        ["sbi", "state bank", "sbin"],
    "TCS":         ["tcs", "tata consultancy"],
    "INFY":        ["infosys", "infy"],
    "KOTAKBANK":   ["kotak bank", "kotak mahindra"],
    "LT":          ["larsen", "l&t", "lt"],
    "ITC":         ["itc"],
    "AXISBANK":    ["axis bank", "axisbank"],
}

# Positive/negative financial keywords for fallback scorer
_POS_WORDS = {
    "surge", "rally", "gain", "profit", "growth", "record", "high",
    "beat", "strong", "positive", "upgrade", "buy", "bullish", "up",
    "rise", "jump", "boom", "recover", "outperform", "inflow",
}
_NEG_WORDS = {
    "crash", "fall", "loss", "decline", "drop", "weak", "miss",
    "downgrade", "sell", "bearish", "down", "slump", "plunge",
    "negative", "concern", "risk", "warning", "outflow", "default",
}


class NewsSentiment:
    """Fetches news headlines and scores sentiment per symbol."""

    def __init__(self) -> None:
        self._cache: dict = {}
        self._cache_time: float = 0.0
        self._vader = None
        self._init_scorer()

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(self, symbol: str) -> float:
        """Return sentiment score [-1, +1] for a symbol. Cached 5 minutes."""
        self._refresh_if_stale()
        short = self._normalise_symbol(symbol)
        return float(self._cache.get(short, 0.0))

    def should_block(self, direction: int, symbol: str) -> bool:
        """Return True if news sentiment opposes the trade direction."""
        s = self.score(symbol)
        if direction == 1 and s < -0.3:   # going LONG but news is bearish
            return True
        if direction == -1 and s > 0.3:   # going SHORT but news is bullish
            return True
        return False

    def latest_headlines(self, symbol: str, n: int = 5) -> list[dict]:
        """Return the n most recent scored headlines for a symbol."""
        self._refresh_if_stale()
        short = self._normalise_symbol(symbol)
        return self._headline_cache.get(short, [])[:n]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _init_scorer(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.debug("News sentiment: VADER scorer loaded")
        except ImportError:
            logger.debug("News sentiment: using keyword fallback scorer")

    def _refresh_if_stale(self) -> None:
        if time.time() - self._cache_time < CACHE_TTL_S:
            return
        self._fetch_and_score()

    def _fetch_and_score(self) -> None:
        import feedparser
        headlines: list[tuple[str, str]] = []   # (title, published)

        for feed_url in _FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:30]:
                    title = entry.get("title", "")
                    pub   = entry.get("published", "")
                    if title:
                        headlines.append((title.lower(), pub))
            except Exception as e:
                logger.debug("Feed {} failed: {}", feed_url, e)

        if not headlines:
            logger.debug("News sentiment: no headlines fetched")
            return

        scores: dict[str, list[float]] = {}
        headline_map: dict[str, list[dict]] = {}

        for title, pub in headlines:
            sent  = self._score_text(title)
            syms  = self._match_symbols(title)
            if not syms:
                syms = ["NIFTY50"]   # broad market news → apply to NIFTY50

            for sym in syms:
                scores.setdefault(sym, []).append(sent)
                headline_map.setdefault(sym, []).append({
                    "title":     title,
                    "score":     round(sent, 3),
                    "published": pub,
                })

        # Average score per symbol, clipped to [-1, +1]
        self._cache = {
            sym: round(max(-1.0, min(1.0, sum(v) / len(v))), 3)
            for sym, v in scores.items()
        }
        self._headline_cache = headline_map
        self._cache_time = time.time()
        logger.info("News sentiment refreshed — {} symbols scored from {} headlines",
                    len(self._cache), len(headlines))

    def _score_text(self, text: str) -> float:
        if self._vader:
            vs = self._vader.polarity_scores(text)
            return float(vs["compound"])   # already in [-1, +1]
        # Keyword fallback
        words = set(text.lower().split())
        pos = len(words & _POS_WORDS)
        neg = len(words & _NEG_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _match_symbols(self, title: str) -> list[str]:
        matched = []
        for sym, keywords in _SYMBOL_KEYWORDS.items():
            if any(kw in title for kw in keywords):
                matched.append(sym)
        return matched

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        return symbol.replace("NSE:", "").replace("-EQ", "").replace("-INDEX", "")


# ── Singleton ─────────────────────────────────────────────────────────────────

_sentiment: NewsSentiment | None = None

def get_sentiment() -> NewsSentiment:
    global _sentiment
    if _sentiment is None:
        _sentiment = NewsSentiment()
    return _sentiment
