"""
News module — fetches and summarizes recent headlines for portfolio holdings.

Architecture
------------
All providers implement the NewsProvider ABC, so swapping data sources requires
only adding a new class — the public fetch_news() function never changes.

Provider priority (first with a configured API key wins):
    1. PolygonNewsProvider  — Polygon.io /v2/reference/news  (preferred)
    2. NewsAPIProvider      — newsapi.org /v2/everything      (fallback)

Public API
----------
    fetch_news(holdings, weights, articles_per_ticker) -> list[NewsArticle]
    summarize_news(articles)                           -> list[TickerSummary]
"""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from config import NEWS_API_KEY, POLYGON_API_KEY, TOP_N_NEWS

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """A single news article, normalised across providers."""
    ticker: str
    headline: str
    summary: str          # article description / excerpt
    url: str
    published_at: str     # ISO-8601 UTC string, e.g. "2024-01-15T10:30:00Z"
    source: str           # publisher name

    def to_dict(self) -> dict:
        return {
            "ticker":       self.ticker,
            "headline":     self.headline,
            "summary":      self.summary,
            "url":          self.url,
            "published_at": self.published_at,
            "source":       self.source,
        }


@dataclass
class TickerSummary:
    """Aggregated headline digest for one ticker."""
    ticker: str
    article_count: int
    digest: str           # short paragraph combining all headlines
    articles: list[NewsArticle] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker":        self.ticker,
            "article_count": self.article_count,
            "digest":        self.digest,
            "articles":      [a.to_dict() for a in self.articles],
        }


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class NewsProvider(ABC):
    """
    Abstract base class for news providers.

    Any new provider (Bloomberg, Finnhub, etc.) only needs to implement
    fetch_for_ticker() — the rest of the pipeline stays the same.
    """

    @abstractmethod
    def fetch_for_ticker(self, ticker: str, limit: int) -> list[NewsArticle]:
        """
        Fetch up to `limit` recent articles for the given ticker.

        Args:
            ticker: Stock symbol, e.g. "AAPL".
            limit:  Max number of articles to return.

        Returns:
            List of NewsArticle objects, newest first.

        Raises:
            NewsProviderError: On API or parse failure.
        """

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the required API key is present in config."""


class NewsProviderError(Exception):
    """Raised when a news provider returns an error or unexpected response."""


# ---------------------------------------------------------------------------
# Polygon.io provider
# ---------------------------------------------------------------------------

class PolygonNewsProvider(NewsProvider):
    """
    Fetches news from Polygon.io.

    Docs: https://polygon.io/docs/stocks/get_v2_reference_news
    Free tier: 5 API calls / minute, delayed data.
    Paid tiers: real-time, higher rate limits.

    Endpoint: GET https://api.polygon.io/v2/reference/news
    Params:   ticker, limit, order=desc, sort=published_utc, apiKey=...
    """

    BASE_URL = "https://api.polygon.io/v2/reference/news"

    def is_configured(self) -> bool:
        return bool(POLYGON_API_KEY)

    def fetch_for_ticker(self, ticker: str, limit: int) -> list[NewsArticle]:
        params = {
            "ticker":   ticker.upper(),
            "limit":    limit,
            "order":    "desc",
            "sort":     "published_utc",
            "apiKey":   POLYGON_API_KEY,
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NewsProviderError(f"Polygon request failed for '{ticker}': {exc}") from exc

        data = response.json()

        if data.get("status") == "ERROR":
            raise NewsProviderError(
                f"Polygon API error for '{ticker}': {data.get('error', 'unknown error')}"
            )

        articles = []
        for item in data.get("results", []):
            articles.append(NewsArticle(
                ticker=      ticker.upper(),
                headline=    item.get("title", ""),
                summary=     item.get("description", ""),
                url=         item.get("article_url", ""),
                published_at=item.get("published_utc", ""),
                source=      item.get("publisher", {}).get("name", "Unknown"),
            ))

        return articles


# ---------------------------------------------------------------------------
# NewsAPI fallback provider
# ---------------------------------------------------------------------------

class NewsAPIProvider(NewsProvider):
    """
    Fetches news from newsapi.org.

    Docs: https://newsapi.org/docs/endpoints/everything
    Free tier: developer use only, 100 requests/day, no commercial use.

    Endpoint: GET https://newsapi.org/v2/everything
    Headers:  Authorization: Bearer {NEWS_API_KEY}
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def is_configured(self) -> bool:
        return bool(NEWS_API_KEY)

    def fetch_for_ticker(self, ticker: str, limit: int) -> list[NewsArticle]:
        headers = {"Authorization": f"Bearer {NEWS_API_KEY}"}
        params = {
            "q":        ticker.upper(),
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": limit,
        }

        try:
            response = requests.get(self.BASE_URL, headers=headers, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NewsProviderError(f"NewsAPI request failed for '{ticker}': {exc}") from exc

        data = response.json()

        if data.get("status") != "ok":
            raise NewsProviderError(
                f"NewsAPI error for '{ticker}': {data.get('message', 'unknown error')}"
            )

        articles = []
        for item in data.get("articles", []):
            articles.append(NewsArticle(
                ticker=      ticker.upper(),
                headline=    item.get("title", ""),
                summary=     item.get("description", "") or "",
                url=         item.get("url", ""),
                published_at=item.get("publishedAt", ""),
                source=      (item.get("source") or {}).get("name", "Unknown"),
            ))

        return articles


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _get_provider() -> NewsProvider:
    """
    Return the first configured provider in priority order.

    Priority: Polygon → NewsAPI

    Raises:
        RuntimeError: If no provider has an API key configured.
    """
    candidates: list[NewsProvider] = [
        PolygonNewsProvider(),
        NewsAPIProvider(),
    ]

    for provider in candidates:
        if provider.is_configured():
            return provider

    raise RuntimeError(
        "No news provider is configured. "
        "Set POLYGON_API_KEY or NEWS_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news(
    holdings: list[dict],
    weights: dict[str, float] | None = None,
    articles_per_ticker: int = 3,
) -> list[NewsArticle]:
    """
    Fetch recent news for the top N holdings by portfolio weight.

    Automatically selects the best available provider (Polygon first, then NewsAPI).

    Args:
        holdings:            List of holding dicts from loader.py.
        weights:             Dict of {ticker: weight} from risk.py.
                             If None, the first TOP_N_NEWS tickers are used.
        articles_per_ticker: How many articles to fetch per ticker.

    Returns:
        Flat list of NewsArticle objects, grouped by ticker, newest first.
    """
    tickers = _select_tickers(holdings, weights)
    provider = _get_provider()

    articles: list[NewsArticle] = []
    for ticker in tickers:
        try:
            fetched = provider.fetch_for_ticker(ticker, limit=articles_per_ticker)
            articles.extend(fetched)
        except NewsProviderError as exc:
            # Log and continue — one bad ticker shouldn't abort the whole run
            print(f"Warning: could not fetch news for {ticker}: {exc}")

    return articles


def summarize_news(articles: list[NewsArticle]) -> list[TickerSummary]:
    """
    Group articles by ticker and produce a short digest for each.

    The digest is a plain-text paragraph listing all headlines — ready to
    drop into the Anthropic prompt in memo.py.

    Args:
        articles: Output of fetch_news().

    Returns:
        List of TickerSummary objects, one per ticker.
    """
    # Group articles by ticker, preserving insertion order
    grouped: dict[str, list[NewsArticle]] = {}
    for article in articles:
        grouped.setdefault(article.ticker, []).append(article)

    summaries = []
    for ticker, ticker_articles in grouped.items():
        headlines = [a.headline for a in ticker_articles if a.headline]
        # Join headlines into a readable digest paragraph
        digest = " | ".join(headlines) if headlines else "No headlines available."
        summaries.append(TickerSummary(
            ticker=        ticker,
            article_count= len(ticker_articles),
            digest=        digest,
            articles=      ticker_articles,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_tickers(
    holdings: list[dict],
    weights: dict[str, float] | None,
) -> list[str]:
    """
    Return the top TOP_N_NEWS tickers to fetch news for.

    If weights are provided, sort by weight descending so we cover the
    positions that matter most to the portfolio first.
    """
    tickers = [h["ticker"].upper() for h in holdings]

    if weights:
        tickers = sorted(tickers, key=lambda t: weights.get(t, 0), reverse=True)

    return tickers[:TOP_N_NEWS]
