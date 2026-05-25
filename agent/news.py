"""
Fetches recent news headlines for the top N holdings by portfolio weight.

Uses the NewsAPI (newsapi.org). Requires NEWS_API_KEY in .env.
"""

import requests
from config import NEWS_API_KEY, TOP_N_NEWS

NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_news(holdings: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    """
    Fetch recent headlines for the top N holdings by weight.

    Args:
        holdings: list of holding dicts from loader.py
        weights: optional dict of {ticker: weight}; if None, uses equal weighting

    Returns:
        [
            {"ticker": "AAPL", "headline": "...", "url": "..."},
            ...
        ]
    """
    top_tickers = _get_top_tickers(holdings, weights)

    headlines = []
    for ticker in top_tickers:
        articles = _fetch_ticker_news(ticker)
        headlines.extend(articles)

    return headlines


def _get_top_tickers(holdings: list[dict], weights: dict[str, float] | None) -> list[str]:
    """Return the top N tickers sorted by weight descending."""
    # TODO: if weights provided, sort holdings by weight and take top TOP_N_NEWS
    # TODO: if weights is None, just take the first TOP_N_NEWS holdings
    raise NotImplementedError


def _fetch_ticker_news(ticker: str) -> list[dict]:
    """Call NewsAPI for a single ticker and return a list of headline dicts."""
    # TODO: build query params: q=ticker, sortBy=publishedAt, language=en, pageSize=3
    # TODO: set Authorization header with NEWS_API_KEY
    # TODO: GET NEWS_API_URL, raise on HTTP error
    # TODO: parse response["articles"] into [{"ticker": ticker, "headline": ..., "url": ...}]
    raise NotImplementedError
