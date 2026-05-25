"""
Tests for agent/news.py

All tests use mocked HTTP responses — no real API calls, no API keys required.

Run from the project root:
    python -m pytest tests/test_news.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.news import (
    NewsAPIProvider,
    NewsArticle,
    NewsProviderError,
    PolygonNewsProvider,
    TickerSummary,
    _select_tickers,
    fetch_news,
    summarize_news,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

HOLDINGS = [
    {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
    {"ticker": "MSFT", "shares": 5,  "cost_basis": 280.0},
    {"ticker": "NVDA", "shares": 3,  "cost_basis": 450.0},
]

WEIGHTS = {"AAPL": 0.50, "MSFT": 0.30, "NVDA": 0.20}

def make_article(ticker="AAPL", headline="Big news", source="Bloomberg") -> NewsArticle:
    return NewsArticle(
        ticker=ticker,
        headline=headline,
        summary="Article body text.",
        url="https://example.com/article",
        published_at="2024-01-15T10:30:00Z",
        source=source,
    )


# ---------------------------------------------------------------------------
# NewsArticle
# ---------------------------------------------------------------------------

class TestNewsArticle:
    def test_to_dict_has_expected_keys(self):
        article = make_article()
        d = article.to_dict()
        assert set(d.keys()) == {"ticker", "headline", "summary", "url", "published_at", "source"}

    def test_to_dict_values_match(self):
        article = make_article(ticker="MSFT", headline="Microsoft quarterly beat")
        d = article.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["headline"] == "Microsoft quarterly beat"


# ---------------------------------------------------------------------------
# TickerSummary
# ---------------------------------------------------------------------------

class TestTickerSummary:
    def test_to_dict_has_expected_keys(self):
        summary = TickerSummary(ticker="AAPL", article_count=2, digest="Headline 1 | Headline 2")
        d = summary.to_dict()
        assert set(d.keys()) == {"ticker", "article_count", "digest", "articles"}

    def test_articles_serialised_as_list_of_dicts(self):
        article = make_article()
        summary = TickerSummary(ticker="AAPL", article_count=1, digest="...", articles=[article])
        d = summary.to_dict()
        assert isinstance(d["articles"], list)
        assert isinstance(d["articles"][0], dict)


# ---------------------------------------------------------------------------
# PolygonNewsProvider
# ---------------------------------------------------------------------------

class TestPolygonNewsProvider:
    def test_is_configured_true_when_key_set(self):
        with patch("agent.news.POLYGON_API_KEY", "fake_key"):
            provider = PolygonNewsProvider()
            assert provider.is_configured() is True

    def test_is_configured_false_when_key_missing(self):
        with patch("agent.news.POLYGON_API_KEY", None):
            provider = PolygonNewsProvider()
            assert provider.is_configured() is False

    def test_fetch_parses_polygon_response(self):
        fake_response = {
            "status": "OK",
            "results": [
                {
                    "title":         "Apple hits record high",
                    "description":   "AAPL surged after earnings.",
                    "article_url":   "https://example.com/1",
                    "published_utc": "2024-01-15T10:00:00Z",
                    "publisher":     {"name": "Reuters"},
                }
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.news.POLYGON_API_KEY", "fake_key"), \
             patch("requests.get", return_value=mock_resp):
            provider = PolygonNewsProvider()
            articles = provider.fetch_for_ticker("AAPL", limit=3)

        assert len(articles) == 1
        assert articles[0].ticker == "AAPL"
        assert articles[0].headline == "Apple hits record high"
        assert articles[0].source == "Reuters"

    def test_fetch_raises_on_polygon_error_status(self):
        fake_response = {"status": "ERROR", "error": "Unknown ticker"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.news.POLYGON_API_KEY", "fake_key"), \
             patch("requests.get", return_value=mock_resp):
            provider = PolygonNewsProvider()
            with pytest.raises(NewsProviderError, match="Polygon API error"):
                provider.fetch_for_ticker("BADTICKER", limit=3)

    def test_fetch_raises_on_network_error(self):
        import requests as req
        with patch("agent.news.POLYGON_API_KEY", "fake_key"), \
             patch("requests.get", side_effect=req.ConnectionError("timeout")):
            provider = PolygonNewsProvider()
            with pytest.raises(NewsProviderError, match="Polygon request failed"):
                provider.fetch_for_ticker("AAPL", limit=3)


# ---------------------------------------------------------------------------
# NewsAPIProvider
# ---------------------------------------------------------------------------

class TestNewsAPIProvider:
    def test_is_configured_true_when_key_set(self):
        with patch("agent.news.NEWS_API_KEY", "fake_key"):
            provider = NewsAPIProvider()
            assert provider.is_configured() is True

    def test_fetch_parses_newsapi_response(self):
        fake_response = {
            "status": "ok",
            "articles": [
                {
                    "title":       "Microsoft beats estimates",
                    "description": "MSFT up 4% after earnings.",
                    "url":         "https://example.com/2",
                    "publishedAt": "2024-01-15T09:00:00Z",
                    "source":      {"name": "CNBC"},
                }
            ],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.news.NEWS_API_KEY", "fake_key"), \
             patch("requests.get", return_value=mock_resp):
            provider = NewsAPIProvider()
            articles = provider.fetch_for_ticker("MSFT", limit=3)

        assert len(articles) == 1
        assert articles[0].ticker == "MSFT"
        assert articles[0].headline == "Microsoft beats estimates"
        assert articles[0].source == "CNBC"

    def test_fetch_raises_on_newsapi_error(self):
        fake_response = {"status": "error", "message": "apiKey is invalid"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("agent.news.NEWS_API_KEY", "fake_key"), \
             patch("requests.get", return_value=mock_resp):
            provider = NewsAPIProvider()
            with pytest.raises(NewsProviderError, match="NewsAPI error"):
                provider.fetch_for_ticker("AAPL", limit=3)


# ---------------------------------------------------------------------------
# _select_tickers
# ---------------------------------------------------------------------------

class TestSelectTickers:
    def test_respects_top_n_limit(self):
        # TOP_N_NEWS is 5; we have 3 holdings so all 3 should come back
        tickers = _select_tickers(HOLDINGS, weights=None)
        assert len(tickers) <= 5

    def test_sorts_by_weight_descending(self):
        tickers = _select_tickers(HOLDINGS, weights=WEIGHTS)
        # AAPL (0.50) should come before MSFT (0.30) before NVDA (0.20)
        assert tickers[0] == "AAPL"
        assert tickers[1] == "MSFT"
        assert tickers[2] == "NVDA"

    def test_no_weights_preserves_holdings_order(self):
        tickers = _select_tickers(HOLDINGS, weights=None)
        assert tickers == ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# summarize_news
# ---------------------------------------------------------------------------

class TestSummarizeNews:
    def test_groups_by_ticker(self):
        articles = [
            make_article("AAPL", "Apple news 1"),
            make_article("AAPL", "Apple news 2"),
            make_article("MSFT", "Microsoft news 1"),
        ]
        summaries = summarize_news(articles)
        tickers = [s.ticker for s in summaries]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_article_count_is_correct(self):
        articles = [make_article("AAPL")] * 3
        summaries = summarize_news(articles)
        assert summaries[0].article_count == 3

    def test_digest_joins_headlines(self):
        articles = [
            make_article("AAPL", "Headline A"),
            make_article("AAPL", "Headline B"),
        ]
        summaries = summarize_news(articles)
        assert "Headline A" in summaries[0].digest
        assert "Headline B" in summaries[0].digest

    def test_empty_articles_returns_empty_list(self):
        assert summarize_news([]) == []


# ---------------------------------------------------------------------------
# fetch_news (integration-style, provider mocked)
# ---------------------------------------------------------------------------

class TestFetchNews:
    def test_returns_list_of_news_articles(self):
        mock_provider = MagicMock()
        mock_provider.fetch_for_ticker.return_value = [make_article("AAPL")]

        with patch("agent.news._get_provider", return_value=mock_provider):
            articles = fetch_news(HOLDINGS[:1])

        assert isinstance(articles, list)
        assert isinstance(articles[0], NewsArticle)

    def test_bad_ticker_does_not_abort_run(self):
        # First ticker raises, second succeeds — result should contain one article
        def side_effect(ticker, limit):
            if ticker == "AAPL":
                raise NewsProviderError("API error")
            return [make_article(ticker)]

        mock_provider = MagicMock()
        mock_provider.fetch_for_ticker.side_effect = side_effect

        with patch("agent.news._get_provider", return_value=mock_provider):
            articles = fetch_news(HOLDINGS[:2])  # AAPL fails, MSFT succeeds

        assert len(articles) == 1
        assert articles[0].ticker == "MSFT"

    def test_no_provider_configured_raises(self):
        with patch("agent.news.POLYGON_API_KEY", None), \
             patch("agent.news.NEWS_API_KEY", None):
            with pytest.raises(RuntimeError, match="No news provider"):
                fetch_news(HOLDINGS)
