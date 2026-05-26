"""
Tests for agent/report.py

All tests use synthetic price data and a pre-built RiskMemo — no network calls.

Run from the project root:
    python -m pytest tests/test_report.py -v
"""

import pandas as pd
import pytest

from agent.memo import RiskMemo
from agent.report import (
    MacroContext,
    PortfolioReport,
    WatchlistItem,
    _all_risk_flags,
    _build_concentration,
    _build_macro_context,
    _build_top_movers,
    _build_watchlist,
    build_report,
    render_html,
    render_markdown,
)
from config import BENCHMARK_TICKER


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_prices(data: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq="B")
    return {
        t: pd.DataFrame({"close": closes}, index=dates)
        for t, closes in data.items()
    }


HOLDINGS = [
    {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
    {"ticker": "MSFT", "shares": 5,  "cost_basis": 280.0},
]

# AAPL last close 200, MSFT last close 100
# Market values: AAPL=2000, MSFT=500 → weights ~0.80, 0.20
PRICES = make_prices({
    "AAPL": [180, 185, 190, 195, 200],
    "MSFT": [95,  96,  97,  98,  100],
    "SPY":  [400, 402, 404, 406, 408],
})

MEMO = RiskMemo(
    overall_risk_level=  "MEDIUM",
    daily_pnl_dollars=   -300.0,
    daily_pnl_pct=       -0.012,
    portfolio_beta=       1.10,
    max_drawdown_pct=    -0.15,
    hhi_score=            0.68,
    largest_position=    "AAPL",
    concentration_flags= ["AAPL"],
    risk_flags=          ["High concentration in AAPL"],
    news_highlights=     [{"ticker": "AAPL", "digest": "Apple beats earnings"}],
    analyst_commentary=  "Portfolio is heavily concentrated in AAPL.",
    recommendations=     ["Trim AAPL to below 20%"],
)


# ---------------------------------------------------------------------------
# _build_top_movers
# ---------------------------------------------------------------------------

class TestBuildTopMovers:
    def test_returns_mover_per_holding(self):
        movers = _build_top_movers(HOLDINGS, PRICES)
        tickers = [m.ticker for m in movers]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_sorted_by_absolute_change_descending(self):
        movers = _build_top_movers(HOLDINGS, PRICES)
        changes = [abs(m.daily_change_pct) for m in movers]
        assert changes == sorted(changes, reverse=True)

    def test_direction_correct_for_rising_price(self):
        movers = _build_top_movers(HOLDINGS, PRICES)
        aapl = next(m for m in movers if m.ticker == "AAPL")
        # AAPL goes 195 → 200, so direction should be "up"
        assert aapl.direction == "up"

    def test_dollar_pnl_uses_shares(self):
        movers = _build_top_movers(HOLDINGS, PRICES)
        aapl = next(m for m in movers if m.ticker == "AAPL")
        # (200 - 195) * 10 shares = $50
        assert abs(aapl.daily_change_dollars - 50.0) < 0.01

    def test_flat_price_gives_flat_direction(self):
        flat_prices = make_prices({
            "AAPL": [100, 100, 100],
            "MSFT": [200, 200, 200],
            "SPY":  [400, 400, 400],
        })
        movers = _build_top_movers(HOLDINGS, flat_prices)
        assert all(m.direction == "flat" for m in movers)


# ---------------------------------------------------------------------------
# _all_risk_flags
# ---------------------------------------------------------------------------

class TestAllRiskFlags:
    def test_includes_memo_risk_flags(self):
        flags = _all_risk_flags(MEMO)
        assert "High concentration in AAPL" in flags

    def test_adds_concentration_flag_for_overweight_tickers(self):
        flags = _all_risk_flags(MEMO)
        assert any("AAPL" in f and "concentration" in f for f in flags)

    def test_no_duplicate_flags(self):
        memo_with_dup = RiskMemo(
            **{**MEMO.__dict__,
               "risk_flags": ["AAPL exceeds 20% concentration threshold"]},
        )
        flags = _all_risk_flags(memo_with_dup)
        matching = [f for f in flags if "AAPL" in f and "20%" in f]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# _build_concentration
# ---------------------------------------------------------------------------

class TestBuildConcentration:
    def test_rows_sorted_by_weight_descending(self):
        from agent.risk import compute_portfolio_weights
        weights = compute_portfolio_weights(HOLDINGS, PRICES)
        rows = _build_concentration(HOLDINGS, PRICES, weights)
        w_values = [r.weight for r in rows]
        assert w_values == sorted(w_values, reverse=True)

    def test_overweight_flag_set_correctly(self):
        from agent.risk import compute_portfolio_weights
        weights = compute_portfolio_weights(HOLDINGS, PRICES)
        rows = _build_concentration(HOLDINGS, PRICES, weights)
        aapl = next(r for r in rows if r.ticker == "AAPL")
        # AAPL weight is ~0.80 → should be flagged
        assert aapl.flagged is True

    def test_market_value_matches_shares_times_price(self):
        from agent.risk import compute_portfolio_weights
        weights = compute_portfolio_weights(HOLDINGS, PRICES)
        rows = _build_concentration(HOLDINGS, PRICES, weights)
        aapl = next(r for r in rows if r.ticker == "AAPL")
        assert abs(aapl.market_value - 200.0 * 10) < 0.01


# ---------------------------------------------------------------------------
# _build_macro_context
# ---------------------------------------------------------------------------

class TestBuildMacroContext:
    def test_returns_macro_context(self):
        mc = _build_macro_context(PRICES)
        assert isinstance(mc, MacroContext)

    def test_benchmark_ticker_is_correct(self):
        mc = _build_macro_context(PRICES)
        assert mc.benchmark_ticker == BENCHMARK_TICKER

    def test_regime_is_valid_value(self):
        mc = _build_macro_context(PRICES)
        assert mc.regime in {"Bull", "Bear", "Sideways"}

    def test_daily_pct_sign_correct_for_rising_benchmark(self):
        mc = _build_macro_context(PRICES)
        # SPY goes 406 → 408, so daily pct should be positive
        assert mc.benchmark_daily_pct > 0

    def test_missing_benchmark_returns_unknown_regime(self):
        no_spy = {k: v for k, v in PRICES.items() if k != BENCHMARK_TICKER}
        mc = _build_macro_context(no_spy)
        assert mc.regime == "Unknown"


# ---------------------------------------------------------------------------
# _build_watchlist
# ---------------------------------------------------------------------------

class TestBuildWatchlist:
    def test_overweight_holding_appears_in_watchlist(self):
        from agent.risk import compute_portfolio_weights
        weights = compute_portfolio_weights(HOLDINGS, PRICES)
        items = _build_watchlist(HOLDINGS, PRICES, weights, MEMO)
        tickers = [i.ticker for i in items]
        # AAPL is ~80% → overweight → should be on watchlist
        assert "AAPL" in tickers

    def test_news_ticker_appears_in_watchlist(self):
        from agent.risk import compute_portfolio_weights
        weights = compute_portfolio_weights(HOLDINGS, PRICES)
        items = _build_watchlist(HOLDINGS, PRICES, weights, MEMO)
        aapl = next((i for i in items if i.ticker == "AAPL"), None)
        assert aapl is not None
        assert any("news" in r.lower() for r in aapl.reasons)


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_returns_portfolio_report(self):
        report = build_report(MEMO, HOLDINGS, PRICES)
        assert isinstance(report, PortfolioReport)

    def test_risk_level_matches_memo(self):
        report = build_report(MEMO, HOLDINGS, PRICES)
        assert report.risk_level == "MEDIUM"

    def test_all_sections_populated(self):
        report = build_report(MEMO, HOLDINGS, PRICES)
        assert report.top_movers
        assert report.risk_flags
        assert report.concentration
        assert report.recommendations


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def setup_method(self):
        self.report = build_report(MEMO, HOLDINGS, PRICES)
        self.md     = render_markdown(self.report)

    def test_contains_all_section_headers(self):
        for section in ["Executive Summary", "Top Movers", "Risk Flags",
                        "Portfolio Concentration", "Macro Context",
                        "Watchlist", "Recommendations"]:
            assert f"## {section}" in self.md, f"Missing section: {section}"

    def test_contains_all_tickers(self):
        assert "AAPL" in self.md
        assert "MSFT" in self.md

    def test_risk_level_in_output(self):
        assert "MEDIUM" in self.md

    def test_recommendation_in_output(self):
        assert "Trim AAPL" in self.md

    def test_news_headline_in_output(self):
        assert "Apple beats earnings" in self.md


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------

class TestRenderHtml:
    def setup_method(self):
        self.report = build_report(MEMO, HOLDINGS, PRICES)
        self.html   = render_html(self.report)

    def test_is_valid_html_skeleton(self):
        assert "<!DOCTYPE html>" in self.html
        assert "<html" in self.html
        assert "</html>" in self.html

    def test_contains_all_section_titles(self):
        for section in ["Executive Summary", "Top Movers", "Risk Flags",
                        "Portfolio Concentration", "Macro Context",
                        "Watchlist", "Recommendations"]:
            assert section in self.html, f"Missing section: {section}"

    def test_risk_level_badge_present(self):
        assert "MEDIUM" in self.html

    def test_all_css_is_inline(self):
        # A <style> block would break most email clients
        assert "<style>" not in self.html

    def test_recommendation_in_output(self):
        assert "Trim AAPL" in self.html

    def test_no_unescaped_ampersands_in_table_cells(self):
        # Raw & in HTML is invalid; should be &amp;
        # P&L appears in headers as P&amp;L
        assert "P&L" not in self.html or "P&amp;L" in self.html
