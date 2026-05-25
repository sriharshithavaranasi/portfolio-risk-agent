"""
Tests for agent/risk.py

All tests use synthetic price data — no network calls.

Run from the project root:
    python -m pytest tests/test_risk.py -v
"""

import numpy as np
import pandas as pd
import pytest

from agent.risk import (
    compute_beta,
    compute_concentration,
    compute_daily_pnl,
    compute_max_drawdown,
    compute_portfolio_weights,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def make_prices(data: dict[str, list[float]]) -> dict[str, pd.DataFrame]:
    """Build a prices dict with a DatetimeIndex from plain lists of closes."""
    dates = pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq="B")
    return {
        ticker: pd.DataFrame({"close": closes}, index=dates)
        for ticker, closes in data.items()
    }


HOLDINGS = [
    {"ticker": "AAPL", "shares": 10, "cost_basis": 100.0},
    {"ticker": "MSFT", "shares": 20, "cost_basis": 200.0},
]

# Last close: AAPL=200, MSFT=100
# Market values: AAPL=10×200=2000, MSFT=20×100=2000 → equal weights (0.5, 0.5)
EQUAL_PRICES = make_prices({
    "AAPL": [180, 185, 190, 195, 200],
    "MSFT": [90,  92,  95,  98,  100],
    "SPY":  [400, 402, 404, 406, 408],
})


# ---------------------------------------------------------------------------
# compute_portfolio_weights
# ---------------------------------------------------------------------------

class TestComputePortfolioWeights:
    def test_weights_sum_to_one(self):
        weights = compute_portfolio_weights(HOLDINGS, EQUAL_PRICES)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_equal_market_value_gives_equal_weights(self):
        # AAPL: 10 shares × $200 = $2000; MSFT: 20 shares × $100 = $2000 → both 50%
        weights = compute_portfolio_weights(HOLDINGS, EQUAL_PRICES)
        assert abs(weights["AAPL"] - 0.5) < 1e-9
        assert abs(weights["MSFT"] - 0.5) < 1e-9

    def test_larger_position_has_higher_weight(self):
        prices = make_prices({
            "AAPL": [100, 100, 300],   # 10 shares × $300 = $3000
            "MSFT": [100, 100, 100],   # 20 shares × $100 = $2000
            "SPY":  [400, 400, 400],
        })
        weights = compute_portfolio_weights(HOLDINGS, prices)
        assert weights["AAPL"] > weights["MSFT"]

    def test_missing_ticker_raises(self):
        with pytest.raises(ValueError, match="No price data"):
            compute_portfolio_weights(
                [{"ticker": "TSLA", "shares": 5, "cost_basis": 100.0}],
                EQUAL_PRICES,
            )


# ---------------------------------------------------------------------------
# compute_concentration
# ---------------------------------------------------------------------------

class TestComputeConcentration:
    def test_returns_expected_keys(self):
        result = compute_concentration(HOLDINGS, EQUAL_PRICES)
        assert {"weights", "hhi", "largest_holding", "overweight_flags"} == set(result.keys())

    def test_equal_weights_hhi(self):
        # Two equal holdings: HHI = 0.5² + 0.5² = 0.5
        result = compute_concentration(HOLDINGS, EQUAL_PRICES)
        assert abs(result["hhi"] - 0.5) < 0.001

    def test_fully_concentrated_hhi_is_one(self):
        # Single holding → weight = 1.0 → HHI = 1.0² = 1.0
        prices = make_prices({"AAPL": [100, 110, 120], "SPY": [400, 410, 420]})
        holdings = [{"ticker": "AAPL", "shares": 10, "cost_basis": 100.0}]
        result = compute_concentration(holdings, prices)
        assert abs(result["hhi"] - 1.0) < 1e-9

    def test_overweight_flag_triggered(self):
        # AAPL at 50% weight — exceeds default threshold of 20%
        result = compute_concentration(HOLDINGS, EQUAL_PRICES)
        assert "AAPL" in result["overweight_flags"]
        assert "MSFT" in result["overweight_flags"]

    def test_no_overweight_flags_when_well_diversified(self):
        # Four equal holdings → each 25% → above 20% threshold
        # Use five holdings to get each below 20%
        holdings = [{"ticker": f"T{i}", "shares": 10, "cost_basis": 100.0} for i in range(6)]
        prices = make_prices({f"T{i}": [100, 100, 100] for i in range(6)})
        prices["SPY"] = make_prices({"SPY": [400, 400, 400]})["SPY"]
        result = compute_concentration(holdings, prices)
        assert result["overweight_flags"] == []


# ---------------------------------------------------------------------------
# compute_beta
# ---------------------------------------------------------------------------

class TestComputeBeta:
    def test_beta_is_float(self):
        beta = compute_beta(HOLDINGS, EQUAL_PRICES)
        assert isinstance(beta, float)

    def test_portfolio_identical_to_benchmark_has_beta_one(self):
        # When portfolio returns == benchmark returns, beta must be 1.0
        prices = make_prices({
            "AAPL": [100, 101, 102, 103, 104],
            "MSFT": [100, 101, 102, 103, 104],
            "SPY":  [100, 101, 102, 103, 104],
        })
        beta = compute_beta(HOLDINGS, prices)
        assert abs(beta - 1.0) < 0.01

    def test_beta_missing_benchmark_raises(self):
        prices_no_spy = {k: v for k, v in EQUAL_PRICES.items() if k != "SPY"}
        with pytest.raises(ValueError, match="Benchmark"):
            compute_beta(HOLDINGS, prices_no_spy)

    def test_amplified_returns_give_beta_greater_than_one(self):
        # Portfolio returns are 2× benchmark returns → beta ≈ 2.0
        spy = [100, 102, 101, 104, 103]
        # Each stock doubles the benchmark daily move
        aapl = [200, 204, 202, 208, 206]
        msft = [200, 204, 202, 208, 206]
        prices = make_prices({"AAPL": aapl, "MSFT": msft, "SPY": spy})
        beta = compute_beta(HOLDINGS, prices)
        assert beta > 1.5


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    def test_returns_negative_float(self):
        dd = compute_max_drawdown(HOLDINGS, EQUAL_PRICES)
        # Prices are rising in EQUAL_PRICES so drawdown should be 0 or negative
        assert dd <= 0.0

    def test_no_drawdown_when_prices_only_rise(self):
        prices = make_prices({
            "AAPL": [100, 110, 120, 130, 140],
            "MSFT": [200, 210, 220, 230, 240],
            "SPY":  [400, 410, 420, 430, 440],
        })
        dd = compute_max_drawdown(HOLDINGS, prices)
        assert dd == 0.0

    def test_known_drawdown_is_correct(self):
        # Portfolio peaks at day 2 ($3000), then falls to $2400 at day 3.
        # Drawdown = (2400 - 3000) / 3000 = -0.20
        #
        # AAPL: 10 shares. MSFT: 20 shares.
        # Day 0: AAPL=100, MSFT=100 → value = 10×100 + 20×100 = 3000  (peak)
        # Day 1: AAPL=80,  MSFT=80  → value = 10×80  + 20×80  = 2400
        # Day 2: AAPL=90,  MSFT=90  → value = 10×90  + 20×90  = 2700
        prices = make_prices({
            "AAPL": [100, 80, 90],
            "MSFT": [100, 80, 90],
            "SPY":  [400, 400, 400],
        })
        dd = compute_max_drawdown(HOLDINGS, prices)
        assert abs(dd - (-0.20)) < 0.001


# ---------------------------------------------------------------------------
# compute_daily_pnl
# ---------------------------------------------------------------------------

class TestComputeDailyPnl:
    def test_returns_expected_keys(self):
        result = compute_daily_pnl(HOLDINGS, EQUAL_PRICES)
        assert {"pnl_dollars", "pnl_pct"} == set(result.keys())

    def test_known_pnl_is_correct(self):
        # AAPL: 10 shares, $195 → $200  (+$5 each = +$50)
        # MSFT: 20 shares, $98  → $100  (+$2 each = +$40)
        # Total PnL = $90
        # Prev value = 10×195 + 20×98 = 1950 + 1960 = 3910
        result = compute_daily_pnl(HOLDINGS, EQUAL_PRICES)
        assert result["pnl_dollars"] == pytest.approx(90.0, abs=0.01)
        assert result["pnl_pct"] == pytest.approx(90 / 3910, rel=1e-4)

    def test_flat_prices_give_zero_pnl(self):
        prices = make_prices({
            "AAPL": [100, 100, 100],
            "MSFT": [200, 200, 200],
            "SPY":  [400, 400, 400],
        })
        result = compute_daily_pnl(HOLDINGS, prices)
        assert result["pnl_dollars"] == 0.0
        assert result["pnl_pct"] == 0.0

    def test_insufficient_data_raises(self):
        prices = make_prices({
            "AAPL": [100],
            "MSFT": [200],
            "SPY":  [400],
        })
        with pytest.raises(ValueError, match="2 days"):
            compute_daily_pnl(HOLDINGS, prices)
