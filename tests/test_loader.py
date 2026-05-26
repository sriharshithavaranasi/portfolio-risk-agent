"""
Tests for agent/loader.py

Run from the project root:
    python -m pytest tests/test_loader.py -v
"""

import json
import textwrap
from pathlib import Path

import pytest

from agent.loader import load_portfolio


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_loads_valid_csv(self, tmp_path):
        f = tmp_path / "portfolio.csv"
        f.write_text("ticker,shares,cost_basis\nAAPL,10,150.00\nMSFT,5,280.00\n")
        holdings = load_portfolio(str(f))
        assert len(holdings) == 2
        assert holdings[0]["ticker"] == "AAPL"
        assert holdings[0]["shares"] == 10.0
        assert holdings[0]["cost_basis"] == 150.0

    def test_normalises_ticker_to_uppercase(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ticker,shares,cost_basis\naapl,10,150\n")
        holdings = load_portfolio(str(f))
        assert holdings[0]["ticker"] == "AAPL"

    def test_strips_whitespace_from_ticker(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ticker,shares,cost_basis\n  AAPL  ,10,150\n")
        holdings = load_portfolio(str(f))
        assert holdings[0]["ticker"] == "AAPL"

    def test_strips_whitespace_from_column_names(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text(" ticker , shares , cost_basis \nAAPL,10,150\n")
        holdings = load_portfolio(str(f))
        assert holdings[0]["ticker"] == "AAPL"

    def test_raises_on_missing_column(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ticker,shares\nAAPL,10\n")   # missing cost_basis
        with pytest.raises(ValueError, match="cost_basis"):
            load_portfolio(str(f))

    def test_raises_on_zero_shares(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ticker,shares,cost_basis\nAAPL,0,150\n")
        with pytest.raises(ValueError, match="shares"):
            load_portfolio(str(f))

    def test_raises_on_negative_cost_basis(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ticker,shares,cost_basis\nAAPL,10,-50\n")
        with pytest.raises(ValueError, match="cost_basis"):
            load_portfolio(str(f))


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

class TestLoadJSON:
    def test_loads_valid_json(self, tmp_path):
        data = [
            {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
            {"ticker": "MSFT", "shares": 5,  "cost_basis": 280.0},
        ]
        f = tmp_path / "portfolio.json"
        f.write_text(json.dumps(data))
        holdings = load_portfolio(str(f))
        assert len(holdings) == 2
        assert holdings[1]["ticker"] == "MSFT"

    def test_normalises_ticker_to_uppercase(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps([{"ticker": "aapl", "shares": 10, "cost_basis": 150}]))
        holdings = load_portfolio(str(f))
        assert holdings[0]["ticker"] == "AAPL"

    def test_raises_if_not_a_list(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps({"ticker": "AAPL", "shares": 10, "cost_basis": 150}))
        with pytest.raises(ValueError, match="list"):
            load_portfolio(str(f))

    def test_raises_on_zero_shares(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text(json.dumps([{"ticker": "AAPL", "shares": 0, "cost_basis": 150}]))
        with pytest.raises(ValueError, match="shares"):
            load_portfolio(str(f))


# ---------------------------------------------------------------------------
# Shared edge cases
# ---------------------------------------------------------------------------

class TestLoadPortfolioEdgeCases:
    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_portfolio(str(tmp_path / "does_not_exist.csv"))

    def test_raises_on_unsupported_extension(self, tmp_path):
        f = tmp_path / "p.xlsx"
        f.write_text("irrelevant")
        with pytest.raises(ValueError, match="Unsupported"):
            load_portfolio(str(f))

    def test_raises_on_empty_portfolio(self, tmp_path):
        f = tmp_path / "p.csv"
        f.write_text("ticker,shares,cost_basis\n")
        with pytest.raises(ValueError, match="empty"):
            load_portfolio(str(f))

    def test_sample_portfolio_csv_loads_correctly(self):
        # Verify the committed sample file is always valid
        holdings = load_portfolio("data/portfolio.csv")
        assert len(holdings) > 0
        for h in holdings:
            assert h["ticker"]
            assert h["shares"] > 0
            assert h["cost_basis"] > 0
