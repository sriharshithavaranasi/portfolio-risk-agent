"""
Tests for agent/prices.py

Run from the project root:
    python -m pytest tests/test_prices.py -v

These tests make real network calls to yfinance.
They use stable, large-cap tickers unlikely to be delisted.
"""

import pytest
import pandas as pd

from agent.prices import PriceFetchError, fetch_prices, get_historical_prices, get_price


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------

def test_get_price_returns_positive_float():
    price = get_price("AAPL")
    assert isinstance(price, float)
    assert price > 0


def test_get_price_is_case_insensitive():
    price_upper = get_price("MSFT")
    price_lower = get_price("msft")
    assert price_upper == price_lower


def test_get_price_invalid_ticker_raises():
    with pytest.raises(PriceFetchError):
        get_price("INVALIDTICKER_XYZ")


# ---------------------------------------------------------------------------
# get_historical_prices
# ---------------------------------------------------------------------------

def test_get_historical_prices_returns_dataframe():
    df = get_historical_prices("AAPL", period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_historical_prices_has_close_column():
    df = get_historical_prices("AAPL", period="1mo")
    assert "Close" in df.columns


def test_get_historical_prices_index_is_datetime():
    df = get_historical_prices("AAPL", period="1mo")
    assert isinstance(df.index, pd.DatetimeIndex)


def test_get_historical_prices_1y_has_roughly_252_rows():
    df = get_historical_prices("AAPL", period="1y")
    # US markets have ~252 trading days per year; allow a reasonable range
    assert 200 <= len(df) <= 270


def test_get_historical_prices_invalid_period_raises():
    with pytest.raises(ValueError, match="Invalid period"):
        get_historical_prices("AAPL", period="99y")


def test_get_historical_prices_invalid_ticker_raises():
    with pytest.raises(PriceFetchError):
        get_historical_prices("INVALIDTICKER_XYZ", period="1mo")


# ---------------------------------------------------------------------------
# fetch_prices
# ---------------------------------------------------------------------------

def test_fetch_prices_returns_dict_with_all_tickers():
    holdings = [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
        {"ticker": "MSFT", "shares": 5,  "cost_basis": 280.0},
    ]
    prices = fetch_prices(holdings)

    assert "AAPL" in prices
    assert "MSFT" in prices


def test_fetch_prices_always_includes_benchmark():
    holdings = [{"ticker": "AAPL", "shares": 10, "cost_basis": 150.0}]
    prices = fetch_prices(holdings)

    # SPY is the default benchmark in config.py
    assert "SPY" in prices


def test_fetch_prices_dataframes_have_close_column():
    holdings = [{"ticker": "AAPL", "shares": 10, "cost_basis": 150.0}]
    prices = fetch_prices(holdings)

    for ticker, df in prices.items():
        assert "close" in df.columns, f"Missing 'close' column for {ticker}"


def test_fetch_prices_invalid_ticker_raises():
    holdings = [{"ticker": "INVALIDTICKER_XYZ", "shares": 10, "cost_basis": 100.0}]
    with pytest.raises(PriceFetchError):
        fetch_prices(holdings)
