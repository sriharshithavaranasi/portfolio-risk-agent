"""
Market data module — fetches current and historical stock prices via yfinance.

Public API:
    get_price(ticker)                        → float
    get_historical_prices(ticker, period)    → pd.DataFrame
    fetch_prices(holdings)                   → dict[str, pd.DataFrame]
"""

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from config import BENCHMARK_TICKER, LOOKBACK_DAYS


class PriceFetchError(Exception):
    """Raised when price data cannot be retrieved for a ticker."""


def get_price(ticker: str) -> float:
    """
    Return the most recent closing price for a single ticker.

    Args:
        ticker: Stock symbol, e.g. "AAPL".

    Returns:
        Latest closing price as a float.

    Raises:
        PriceFetchError: If the ticker is invalid or data is unavailable.
    """
    ticker = ticker.upper().strip()
    data = yf.Ticker(ticker)

    # fast_info is lightweight — no full history download needed
    price = data.fast_info.get("last_price")

    if price is None:
        raise PriceFetchError(
            f"Could not fetch price for '{ticker}'. "
            "Check that the ticker symbol is correct."
        )

    return float(price)


def get_historical_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Return daily OHLCV history for a single ticker.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        period: How far back to fetch. Accepts yfinance period strings:
                "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max".

    Returns:
        DataFrame with a DatetimeIndex and columns:
        Open, High, Low, Close, Volume, Dividends, Stock Splits.

    Raises:
        PriceFetchError: If the ticker is invalid or the period returns no data.
    """
    ticker = ticker.upper().strip()

    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in valid_periods:
        raise ValueError(
            f"Invalid period '{period}'. Choose from: {', '.join(sorted(valid_periods))}"
        )

    data = yf.Ticker(ticker).history(period=period)

    if data.empty:
        raise PriceFetchError(
            f"No historical data returned for '{ticker}' with period='{period}'. "
            "The ticker may be delisted or misspelled."
        )

    return data


def fetch_prices(holdings: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Download LOOKBACK_DAYS of daily close prices for every holding plus the benchmark.

    This is the portfolio-level wrapper used by risk.py. It calls
    get_historical_prices() for each ticker individually so errors on one
    ticker don't silently drop others.

    Args:
        holdings: List of holding dicts from loader.py.
                  Each dict must have a "ticker" key.

    Returns:
        Dict mapping ticker symbol → DataFrame of daily closes, e.g.:
        {
            "AAPL": DataFrame(...),
            "SPY":  DataFrame(...),   ← benchmark always included
        }

    Raises:
        PriceFetchError: If any ticker fails to fetch.
    """
    tickers = [h["ticker"].upper().strip() for h in holdings]

    # Always include the benchmark so risk.py can compute beta
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)

    # Convert lookback days to a yfinance-compatible period string.
    # yfinance doesn't accept arbitrary day counts, so we use the start= parameter.
    start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()

    prices: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).history(start=start_date)
        except Exception as exc:
            raise PriceFetchError(f"yfinance error fetching '{ticker}': {exc}") from exc

        if raw.empty:
            raise PriceFetchError(
                f"No data returned for '{ticker}' since {start_date}. "
                "Check the ticker symbol or try a shorter lookback period."
            )

        # Keep only the Close column — that's all risk.py needs
        prices[ticker] = raw[["Close"]].rename(columns={"Close": "close"})

    return prices
