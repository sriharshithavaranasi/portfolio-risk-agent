"""
Risk analytics module — computes portfolio risk metrics from holdings and price data.

All functions are pure (no network calls, no file I/O) so they are easy to unit test.
Inputs come from loader.py (holdings) and prices.py (price DataFrames).

Metrics:
    compute_portfolio_weights  — market value and weight of each holding
    compute_concentration      — weight distribution + HHI score
    compute_beta               — sensitivity to the market benchmark
    compute_max_drawdown       — worst peak-to-trough loss over the lookback window
    compute_daily_pnl          — yesterday's dollar and percentage gain/loss
"""

import numpy as np
import pandas as pd

from config import BENCHMARK_TICKER, CONCENTRATION_THRESHOLD


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def compute_risk_metrics(holdings: list[dict], prices: dict[str, pd.DataFrame]) -> dict:
    """
    Run all risk calculations and return a single combined metrics dict.

    Args:
        holdings: List of holding dicts from loader.py.
                  Each dict must have: ticker (str), shares (float), cost_basis (float).
        prices:   Dict from prices.py — {ticker: DataFrame with a "close" column}.

    Returns:
        {
            "weights":      {ticker: weight, ...},
            "concentration": {...},
            "beta":         float,
            "max_drawdown": float,
            "daily_pnl":    {...},
        }
    """
    weights = compute_portfolio_weights(holdings, prices)
    return {
        "weights":       weights,
        "concentration": compute_concentration(holdings, prices, weights),
        "beta":          compute_beta(holdings, prices, weights),
        "max_drawdown":  compute_max_drawdown(holdings, prices),
        "daily_pnl":     compute_daily_pnl(holdings, prices),
    }


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------

def compute_portfolio_weights(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """
    Calculate what fraction of the total portfolio each holding represents.

    The Math
    --------
    Market value of one holding:
        market_value = shares × latest_close_price

    Weight of one holding:
        weight = market_value / total_portfolio_value

    Weights always sum to 1.0 (100%).

    Args:
        holdings: List of holding dicts (ticker, shares, cost_basis).
        prices:   Price DataFrames keyed by ticker.

    Returns:
        Dict of {ticker: weight}, e.g. {"AAPL": 0.42, "MSFT": 0.31, ...}

    Raises:
        ValueError: If a holding's ticker is missing from the prices dict.
    """
    market_values: dict[str, float] = {}

    for holding in holdings:
        ticker = holding["ticker"]

        if ticker not in prices:
            raise ValueError(
                f"No price data for '{ticker}'. "
                "Make sure fetch_prices() was called with the same holdings."
            )

        # Take the most recent closing price (last row of the DataFrame)
        latest_close = float(prices[ticker]["close"].iloc[-1])
        market_values[ticker] = holding["shares"] * latest_close

    total_value = sum(market_values.values())

    if total_value == 0:
        raise ValueError("Total portfolio value is zero — check shares and price data.")

    # Divide each holding's value by the total to get its weight
    weights = {ticker: mv / total_value for ticker, mv in market_values.items()}

    return weights


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

def compute_concentration(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Measure how concentrated (vs. diversified) the portfolio is.

    Two outputs:
      1. Per-holding weights — how much of the portfolio each stock represents.
      2. HHI score — a single number summarising overall concentration.

    The Math: Herfindahl-Hirschman Index (HHI)
    -------------------------------------------
    HHI = sum of each holding's weight squared:
        HHI = w₁² + w₂² + w₃² + ...

    Interpretation:
        HHI = 1/n   → perfectly equal weights (most diversified)
        HHI = 1.0   → 100% in one stock (fully concentrated)

    Example — 4 equal holdings (25% each):
        HHI = 0.25² + 0.25² + 0.25² + 0.25² = 0.25

    Example — one stock is 80%, three others split the rest:
        HHI = 0.80² + ... = 0.64 + ... > 0.64  ← high concentration

    Args:
        holdings: List of holding dicts.
        prices:   Price DataFrames keyed by ticker.
        weights:  Pre-computed weights dict. If None, computed internally.

    Returns:
        {
            "weights":            {"AAPL": 0.42, "MSFT": 0.31, ...},
            "hhi":                0.27,
            "largest_holding":    "AAPL",
            "overweight_flags":   ["AAPL"],   # holdings above CONCENTRATION_THRESHOLD
        }
    """
    if weights is None:
        weights = compute_portfolio_weights(holdings, prices)

    # HHI = sum of squared weights
    hhi = sum(w ** 2 for w in weights.values())

    largest = max(weights, key=lambda t: weights[t])

    # Flag any single holding that exceeds the threshold (default 20%)
    overweight = [t for t, w in weights.items() if w > CONCENTRATION_THRESHOLD]

    return {
        "weights":          weights,
        "hhi":              round(hhi, 4),
        "largest_holding":  largest,
        "overweight_flags": overweight,
    }


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

def compute_beta(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Measure how much the portfolio moves relative to the market benchmark (SPY).

    The Math
    --------
    Step 1 — Daily return for each stock:
        return_t = (price_t - price_{t-1}) / price_{t-1}

    Step 2 — Portfolio daily return (weighted average of individual returns):
        portfolio_return_t = Σ (weight_i × return_i_t)

    Step 3 — Beta formula (linear regression slope):
        Beta = Covariance(portfolio_returns, benchmark_returns)
               ─────────────────────────────────────────────
               Variance(benchmark_returns)

    Interpretation:
        Beta = 1.0  → moves exactly with the market
        Beta > 1.0  → amplifies market moves (e.g. 1.5 = 50% more volatile)
        Beta < 1.0  → dampens market moves (e.g. 0.5 = half as volatile)
        Beta < 0    → moves opposite to the market (rare)

    Args:
        holdings: List of holding dicts.
        prices:   Price DataFrames keyed by ticker (must include BENCHMARK_TICKER).
        weights:  Pre-computed weights dict. If None, computed internally.

    Returns:
        Portfolio beta as a float, rounded to 4 decimal places.

    Raises:
        ValueError: If the benchmark ticker is missing from prices.
    """
    if weights is None:
        weights = compute_portfolio_weights(holdings, prices)

    if BENCHMARK_TICKER not in prices:
        raise ValueError(
            f"Benchmark '{BENCHMARK_TICKER}' not found in prices. "
            "fetch_prices() should always include the benchmark."
        )

    # Build a DataFrame of daily close prices, one column per holding
    close_df = pd.DataFrame({
        ticker: prices[ticker]["close"]
        for ticker in weights
    })

    # pct_change() computes (price_t - price_{t-1}) / price_{t-1} for each row
    returns_df = close_df.pct_change().dropna()

    # Portfolio return on each day = weighted sum of individual stock returns
    weight_series = pd.Series(weights)
    portfolio_returns = returns_df[list(weights.keys())].dot(weight_series)

    # Benchmark daily returns
    benchmark_returns = prices[BENCHMARK_TICKER]["close"].pct_change().dropna()

    # Align both series to the same dates (inner join on index)
    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner")
    combined.columns = ["portfolio", "benchmark"]
    combined = combined.dropna()

    # np.cov returns a 2×2 covariance matrix:
    #   [[Var(portfolio),              Cov(portfolio, benchmark)],
    #    [Cov(benchmark, portfolio),   Var(benchmark)           ]]
    cov_matrix = np.cov(combined["portfolio"], combined["benchmark"])
    covariance = cov_matrix[0, 1]   # Cov(portfolio, benchmark)
    variance   = cov_matrix[1, 1]   # Var(benchmark)

    if variance == 0:
        raise ValueError("Benchmark variance is zero — cannot compute beta.")

    beta = covariance / variance
    return round(float(beta), 4)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def compute_max_drawdown(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
) -> float:
    """
    Find the worst peak-to-trough decline the portfolio suffered over the lookback window.

    The Math
    --------
    Step 1 — Portfolio value on each day:
        value_t = Σ (shares_i × close_i_t)

    Step 2 — Running peak (the highest value seen so far up to day t):
        peak_t = max(value_0, value_1, ..., value_t)

    Step 3 — Drawdown on each day:
        drawdown_t = (value_t - peak_t) / peak_t

        This is always ≤ 0 (you can only be at or below a prior peak).

    Step 4 — Max drawdown = the most negative value in the drawdown series:
        max_drawdown = min(drawdown_t)

    Example:
        Portfolio peaks at $100,000, then falls to $82,000.
        Drawdown = (82,000 - 100,000) / 100,000 = -0.18  → "18% drawdown"

    Args:
        holdings: List of holding dicts (ticker, shares, cost_basis).
        prices:   Price DataFrames keyed by ticker.

    Returns:
        A negative float, e.g. -0.18 means the portfolio fell 18% from its peak.
    """
    # Build a daily portfolio value series: sum of (shares × close) across all holdings
    value_series = pd.Series(0.0, index=next(iter(prices.values())).index)

    for holding in holdings:
        ticker = holding["ticker"]
        if ticker not in prices:
            continue
        # Add this holding's daily dollar value to the total
        value_series = value_series.add(
            prices[ticker]["close"] * holding["shares"],
            fill_value=0,
        )

    value_series = value_series.dropna()

    # cummax() tracks the highest value seen up to each date
    running_peak = value_series.cummax()

    # Drawdown on each day: how far below the peak are we?
    drawdown_series = (value_series - running_peak) / running_peak

    # The worst (most negative) point is the max drawdown
    max_dd = float(drawdown_series.min())
    return round(max_dd, 4)


# ---------------------------------------------------------------------------
# Daily PnL
# ---------------------------------------------------------------------------

def compute_daily_pnl(
    holdings: list[dict],
    prices: dict[str, pd.DataFrame],
) -> dict:
    """
    Compute yesterday's portfolio profit or loss.

    The Math
    --------
    For each holding:
        pnl_i = (close_yesterday - close_day_before) × shares_i

    Total PnL:
        total_pnl = Σ pnl_i

    Percentage return:
        pnl_pct = total_pnl / portfolio_value_day_before

    Args:
        holdings: List of holding dicts (ticker, shares, cost_basis).
        prices:   Price DataFrames keyed by ticker.

    Returns:
        {
            "pnl_dollars": -1250.00,   # negative = loss, positive = gain
            "pnl_pct":     -0.008,     # e.g. -0.008 = -0.8%
        }

    Raises:
        ValueError: If there are fewer than 2 trading days of data.
    """
    first_ticker = next(t for t in holdings if t["ticker"] in prices)["ticker"]
    if len(prices[first_ticker]) < 2:
        raise ValueError("Need at least 2 days of price data to compute daily PnL.")

    total_pnl = 0.0
    prev_portfolio_value = 0.0

    for holding in holdings:
        ticker = holding["ticker"]
        if ticker not in prices:
            continue

        closes = prices[ticker]["close"]
        close_today     = float(closes.iloc[-1])
        close_yesterday = float(closes.iloc[-2])

        total_pnl          += (close_today - close_yesterday) * holding["shares"]
        prev_portfolio_value += close_yesterday * holding["shares"]

    if prev_portfolio_value == 0:
        raise ValueError("Previous portfolio value is zero — cannot compute PnL percentage.")

    pnl_pct = total_pnl / prev_portfolio_value

    return {
        "pnl_dollars": round(total_pnl, 2),
        "pnl_pct":     round(pnl_pct, 6),
    }
