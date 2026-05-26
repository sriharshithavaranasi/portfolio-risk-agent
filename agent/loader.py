"""
Loads and validates portfolio holdings from a CSV or JSON file.

Expected CSV columns : ticker, shares, cost_basis
Expected JSON format : list of {"ticker": str, "shares": number, "cost_basis": number}

Returns a list of normalised dicts, one per holding.
"""

import json
import pandas as pd
from pathlib import Path


def load_portfolio(path: str) -> list[dict]:
    """
    Load holdings from a CSV or JSON file and return a validated list of dicts.

    Args:
        path: File path. Extension must be .csv or .json.

    Returns:
        [{"ticker": "AAPL", "shares": 50.0, "cost_basis": 150.0}, ...]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required columns/keys are missing or values are invalid.
    """
    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(f"Portfolio file not found: {path}")

    if file.suffix == ".csv":
        holdings = _load_csv(file)
    elif file.suffix == ".json":
        holdings = _load_json(file)
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}. Use .csv or .json")

    _validate_holdings(holdings)
    return holdings


def _load_csv(file: Path) -> list[dict]:
    required = {"ticker", "shares", "cost_basis"}

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["ticker"]     = df["ticker"].astype(str).str.strip().str.upper()
    df["shares"]     = pd.to_numeric(df["shares"],     errors="raise")
    df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="raise")

    return df[["ticker", "shares", "cost_basis"]].to_dict(orient="records")


def _load_json(file: Path) -> list[dict]:
    with file.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON portfolio must be a list of holding objects.")

    holdings = []
    for i, item in enumerate(data):
        entry = {k.strip().lower(): v for k, v in item.items()}
        holdings.append({
            "ticker":     str(entry.get("ticker", "")).strip().upper(),
            "shares":     float(entry.get("shares", 0)),
            "cost_basis": float(entry.get("cost_basis", 0)),
        })

    return holdings


def _validate_holdings(holdings: list[dict]) -> None:
    """Raise ValueError if any holding has missing or invalid fields."""
    if not holdings:
        raise ValueError("Portfolio is empty — no holdings found.")

    required = {"ticker", "shares", "cost_basis"}

    for i, h in enumerate(holdings):
        missing = required - set(h.keys())
        if missing:
            raise ValueError(f"Holding {i} is missing fields: {missing}")
        if not h["ticker"]:
            raise ValueError(f"Holding {i} has an empty ticker symbol.")
        if h["shares"] <= 0:
            raise ValueError(f"{h['ticker']}: shares must be positive, got {h['shares']}")
        if h["cost_basis"] <= 0:
            raise ValueError(f"{h['ticker']}: cost_basis must be positive, got {h['cost_basis']}")
