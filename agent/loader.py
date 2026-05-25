"""
Loads portfolio holdings from a CSV or JSON file.

Expected CSV columns: ticker, shares, cost_basis
Expected JSON format: list of {"ticker": ..., "shares": ..., "cost_basis": ...}

Returns a list of dicts, one per holding.
"""

import json
import pandas as pd
from pathlib import Path


def load_portfolio(path: str) -> list[dict]:
    """Load and validate portfolio holdings from a CSV or JSON file."""
    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(f"Portfolio file not found: {path}")

    if file.suffix == ".csv":
        return _load_csv(file)
    elif file.suffix == ".json":
        return _load_json(file)
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}. Use .csv or .json")


def _load_csv(file: Path) -> list[dict]:
    # TODO: read CSV with pandas
    # TODO: validate required columns: ticker, shares, cost_basis
    # TODO: strip whitespace from ticker symbols
    # TODO: return as list of dicts
    raise NotImplementedError


def _load_json(file: Path) -> list[dict]:
    # TODO: read JSON file
    # TODO: validate each entry has ticker, shares, cost_basis keys
    # TODO: return as list of dicts
    raise NotImplementedError


def _validate_holdings(holdings: list[dict]) -> None:
    """Raise ValueError if any required field is missing or malformed."""
    required = {"ticker", "shares", "cost_basis"}
    # TODO: check every holding has all required fields
    # TODO: check shares and cost_basis are positive numbers
    raise NotImplementedError
