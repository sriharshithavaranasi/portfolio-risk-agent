"""
Entry point for the Portfolio Risk Briefing Agent.

Pipeline:
    load holdings → fetch prices → run agent loop → format + save memo
"""

import argparse
from datetime import date
from pathlib import Path

from agent.loader import load_portfolio
from agent.memo import format_memo, generate_memo
from agent.prices import fetch_prices


def run(portfolio_path: str = "data/portfolio.csv") -> None:
    print(f"Loading portfolio from {portfolio_path}...")
    holdings = load_portfolio(portfolio_path)

    print(f"Fetching prices for {len(holdings)} holdings...")
    prices = fetch_prices(holdings)

    print("Running risk agent...\n")
    memo = generate_memo(holdings, prices)

    text = format_memo(memo)
    print(text)

    _save_memo(text)


def _save_memo(text: str) -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"risk_brief_{date.today().isoformat()}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"\nMemo saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Risk Briefing Agent")
    parser.add_argument(
        "--portfolio",
        default="data/portfolio.csv",
        help="Path to portfolio CSV or JSON file (default: data/portfolio.csv)",
    )
    args = parser.parse_args()
    run(args.portfolio)
