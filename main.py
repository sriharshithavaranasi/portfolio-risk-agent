"""
Entry point for the Portfolio Risk Briefing Agent.

Pipeline:
    load holdings → fetch prices → run agent loop → build report → save MD + HTML
"""

import argparse
from datetime import date
from pathlib import Path

from agent.loader import load_portfolio
from agent.memo import format_memo, generate_memo
from agent.prices import fetch_prices
from agent.report import build_report, render_html, render_markdown


def run(portfolio_path: str = "data/portfolio.csv") -> None:
    print(f"Loading portfolio from {portfolio_path}...")
    holdings = load_portfolio(portfolio_path)

    print(f"Fetching prices for {len(holdings)} holdings...")
    prices = fetch_prices(holdings)

    print("Running risk agent...\n")
    memo = generate_memo(holdings, prices)

    print("Building report...\n")
    report = build_report(memo, holdings, prices)

    md   = render_markdown(report)
    html = render_html(report)

    print(format_memo(memo))
    _save(md, html)


def _save(md: str, html: str) -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    today = date.today().isoformat()

    md_path   = out / f"risk_brief_{today}.md"
    html_path = out / f"risk_brief_{today}.html"

    md_path.write_text(md,   encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    print(f"Saved: {md_path}")
    print(f"Saved: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Risk Briefing Agent")
    parser.add_argument(
        "--portfolio",
        default="data/portfolio.csv",
        help="Path to portfolio CSV or JSON file (default: data/portfolio.csv)",
    )
    args = parser.parse_args()
    run(args.portfolio)
