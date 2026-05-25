"""
Entry point for the Portfolio Risk Briefing Agent.

Pipeline:
    load holdings → fetch prices → compute risk → fetch news → generate memo → save
"""

from agent.loader import load_portfolio
from agent.prices import fetch_prices
from agent.risk import compute_risk_metrics
from agent.news import fetch_news
from agent.memo import generate_memo


def run(portfolio_path: str = "data/portfolio.csv") -> None:
    # TODO: add CLI argument parsing (argparse) so portfolio path can be passed at runtime

    print(f"Loading portfolio from {portfolio_path}...")
    holdings = load_portfolio(portfolio_path)

    print("Fetching prices...")
    prices = fetch_prices(holdings)

    print("Computing risk metrics...")
    metrics = compute_risk_metrics(holdings, prices)

    print("Fetching news for top holdings...")
    news = fetch_news(holdings)

    print("Generating risk memo...")
    memo = generate_memo(metrics, news)

    # TODO: save memo to outputs/ with a timestamp in the filename
    print("\n--- MORNING RISK MEMO ---\n")
    print(memo)


if __name__ == "__main__":
    run()
