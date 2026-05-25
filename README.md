# Portfolio Risk Agent

An AI-powered agent that reads your portfolio holdings, computes key risk metrics, fetches relevant news, and generates a morning risk briefing memo using Claude.

## Features

- Load portfolio from CSV or JSON
- Fetch historical prices via `yfinance`
- Compute concentration, beta, max drawdown, and daily PnL
- Fetch top-holding news via NewsAPI
- Generate a plain-English risk memo via the Anthropic API
- Scheduled daily runs via GitHub Actions

## Project Structure

```
portfolio-risk-agent/
├── agent/
│   ├── loader.py      # Read portfolio from CSV or JSON
│   ├── prices.py      # Fetch historical prices (yfinance)
│   ├── risk.py        # Compute risk metrics (pure functions)
│   ├── news.py        # Fetch news for top holdings
│   └── memo.py        # Generate memo via Anthropic API
├── data/
│   └── portfolio.csv  # Your portfolio input
├── outputs/           # Generated memos (gitignored)
├── .github/
│   └── workflows/
│       └── daily_brief.yml
├── config.py          # Constants and env var loading
├── main.py            # Pipeline entry point
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux

pip install -r requirements.txt
cp .env.example .env       # then fill in your API keys
```

## Configuration

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | From [console.anthropic.com](https://console.anthropic.com) |
| `NEWS_API_KEY` | From [newsapi.org](https://newsapi.org) |

Tuneable constants (no secrets) live in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `LOOKBACK_DAYS` | 90 | Trading days for risk calculations |
| `TOP_N_NEWS` | 5 | Holdings to fetch news for |
| `CONCENTRATION_THRESHOLD` | 0.20 | Single-holding weight warning level |
| `BENCHMARK_TICKER` | SPY | Beta benchmark |

## Portfolio Format

**CSV** (`data/portfolio.csv`):
```
ticker,shares,cost_basis
AAPL,50,150.00
MSFT,30,280.00
```

**JSON** (`data/portfolio.json`):
```json
[
  {"ticker": "AAPL", "shares": 50, "cost_basis": 150.00}
]
```

## Usage

```bash
python main.py
```

## Automation

The GitHub Actions workflow (`.github/workflows/daily_brief.yml`) runs `main.py` at 7:00 AM UTC on weekdays. Add `ANTHROPIC_API_KEY` and `NEWS_API_KEY` as repository secrets to enable it.

## Requirements

- Python 3.10+
- See `requirements.txt`
