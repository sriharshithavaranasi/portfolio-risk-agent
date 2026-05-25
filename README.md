# Portfolio Risk Agent

An AI-powered agent that analyzes portfolio risk metrics, surfaces exposures, and generates actionable insights for investment portfolios.

## Overview

This agent ingests portfolio holdings and market data, computes standard risk measures (VaR, volatility, drawdown, correlation), and uses an LLM to interpret results and flag concerns in plain language.

## Features

- Portfolio risk metric calculation (Value at Risk, Sharpe ratio, max drawdown, beta)
- Sector and factor exposure analysis
- Correlation and concentration risk detection
- Natural language risk summaries and alerts

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --portfolio portfolio.csv
```

## Requirements

- Python 3.10+
- See `requirements.txt`
