"""
Generates a morning risk briefing memo using the Anthropic API.

Takes structured risk metrics and news headlines, builds a prompt,
and returns a plain-text memo suitable for email or Slack.
"""

import anthropic
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def generate_memo(metrics: dict, news: list[dict]) -> str:
    """
    Generate a morning risk memo from computed metrics and news.

    Args:
        metrics: output of compute_risk_metrics() from risk.py
        news:    output of fetch_news() from news.py

    Returns:
        Plain-text risk memo as a string.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = _build_prompt(metrics, news)

    # TODO: call client.messages.create() with ANTHROPIC_MODEL
    # TODO: use a system prompt that sets the tone (concise, professional, risk-focused)
    # TODO: return the text content of the response
    raise NotImplementedError


def _build_prompt(metrics: dict, news: list[dict]) -> str:
    """
    Construct the user prompt from metrics and news.

    The prompt should include:
        - Portfolio concentration (weights, HHI, largest holding)
        - Beta vs benchmark
        - Max drawdown over the lookback window
        - Daily PnL ($ and %)
        - Top news headlines per holding
    """
    # TODO: format metrics dict into a readable text block
    # TODO: format news list into a bulleted section
    # TODO: return the combined prompt string
    raise NotImplementedError
