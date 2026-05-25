"""
Tool definitions and handlers for the Portfolio Risk Agent.

Architecture
------------
This module is the bridge between the Anthropic tool-use API and our analytics code.

Two layers:
  1. TOOL SCHEMAS  — JSON Schema objects that describe each tool to Claude.
                     Claude reads these to know what tools exist and what arguments they take.
  2. TOOL HANDLERS — Python functions that actually run when Claude calls a tool.
                     They are bound to a PortfolioContext (pre-loaded holdings + prices)
                     so they need no arguments beyond what Claude supplies.

The tool-use loop in memo.py calls execute_tool() to dispatch by name.
Adding a new tool means: add a schema to ANALYTICS_TOOLS, add a handler to build_handlers().
"""

import json
from dataclasses import dataclass

import pandas as pd

from agent.news import fetch_news, summarize_news
from agent.risk import (
    compute_beta,
    compute_concentration,
    compute_daily_pnl,
    compute_max_drawdown,
    compute_portfolio_weights,
)


# ---------------------------------------------------------------------------
# Context object — holds pre-loaded data so handlers don't do I/O
# ---------------------------------------------------------------------------

@dataclass
class PortfolioContext:
    """
    Pre-loaded portfolio data passed to every tool handler.

    Loaded once in main.py and shared across all tool calls in a single run.
    Keeping I/O out of tool handlers makes them fast and easy to test.
    """
    holdings: list[dict]
    prices:   dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Tool schemas (what Claude sees)
# ---------------------------------------------------------------------------
# Each schema follows the Anthropic tool-use format:
#   name          — must match the key in build_handlers()
#   description   — Claude reads this to decide when to call the tool
#   input_schema  — JSON Schema for the arguments Claude must supply

ANALYTICS_TOOLS: list[dict] = [
    {
        "name": "compute_concentration",
        "description": (
            "Calculate portfolio concentration: the weight of each holding, "
            "the HHI score (0=diversified, 1=fully concentrated), the largest "
            "position, and any holdings that exceed the 20% concentration threshold."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_beta",
        "description": (
            "Calculate portfolio beta relative to SPY. "
            "Beta > 1 means the portfolio amplifies market moves; "
            "Beta < 1 means it dampens them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_max_drawdown",
        "description": (
            "Calculate the maximum peak-to-trough decline of the portfolio "
            "over the historical price window. Returns a negative percentage, "
            "e.g. -0.18 means the portfolio fell 18% from its peak."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_daily_pnl",
        "description": (
            "Calculate yesterday's portfolio profit or loss in dollars and as "
            "a percentage of the prior day's portfolio value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "fetch_top_holding_news",
        "description": (
            "Fetch recent news headlines for the largest portfolio positions. "
            "Use this to surface market-moving events for concentrated holdings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type":        "integer",
                    "description": "How many top holdings (by weight) to fetch news for.",
                    "default":     3,
                    "minimum":     1,
                    "maximum":     10,
                },
            },
            "required": [],
        },
    },
]

# The memo tool is kept separate because it ends the loop — it's not an analytics call,
# it's the structured output contract. Claude must call it exactly once, at the end.
MEMO_TOOL: dict = {
    "name": "submit_risk_memo",
    "description": (
        "Call this when you have gathered all the analytics you need. "
        "Submit the completed morning risk memo as structured data. "
        "This is the final step — do not call any other tools after this."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "overall_risk_level": {
                "type":        "string",
                "enum":        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                "description": "Single overall risk rating for the portfolio today.",
            },
            "daily_pnl_dollars": {
                "type":        "number",
                "description": "Yesterday's PnL in dollars. Negative = loss.",
            },
            "daily_pnl_pct": {
                "type":        "number",
                "description": "Yesterday's PnL as a decimal fraction, e.g. -0.012 = -1.2%.",
            },
            "portfolio_beta": {
                "type":        "number",
                "description": "Portfolio beta vs the benchmark.",
            },
            "max_drawdown_pct": {
                "type":        "number",
                "description": "Max drawdown as a decimal fraction, e.g. -0.15 = -15%.",
            },
            "hhi_score": {
                "type":        "number",
                "description": "Herfindahl-Hirschman Index concentration score (0 to 1).",
            },
            "largest_position": {
                "type":        "string",
                "description": "Ticker of the largest holding by market value.",
            },
            "concentration_flags": {
                "type":  "array",
                "items": {"type": "string"},
                "description": "List of tickers that exceed the concentration threshold.",
            },
            "risk_flags": {
                "type":  "array",
                "items": {"type": "string"},
                "description": "Concise risk observations, one per item.",
            },
            "news_highlights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "digest": {"type": "string"},
                    },
                },
                "description": "Key news headlines grouped by ticker.",
            },
            "analyst_commentary": {
                "type":        "string",
                "description": "2-3 paragraph narrative risk analysis.",
            },
            "recommendations": {
                "type":  "array",
                "items": {"type": "string"},
                "description": "Actionable recommendations, one per item.",
            },
        },
        "required": [
            "overall_risk_level",
            "daily_pnl_dollars",
            "daily_pnl_pct",
            "portfolio_beta",
            "max_drawdown_pct",
            "hhi_score",
            "largest_position",
            "concentration_flags",
            "risk_flags",
            "news_highlights",
            "analyst_commentary",
            "recommendations",
        ],
    },
}

ALL_TOOLS: list[dict] = ANALYTICS_TOOLS + [MEMO_TOOL]


# ---------------------------------------------------------------------------
# Tool handlers (what runs when Claude calls a tool)
# ---------------------------------------------------------------------------

def build_handlers(ctx: PortfolioContext) -> dict[str, callable]:
    """
    Return a dict mapping tool name → handler function, all bound to ctx.

    Each handler takes only the arguments Claude supplies (from the schema),
    runs the corresponding analytics function, and returns a JSON string.
    Returning JSON keeps the tool result format consistent and easy for
    Claude to parse and reason about.
    """

    def _compute_concentration(**_) -> str:
        weights = compute_portfolio_weights(ctx.holdings, ctx.prices)
        result  = compute_concentration(ctx.holdings, ctx.prices, weights)
        return json.dumps(result)

    def _compute_beta(**_) -> str:
        beta = compute_beta(ctx.holdings, ctx.prices)
        return json.dumps({"beta": beta})

    def _compute_max_drawdown(**_) -> str:
        dd = compute_max_drawdown(ctx.holdings, ctx.prices)
        return json.dumps({"max_drawdown": dd})

    def _compute_daily_pnl(**_) -> str:
        pnl = compute_daily_pnl(ctx.holdings, ctx.prices)
        return json.dumps(pnl)

    def _fetch_top_holding_news(top_n: int = 3, **_) -> str:
        weights  = compute_portfolio_weights(ctx.holdings, ctx.prices)
        articles = fetch_news(ctx.holdings, weights=weights, articles_per_ticker=3)
        summaries = summarize_news(articles)[:top_n]
        return json.dumps([s.to_dict() for s in summaries])

    return {
        "compute_concentration":    _compute_concentration,
        "compute_beta":             _compute_beta,
        "compute_max_drawdown":     _compute_max_drawdown,
        "compute_daily_pnl":        _compute_daily_pnl,
        "fetch_top_holding_news":   _fetch_top_holding_news,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict, handlers: dict[str, callable]) -> str:
    """
    Dispatch a tool call by name and return the result as a JSON string.

    Errors are caught and returned as JSON too — Claude can see the error
    and decide whether to retry or skip, rather than crashing the whole run.
    """
    handler = handlers.get(name)

    if handler is None:
        return json.dumps({"error": f"Unknown tool: '{name}'"})

    try:
        return handler(**inputs)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
