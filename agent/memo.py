"""
Agent orchestration layer — drives the tool-use loop and generates the risk memo.

How the tool-use loop works
---------------------------
Unlike a single-shot prompt, a tool-use agent runs in a conversation loop:

  1. We send Claude the portfolio and a list of available tools.
  2. Claude responds with tool_use blocks — the tools it wants to call.
  3. We execute those tools (in agent/tools.py) and send the results back.
  4. Claude processes the results and either calls more tools or calls
     submit_risk_memo, which signals it is done.
  5. We extract the structured RiskMemo from submit_risk_memo's inputs and return it.

Why submit_risk_memo instead of parsing free text?
--------------------------------------------------
Free-text responses are hard to parse reliably. By making the final output
a tool call with a strict JSON Schema, we get guaranteed structure —
every field is always present and correctly typed, ready to render or store.

Public API
----------
    generate_memo(holdings, prices) -> RiskMemo
    format_memo(memo)               -> str   (human-readable text)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import anthropic

from agent.tools import ALL_TOOLS, PortfolioContext, build_handlers, execute_tool
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

MAX_LOOP_ITERATIONS = 10   # safety cap — prevents runaway tool-use loops


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

@dataclass
class RiskMemo:
    """
    Fully structured risk memo produced by the agent.

    Fields mirror the submit_risk_memo tool schema in tools.py so that
    Claude's tool call can be unpacked directly into this dataclass.
    """
    overall_risk_level:  str
    daily_pnl_dollars:   float
    daily_pnl_pct:       float
    portfolio_beta:      float
    max_drawdown_pct:    float
    hhi_score:           float
    largest_position:    str
    concentration_flags: list[str]
    risk_flags:          list[str]
    news_highlights:     list[dict]
    analyst_commentary:  str
    recommendations:     list[str]
    generated_on:        str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return {
            "generated_on":       self.generated_on,
            "overall_risk_level": self.overall_risk_level,
            "daily_pnl": {
                "dollars": self.daily_pnl_dollars,
                "pct":     self.daily_pnl_pct,
            },
            "portfolio_beta":     self.portfolio_beta,
            "max_drawdown_pct":   self.max_drawdown_pct,
            "concentration": {
                "hhi_score":        self.hhi_score,
                "largest_position": self.largest_position,
                "flags":            self.concentration_flags,
            },
            "risk_flags":         self.risk_flags,
            "news_highlights":    self.news_highlights,
            "analyst_commentary": self.analyst_commentary,
            "recommendations":    self.recommendations,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative risk analyst generating a morning portfolio risk briefing.

Your job:
1. Use the available tools to compute the risk metrics you need.
2. Fetch news for the largest positions.
3. When you have enough information, call submit_risk_memo with a complete analysis.

Guidelines:
- Be concise and precise. Risk managers read fast.
- Flag anything unusual — high beta, concentrated positions, large drawdowns, negative news.
- overall_risk_level must reflect the combined picture, not just one metric.
- analyst_commentary should read like a professional morning note: 2-3 tight paragraphs.
- recommendations should be specific and actionable.
- Call submit_risk_memo exactly once, as your final action."""


def _build_initial_message(holdings: list[dict]) -> str:
    """
    Format the portfolio holdings into the opening user message.

    Giving Claude the raw holdings upfront means it can reason about the
    portfolio (e.g. identify large positions) before calling any tools.
    """
    lines = [f"Date: {date.today().isoformat()}", "", "Portfolio holdings:", ""]
    for h in holdings:
        lines.append(
            f"  - {h['ticker']}: {h['shares']} shares, "
            f"cost basis ${float(h['cost_basis']):.2f}/share"
        )
    lines += [
        "",
        "Please analyze this portfolio and produce a morning risk briefing.",
        "Use the available tools to compute the metrics you need, then call submit_risk_memo.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def _run_agent_loop(
    client: anthropic.Anthropic,
    messages: list[dict],
    handlers: dict[str, callable],
) -> RiskMemo:
    """
    Drive the tool-use conversation until Claude calls submit_risk_memo.

    Each iteration:
      - Sends the current message history to Claude.
      - If Claude returns tool_use blocks, executes each tool.
      - Collects all tool results into a single user turn (Anthropic requires
        all results from one assistant turn to be batched in one user message).
      - Stops when submit_risk_memo is called or the iteration cap is reached.

    Args:
        client:   Authenticated Anthropic client.
        messages: Conversation history — modified in place each iteration.
        handlers: Tool name → handler function from build_handlers().

    Returns:
        RiskMemo built from the inputs Claude passed to submit_risk_memo.

    Raises:
        RuntimeError: If the loop cap is hit without a memo being submitted.
    """
    for iteration in range(MAX_LOOP_ITERATIONS):
        response = client.messages.create(
            model=      ANTHROPIC_MODEL,
            system=     SYSTEM_PROMPT,
            tools=      ALL_TOOLS,
            messages=   messages,
            max_tokens= 4096,
        )

        # Append Claude's response to the conversation history
        messages.append({"role": "assistant", "content": response.content})

        # Claude finished talking without using any tools — shouldn't happen
        # given the system prompt, but handle it gracefully
        if response.stop_reason == "end_turn":
            raise RuntimeError(
                "Agent ended without calling submit_risk_memo. "
                "Check the system prompt or increase max_tokens."
            )

        if response.stop_reason != "tool_use":
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")

        # --- Process all tool calls from this assistant turn ---
        # Claude can request multiple tools in one response (parallel tool use).
        # We must return ALL results in a single user message — the API rejects
        # partial batches.

        tool_results: list[dict] = []
        memo_inputs:  dict | None = None

        for block in response.content:
            if block.type != "tool_use":
                continue  # text blocks mixed in with tool_use blocks — skip

            if block.name == "submit_risk_memo":
                # The agent is done. Acknowledge the tool call and capture the memo.
                memo_inputs = block.input
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     "Memo received. Thank you.",
                })
            else:
                print(f"  [tool] {block.name}({block.input or ''})")
                result = execute_tool(block.name, block.input, handlers)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

        # Send tool results back before checking memo — the API requires the
        # tool_result turn even when we're about to exit the loop
        messages.append({"role": "user", "content": tool_results})

        if memo_inputs is not None:
            return RiskMemo(**memo_inputs)

    raise RuntimeError(
        f"Agent did not call submit_risk_memo within {MAX_LOOP_ITERATIONS} iterations."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_memo(holdings: list[dict], prices: dict) -> RiskMemo:
    """
    Run the full agent loop and return a structured RiskMemo.

    Args:
        holdings: List of holding dicts from loader.py.
        prices:   Price DataFrames from prices.py.

    Returns:
        RiskMemo with all fields populated by the agent.
    """
    client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    ctx     = PortfolioContext(holdings=holdings, prices=prices)
    handlers = build_handlers(ctx)

    messages = [{"role": "user", "content": _build_initial_message(holdings)}]

    return _run_agent_loop(client, messages, handlers)


def format_memo(memo: RiskMemo) -> str:
    """
    Render a RiskMemo as a readable plain-text morning brief.

    Suitable for printing to terminal, emailing, or posting to Slack.
    The RiskMemo dataclass holds structured data; this function is the
    presentation layer — separate so you can add other renderers (HTML,
    Markdown, PDF) without touching the agent logic.
    """
    pnl_sign  = "+" if memo.daily_pnl_dollars >= 0 else ""
    pnl_pct   = memo.daily_pnl_pct * 100
    dd_pct    = memo.max_drawdown_pct * 100

    flags_text  = "\n".join(f"  ! {f}" for f in memo.risk_flags) or "  None"
    recs_text   = "\n".join(f"  • {r}" for r in memo.recommendations) or "  None"
    news_text   = "\n".join(
        f"  [{s['ticker']}] {s['digest']}" for s in memo.news_highlights
    ) or "  No headlines available."

    return textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════╗
    ║         MORNING PORTFOLIO RISK BRIEF                 ║
    ║         {memo.generated_on}                          ║
    ╚══════════════════════════════════════════════════════╝

    RISK LEVEL:  {memo.overall_risk_level}
    ──────────────────────────────────────────────────────
    Daily PnL    {pnl_sign}${memo.daily_pnl_dollars:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)
    Beta         {memo.portfolio_beta:.2f}
    Max Drawdown {dd_pct:.1f}%
    HHI Score    {memo.hhi_score:.3f}  (largest: {memo.largest_position})

    RISK FLAGS
    {flags_text}

    NEWS HIGHLIGHTS
    {news_text}

    ANALYSIS
    {memo.analyst_commentary}

    RECOMMENDATIONS
    {recs_text}
    """).strip()
