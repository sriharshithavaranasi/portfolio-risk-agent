"""
Tests for agent/memo.py and agent/tools.py

The Anthropic client and all tool handlers are mocked — no API calls,
no network traffic, no API key required.

Run from the project root:
    python -m pytest tests/test_memo.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.memo import (
    MAX_LOOP_ITERATIONS,
    RiskMemo,
    _build_initial_message,
    _run_agent_loop,
    format_memo,
)
from agent.tools import (
    ALL_TOOLS,
    ANALYTICS_TOOLS,
    MEMO_TOOL,
    PortfolioContext,
    build_handlers,
    execute_tool,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HOLDINGS = [
    {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
    {"ticker": "MSFT", "shares": 5,  "cost_basis": 280.0},
]

MEMO_INPUTS = {
    "overall_risk_level":  "MEDIUM",
    "daily_pnl_dollars":   -420.00,
    "daily_pnl_pct":       -0.0032,
    "portfolio_beta":       1.12,
    "max_drawdown_pct":    -0.14,
    "hhi_score":            0.52,
    "largest_position":    "AAPL",
    "concentration_flags": ["AAPL"],
    "risk_flags":          ["AAPL exceeds 20% concentration threshold"],
    "news_highlights":     [{"ticker": "AAPL", "digest": "Apple beats estimates"}],
    "analyst_commentary":  "The portfolio is moderately concentrated in tech.",
    "recommendations":     ["Consider trimming AAPL to below 20%"],
}


def make_tool_use_block(name: str, inputs: dict, block_id: str = "tu_001") -> MagicMock:
    """Return a mock content block that looks like a tool_use response."""
    block = MagicMock()
    block.type  = "tool_use"
    block.name  = name
    block.input = inputs
    block.id    = block_id
    return block


def make_response(blocks: list, stop_reason: str = "tool_use") -> MagicMock:
    """Return a mock Anthropic Messages response."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content     = blocks
    return resp


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_all_analytics_tools_have_required_fields(self):
        for tool in ANALYTICS_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_memo_tool_requires_all_memo_fields(self):
        required = set(MEMO_TOOL["input_schema"]["required"])
        assert "overall_risk_level" in required
        assert "analyst_commentary" in required
        assert "recommendations" in required

    def test_all_tools_list_includes_memo_tool(self):
        names = [t["name"] for t in ALL_TOOLS]
        assert "submit_risk_memo" in names

    def test_overall_risk_level_has_enum_constraint(self):
        props = MEMO_TOOL["input_schema"]["properties"]
        assert set(props["overall_risk_level"]["enum"]) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_dispatches_to_correct_handler(self):
        handlers = {"my_tool": lambda **_: json.dumps({"result": 42})}
        result = execute_tool("my_tool", {}, handlers)
        assert json.loads(result) == {"result": 42}

    def test_unknown_tool_returns_error_json(self):
        result = execute_tool("nonexistent", {}, {})
        assert "error" in json.loads(result)

    def test_handler_exception_returns_error_json(self):
        def bad_handler(**_):
            raise ValueError("Something went wrong")

        handlers = {"bad": bad_handler}
        result = execute_tool("bad", {}, handlers)
        assert "error" in json.loads(result)
        assert "Something went wrong" in json.loads(result)["error"]

    def test_handler_receives_inputs_as_kwargs(self):
        received = {}

        def capture_handler(**kwargs):
            received.update(kwargs)
            return json.dumps({})

        handlers = {"cap": capture_handler}
        execute_tool("cap", {"top_n": 5}, handlers)
        assert received == {"top_n": 5}


# ---------------------------------------------------------------------------
# _build_initial_message
# ---------------------------------------------------------------------------

class TestBuildInitialMessage:
    def test_includes_all_tickers(self):
        msg = _build_initial_message(HOLDINGS)
        assert "AAPL" in msg
        assert "MSFT" in msg

    def test_includes_shares_and_cost_basis(self):
        msg = _build_initial_message(HOLDINGS)
        assert "10" in msg      # AAPL shares
        assert "150.00" in msg  # AAPL cost basis

    def test_includes_today_date(self):
        from datetime import date
        msg = _build_initial_message(HOLDINGS)
        assert date.today().isoformat() in msg

    def test_mentions_submit_risk_memo(self):
        msg = _build_initial_message(HOLDINGS)
        assert "submit_risk_memo" in msg


# ---------------------------------------------------------------------------
# _run_agent_loop
# ---------------------------------------------------------------------------

class TestRunAgentLoop:
    def _make_client(self, side_effects: list) -> MagicMock:
        client = MagicMock()
        client.messages.create.side_effect = side_effects
        return client

    def test_single_turn_memo_submission(self):
        """Agent calls submit_risk_memo on the very first response."""
        memo_block = make_tool_use_block("submit_risk_memo", MEMO_INPUTS, "tu_memo")
        response   = make_response([memo_block])
        client     = self._make_client([response])
        handlers   = {}  # no analytics tools needed

        messages = [{"role": "user", "content": "Analyze this."}]
        memo = _run_agent_loop(client, messages, handlers)

        assert isinstance(memo, RiskMemo)
        assert memo.overall_risk_level == "MEDIUM"
        assert memo.portfolio_beta == 1.12

    def test_analytics_tool_called_before_memo(self):
        """Agent calls one analytics tool, then submits the memo."""
        analytics_block = make_tool_use_block("compute_beta", {}, "tu_beta")
        memo_block      = make_tool_use_block("submit_risk_memo", MEMO_INPUTS, "tu_memo")

        analytics_response = make_response([analytics_block])
        memo_response      = make_response([memo_block])

        handlers = {"compute_beta": lambda **_: json.dumps({"beta": 1.12})}
        client   = self._make_client([analytics_response, memo_response])

        messages = [{"role": "user", "content": "Analyze."}]
        memo = _run_agent_loop(client, messages, handlers)

        assert memo.portfolio_beta == 1.12
        assert client.messages.create.call_count == 2

    def test_tool_results_appended_to_messages(self):
        """Tool results must be sent back before the memo response."""
        analytics_block = make_tool_use_block("compute_beta", {}, "tu_beta")
        memo_block      = make_tool_use_block("submit_risk_memo", MEMO_INPUTS, "tu_memo")

        handlers = {"compute_beta": lambda **_: json.dumps({"beta": 1.0})}
        client   = self._make_client([
            make_response([analytics_block]),
            make_response([memo_block]),
        ])

        messages = [{"role": "user", "content": "Analyze."}]
        _run_agent_loop(client, messages, handlers)

        # messages should be: user | assistant | tool_result | assistant | tool_result
        roles = [m["role"] for m in messages]
        assert roles.count("assistant") == 2
        assert roles.count("user") == 3   # initial + 2 tool result batches

    def test_end_turn_without_memo_raises(self):
        """If Claude stops without calling submit_risk_memo, raise RuntimeError."""
        response = make_response([], stop_reason="end_turn")
        client   = self._make_client([response])

        with pytest.raises(RuntimeError, match="submit_risk_memo"):
            _run_agent_loop(client, [{"role": "user", "content": "Go."}], {})

    def test_loop_cap_raises_if_never_submits(self):
        """Never-ending analytics tool calls must hit the cap and raise."""
        always_tool = make_response(
            [make_tool_use_block("compute_beta", {}, "tu_x")]
        )
        # Return the same response every time
        handlers = {"compute_beta": lambda **_: json.dumps({"beta": 1.0})}
        client   = self._make_client([always_tool] * (MAX_LOOP_ITERATIONS + 1))

        with pytest.raises(RuntimeError, match=str(MAX_LOOP_ITERATIONS)):
            _run_agent_loop(
                client, [{"role": "user", "content": "Go."}], handlers
            )


# ---------------------------------------------------------------------------
# RiskMemo
# ---------------------------------------------------------------------------

class TestRiskMemo:
    def make(self) -> RiskMemo:
        return RiskMemo(**MEMO_INPUTS)

    def test_to_dict_has_expected_top_level_keys(self):
        d = self.make().to_dict()
        assert {"generated_on", "overall_risk_level", "daily_pnl",
                "portfolio_beta", "max_drawdown_pct", "concentration",
                "risk_flags", "news_highlights", "analyst_commentary",
                "recommendations"} == set(d.keys())

    def test_generated_on_is_set_automatically(self):
        from datetime import date
        memo = self.make()
        assert memo.generated_on == date.today().isoformat()


# ---------------------------------------------------------------------------
# format_memo
# ---------------------------------------------------------------------------

class TestFormatMemo:
    def make(self) -> RiskMemo:
        return RiskMemo(**MEMO_INPUTS)

    def test_includes_risk_level(self):
        text = format_memo(self.make())
        assert "MEDIUM" in text

    def test_includes_beta(self):
        text = format_memo(self.make())
        assert "1.12" in text

    def test_includes_recommendations(self):
        text = format_memo(self.make())
        assert "AAPL" in text

    def test_includes_news_highlights(self):
        text = format_memo(self.make())
        assert "Apple beats estimates" in text
