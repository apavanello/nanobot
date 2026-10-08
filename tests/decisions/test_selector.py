"""Tests for the tool-selection decider and scope detection."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.decisions.selector import (
    ToolSelectionDecider,
    build_choice_question,
    build_state,
    detect_decision_scope,
    has_zero_required_params,
    tool_catalog,
)
from nanobot.decisions.types import DecisionAnswer, DecisionResult
from nanobot.llm_usage.context import llm_usage_source


def _ctx(session_key: str | None) -> RequestContext:
    return RequestContext(channel="websocket", chat_id="c1", session_key=session_key)


def _tool(name: str, description: str = "does things", required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": required or []},
        },
    }


def _tools(count: int) -> list[dict[str, Any]]:
    return [_tool(f"tool_{i}", f"tool number {i}") for i in range(count)]


class FakeDecisionProvider:
    def __init__(self, answer: DecisionAnswer | None = None, error: str | None = None) -> None:
        self.answer = answer or DecisionAnswer(type="choice", choice="none")
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def decide(self, state: Any, questions: dict[str, Any]) -> DecisionResult:
        self.calls.append({"state": state, "questions": questions})
        return DecisionResult(
            provider="fake",
            model="fake-1",
            answers={"tool_selection": self.answer} if self.error is None else {},
            error=self.error,
            latency_ms=12,
        )


class TestScopeDetection:
    def test_subagent_prefix_wins(self) -> None:
        with request_context(_ctx("subagent:abc")):
            assert detect_decision_scope() == "subagent"

    def test_cron_source_maps_to_cron(self) -> None:
        with request_context(_ctx("chat-key")), llm_usage_source("cron"):
            assert detect_decision_scope() == "cron"

    def test_dream_and_system_map_to_cron(self) -> None:
        with request_context(_ctx("dream:x")):
            assert detect_decision_scope() == "cron"
        with llm_usage_source("system"):
            assert detect_decision_scope() == "cron"

    def test_plain_chat(self) -> None:
        with request_context(_ctx("chat-key")), llm_usage_source("user"):
            assert detect_decision_scope() == "chat"

    def test_no_context_defaults(self) -> None:
        with llm_usage_source("api"):
            assert detect_decision_scope() == "chat"


class TestStateAndQuestion:
    def test_state_contains_last_user_and_catalog(self) -> None:
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "list my files"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "result"},
        ]
        state = build_state(messages, [{"name": "list_files", "description": "lists files"}])
        assert state["last_user_message"] == "list my files"
        assert {"role": "system"} not in [m for m in state["recent_messages"]]
        assert state["available_tools"] == [{"name": "list_files", "description": "lists files"}]

    def test_state_lists_multimodal_content(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}, "raw"]}]
        state = build_state(messages, [])
        assert "hello" in (state["last_user_message"] or "")

    def test_catalog_skips_degenerate_defs(self) -> None:
        catalog = tool_catalog([_tool("ok"), {"type": "function", "function": {}}])
        assert [e["name"] for e in catalog] == ["ok"]

    def test_choice_question_includes_none(self) -> None:
        question = build_choice_question([{"name": "a", "description": "A tool"}])
        assert question["type"] == "choice"
        assert "none" in question["criteria"]
        assert question["criteria"]["a"] == "A tool"

    def test_zero_required_params(self) -> None:
        assert has_zero_required_params(_tool("t", required=[]))
        no_required_key = _tool("t")
        del no_required_key["function"]["parameters"]["required"]
        assert has_zero_required_params(no_required_key)
        assert not has_zero_required_params(_tool("t", required=["path"]))
        assert not has_zero_required_params(None)


class TestDeciderGates:
    async def test_scope_gate_skips_without_call(self) -> None:
        provider = FakeDecisionProvider()
        decider = ToolSelectionDecider(
            provider, mode="shadow", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "hi"}], _tools(3), "cron")
        assert selection.skipped_reason == "scope:cron"
        assert provider.calls == []

    async def test_min_tools_gate(self) -> None:
        provider = FakeDecisionProvider()
        decider = ToolSelectionDecider(
            provider, mode="shadow", scopes=["chat"], threshold=0.75, top_k=5, min_tools=8
        )
        selection = await decider.select([{"role": "user", "content": "hi"}], _tools(3), "chat")
        assert selection.skipped_reason == "min_tools"

    async def test_no_tools_gate(self) -> None:
        decider = ToolSelectionDecider(
            FakeDecisionProvider(), mode="shadow", scopes=["chat"], threshold=0.75, top_k=5, min_tools=1
        )
        selection = await decider.select([{"role": "user", "content": "hi"}], [], "chat")
        assert selection.skipped_reason == "no_tools"

    async def test_error_result_defers(self) -> None:
        decider = ToolSelectionDecider(
            FakeDecisionProvider(error="request failed"),
            mode="fastpath",
            scopes=["chat"],
            threshold=0.75,
            top_k=5,
            min_tools=2,
        )
        selection = await decider.select([{"role": "user", "content": "hi"}], _tools(3), "chat")
        assert selection.skipped_reason == "error:request failed"
        assert selection.fastpath_tool is None

    async def test_unknown_tool_defers(self) -> None:
        provider = FakeDecisionProvider(DecisionAnswer(type="choice", choice="ghost"))
        decider = ToolSelectionDecider(
            provider, mode="filter", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "hi"}], _tools(3), "chat")
        assert selection.skipped_reason == "unknown_tool:ghost"


class TestDeciderPolicy:
    async def test_shadow_mode_never_filters_or_fastpaths(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(
                type="choice",
                choice="tool_1",
                probabilities={"tool_1": 0.9, "tool_0": 0.1},
                confidence=0.95,
            )
        )
        decider = ToolSelectionDecider(
            provider, mode="shadow", scopes=["chat"], threshold=0.75, top_k=2, min_tools=2
        )
        tools = _tools(5)
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert not selection.skipped
        assert selection.chosen == "tool_1"
        assert selection.filtered_tools is None
        # Eligibility still recorded for calibration even in shadow mode.
        tools[1]["function"]["parameters"]["required"] = []
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_eligible is True
        assert selection.fastpath_tool is None

    async def test_filter_mode_ranks_and_truncates(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(
                type="choice",
                choice="tool_3",
                probabilities={"tool_3": 0.6, "tool_0": 0.3, "tool_2": 0.1},
                confidence=0.4,
            )
        )
        decider = ToolSelectionDecider(
            provider, mode="filter", scopes=["chat"], threshold=0.75, top_k=3, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(5), "chat")
        assert selection.filtered_tools is not None
        names = [t["function"]["name"] for t in selection.filtered_tools]
        assert names[0] == "tool_3"
        assert names[1] == "tool_0"
        assert names[2] == "tool_2"
        assert len(names) == 3

    async def test_fastpath_requires_confidence_threshold_and_zero_args(self) -> None:
        tools = _tools(3)
        tools[2]["function"]["parameters"]["required"] = []
        high = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_2", confidence=0.9, probabilities={"tool_2": 0.9})
        )
        decider = ToolSelectionDecider(
            high, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool == "tool_2"

        low = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_2", confidence=0.5, probabilities={"tool_2": 0.5})
        )
        decider = ToolSelectionDecider(
            low, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None
        assert selection.fastpath_eligible is False

        no_confidence = FakeDecisionProvider(DecisionAnswer(type="choice", choice="tool_2"))
        decider = ToolSelectionDecider(
            no_confidence, mode="fastpath", scopes=["chat"], threshold=0.0, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None

    async def test_required_args_never_fastpath(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_1", confidence=0.99)
        )
        tools = [_tool(f"tool_{i}", required=["path"]) for i in range(3)]
        decider = ToolSelectionDecider(
            provider, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None
        assert selection.filtered_tools is not None

    async def test_none_choice_disables_filter(self) -> None:
        provider = FakeDecisionProvider(DecisionAnswer(type="choice", choice="none", confidence=0.99))
        decider = ToolSelectionDecider(
            provider, mode="filter", scopes=["chat"], threshold=0.75, top_k=3, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(3), "chat")
        assert selection.chosen == "none"
        assert selection.filtered_tools is None
