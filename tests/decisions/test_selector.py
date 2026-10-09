"""Tests for the tool-selection decider and scope detection."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.decisions.selector import (
    ToolSelectionDecider,
    arg_question_id,
    build_choice_question,
    build_questions,
    build_state,
    detect_decision_scope,
    plan_tool,
    resolve_args,
    stated_question_id,
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


def _param_tool(
    name: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


def _tools(count: int) -> list[dict[str, Any]]:
    return [_tool(f"tool_{i}", f"tool number {i}") for i in range(count)]


class FakeDecisionProvider:
    def __init__(
        self,
        answer: DecisionAnswer | None = None,
        error: str | None = None,
        needs: float | None = 0.9,
        extra_answers: dict[str, DecisionAnswer] | None = None,
    ) -> None:
        self.answer = answer or DecisionAnswer(type="choice", choice="none")
        self.needs = needs
        self.error = error
        self.extra_answers = extra_answers or {}
        self.calls: list[dict[str, Any]] = []

    async def decide(self, state: Any, questions: dict[str, Any]) -> DecisionResult:
        self.calls.append({"state": state, "questions": questions})
        if self.error is not None:
            return DecisionResult(provider="fake", model="fake-1", error=self.error, latency_ms=12)
        answers: dict[str, DecisionAnswer] = {"tool_selection": self.answer}
        if self.needs is not None:
            answers["needs_tool"] = DecisionAnswer(type="noul", noul=self.needs)
        answers.update(self.extra_answers)
        return DecisionResult(provider="fake", model="fake-1", answers=answers, latency_ms=12)


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


class TestPlanTool:
    def test_no_properties_is_empty_closed(self) -> None:
        plan = plan_tool(_tool("t"))
        assert plan.name == "t"
        assert plan.closed_params == []

    def test_missing_parameters_key_is_empty_closed(self) -> None:
        tool = {"type": "function", "function": {"name": "t", "description": "d"}}
        assert plan_tool(tool).closed_params == []

    def test_const_param(self) -> None:
        plan = plan_tool(_param_tool("t", {"mode": {"const": "fast"}}, ["mode"]))
        assert plan.closed_params is not None and len(plan.closed_params) == 1
        param = plan.closed_params[0]
        assert param.kind == "const" and param.value == "fast" and param.required

    def test_single_value_enum_becomes_const(self) -> None:
        plan = plan_tool(_param_tool("t", {"format": {"enum": ["json"]}}, ["format"]))
        assert plan.closed_params is not None
        assert plan.closed_params[0].kind == "const"
        assert plan.closed_params[0].value == "json"

    def test_enum_param_keeps_wire_values(self) -> None:
        plan = plan_tool(
            _param_tool("t", {"level": {"enum": [1, 2, 3]}}, ["level"])
        )
        assert plan.closed_params is not None
        param = plan.closed_params[0]
        assert param.kind == "enum"
        assert param.values == {"1": 1, "2": 2, "3": 3}

    def test_enum_with_object_member_is_open(self) -> None:
        plan = plan_tool(_param_tool("t", {"cfg": {"enum": [{"a": 1}, "b"]}}, ["cfg"]))
        assert plan.closed_params is None

    def test_enum_with_colliding_labels_is_open(self) -> None:
        plan = plan_tool(_param_tool("t", {"v": {"enum": ["1", 1]}}, ["v"]))
        assert plan.closed_params is None

    def test_boolean_param(self) -> None:
        plan = plan_tool(_param_tool("t", {"verbose": {"type": "boolean"}}, ["verbose"]))
        assert plan.closed_params is not None
        assert plan.closed_params[0].kind == "boolean"

    def test_open_param_makes_tool_open(self) -> None:
        plan = plan_tool(_param_tool("t", {"path": {"type": "string"}}, ["path"]))
        assert plan.closed_params is None

    def test_mixed_closed_and_open_is_open(self) -> None:
        plan = plan_tool(
            _param_tool("t", {"verbose": {"type": "boolean"}, "path": {"type": "string"}}, ["verbose"])
        )
        assert plan.closed_params is None

    def test_required_missing_from_properties_is_open(self) -> None:
        plan = plan_tool(_param_tool("t", {}, ["path"]))
        assert plan.closed_params is None

    def test_non_object_schema_is_open(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {"type": "string"},
            },
        }
        assert plan_tool(tool).closed_params is None

    def test_non_dict_parameters_is_open(self) -> None:
        tool = {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": "oops"},
        }
        plan = plan_tool(tool)
        assert plan.closed_params is None

    def test_degenerate_def_has_no_name(self) -> None:
        assert plan_tool({"type": "function", "function": {}}).name is None


class TestBuildQuestions:
    def test_base_questions_without_args(self) -> None:
        questions, plans = build_questions(_tools(3), with_args=False)
        assert set(questions) == {"tool_selection", "needs_tool"}
        assert questions["needs_tool"]["type"] == "noul"
        assert len(plans) == 3

    def test_enum_and_bool_fan_out(self) -> None:
        tools = [
            _param_tool(
                "set_lights",
                {"room": {"enum": ["kitchen", "bedroom"]}, "on": {"type": "boolean"}},
                ["room", "on"],
            ),
            _tool("plain"),
        ]
        questions, plans = build_questions(tools, with_args=True)
        assert questions[arg_question_id(0, "room")]["type"] == "choice"
        assert questions[arg_question_id(0, "room")]["criteria"] == {"kitchen": None, "bedroom": None}
        assert questions[arg_question_id(0, "on")]["type"] == "noul"
        # Required parameters get no stated question.
        assert stated_question_id(0, "room") not in questions
        assert stated_question_id(0, "on") not in questions
        # Plain tools contribute no argument questions.
        assert arg_question_id(1, "") not in questions

    def test_optional_param_gets_stated_question(self) -> None:
        tools = [_param_tool("t", {"mode": {"enum": ["a", "b"]}}, [])]
        questions, _ = build_questions(tools, with_args=True)
        assert stated_question_id(0, "mode") in questions
        assert questions[stated_question_id(0, "mode")]["type"] == "noul"

    def test_const_param_has_no_question(self) -> None:
        tools = [_param_tool("t", {"mode": {"const": "fast"}}, ["mode"])]
        questions, plans = build_questions(tools, with_args=True)
        assert arg_question_id(0, "mode") not in questions
        assert plans[0].closed_params is not None

    def test_fan_out_cap_drops_late_plans(self) -> None:
        # Each tool carries one required enum parameter (cost 1); 100 tools
        # exceed the 96-question cap, so the trailing plans open up.
        tools = [
            _param_tool(f"t{i}", {"v": {"enum": ["a", "b"]}}, ["v"]) for i in range(100)
        ]
        questions, plans = build_questions(tools, with_args=True)
        arg_count = sum(1 for key in questions if key.startswith("arg:"))
        assert arg_count == 96
        assert plans[95].closed_params is not None
        assert plans[96].closed_params is None


class TestResolveArgs:
    def _plan(self, tool_index: int = 0) -> Any:
        tools = [
            _param_tool(
                "set_lights",
                {
                    "room": {"enum": ["kitchen", "bedroom"]},
                    "on": {"type": "boolean"},
                    "mode": {"const": "auto"},
                },
                ["room", "on"],
            )
        ]
        _, plans = build_questions(tools, with_args=True)
        return plans[tool_index]

    def test_full_resolution(self) -> None:
        answers = {
            arg_question_id(0, "room"): DecisionAnswer(
                type="choice", choice="kitchen", confidence=0.95
            ),
            arg_question_id(0, "on"): DecisionAnswer(type="noul", noul=0.9),
        }
        resolution = resolve_args(self._plan(), 0, answers, 0.8)
        # "mode" is an optional const: omitted, since no question can tell
        # whether the fixed value should be sent this call.
        assert resolution.args == {"room": "kitchen", "on": True}
        assert resolution.certainty == 0.9  # min(0.95, max(0.9, 0.1)) == 0.9

    def test_required_const_is_filled(self) -> None:
        tools = [_param_tool("t", {"mode": {"const": "auto"}}, ["mode"])]
        _, plans = build_questions(tools, with_args=True)
        resolution = resolve_args(plans[0], 0, {}, 0.8)
        assert resolution.args == {"mode": "auto"}
        assert resolution.certainty == 1.0

    def test_optional_omitted_when_not_stated(self) -> None:
        tools = [_param_tool("t", {"mode": {"enum": ["a", "b"]}}, [])]
        _, plans = build_questions(tools, with_args=True)
        answers = {
            stated_question_id(0, "mode"): DecisionAnswer(type="noul", noul=0.1),
            arg_question_id(0, "mode"): DecisionAnswer(type="choice", choice="a", confidence=0.99),
        }
        resolution = resolve_args(plans[0], 0, answers, 0.8)
        assert resolution.args == {}

    def test_optional_included_when_stated(self) -> None:
        tools = [_param_tool("t", {"mode": {"enum": ["a", "b"]}}, [])]
        _, plans = build_questions(tools, with_args=True)
        answers = {
            stated_question_id(0, "mode"): DecisionAnswer(type="noul", noul=0.95),
            arg_question_id(0, "mode"): DecisionAnswer(type="choice", choice="b", confidence=0.9),
        }
        resolution = resolve_args(plans[0], 0, answers, 0.8)
        assert resolution.args == {"mode": "b"}
        assert resolution.certainty == 0.9

    def test_missing_stated_answer_defers(self) -> None:
        tools = [_param_tool("t", {"mode": {"enum": ["a", "b"]}}, [])]
        _, plans = build_questions(tools, with_args=True)
        resolution = resolve_args(plans[0], 0, {}, 0.8)
        assert resolution.args is None
        assert resolution.deferred_reason == "args_unresolved"

    def test_choice_without_confidence_defers(self) -> None:
        answers = {
            arg_question_id(0, "room"): DecisionAnswer(type="choice", choice="kitchen"),
            arg_question_id(0, "on"): DecisionAnswer(type="noul", noul=0.95),
        }
        resolution = resolve_args(self._plan(), 0, answers, 0.8)
        assert resolution.args is None
        assert resolution.deferred_reason == "args_unresolved"

    def test_enum_choice_outside_values_defers(self) -> None:
        answers = {
            arg_question_id(0, "room"): DecisionAnswer(
                type="choice", choice="attic", confidence=0.99
            ),
            arg_question_id(0, "on"): DecisionAnswer(type="noul", noul=0.95),
        }
        resolution = resolve_args(self._plan(), 0, answers, 0.8)
        assert resolution.deferred_reason == "args_unresolved"

    def test_below_threshold_reports_certainty(self) -> None:
        answers = {
            arg_question_id(0, "room"): DecisionAnswer(
                type="choice", choice="kitchen", confidence=0.5
            ),
            arg_question_id(0, "on"): DecisionAnswer(type="noul", noul=0.99),
        }
        resolution = resolve_args(self._plan(), 0, answers, 0.8)
        assert resolution.args is None
        assert resolution.certainty == 0.5
        assert resolution.deferred_reason == "args_below_threshold"

    def test_boolean_boundary_is_half_certain(self) -> None:
        answers = {
            arg_question_id(0, "room"): DecisionAnswer(
                type="choice", choice="kitchen", confidence=0.99
            ),
            arg_question_id(0, "on"): DecisionAnswer(type="noul", noul=0.5),
        }
        resolution = resolve_args(self._plan(), 0, answers, 0.8)
        assert resolution.deferred_reason == "args_below_threshold"


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

    async def test_missing_tool_answer_defers(self) -> None:
        class NeedsOnlyProvider(FakeDecisionProvider):
            async def decide(self, state: Any, questions: dict[str, Any]) -> DecisionResult:
                self.calls.append({"state": state, "questions": questions})
                return DecisionResult(
                    provider="fake",
                    model="fake-1",
                    answers={"needs_tool": DecisionAnswer(type="noul", noul=0.9)},
                    latency_ms=12,
                )

        decider = ToolSelectionDecider(
            NeedsOnlyProvider(), mode="filter", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "hi"}], _tools(3), "chat")
        assert selection.skipped_reason == "error:missing answer"


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
        assert selection.needs_tool == 0.9
        assert selection.deferred_reason is None
        # Eligibility still recorded for calibration even in shadow mode.
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_eligible is True
        assert selection.fastpath_tool is None
        assert selection.fastpath_args == {}

    async def test_filter_below_threshold_does_not_narrow(self) -> None:
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
        assert selection.filtered_tools is None
        assert selection.deferred_reason == "low_confidence"
        assert selection.fastpath_eligible is False

    async def test_filter_top_k_strategy_ranks_and_truncates(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(
                type="choice",
                choice="tool_3",
                probabilities={"tool_3": 0.6, "tool_0": 0.3, "tool_2": 0.1},
                confidence=0.9,
            )
        )
        decider = ToolSelectionDecider(
            provider,
            mode="filter",
            scopes=["chat"],
            threshold=0.75,
            top_k=3,
            min_tools=2,
            filter_strategy="top_k",
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(5), "chat")
        assert selection.filtered_tools is not None
        names = [t["function"]["name"] for t in selection.filtered_tools]
        assert names == ["tool_3", "tool_0", "tool_2"]

    async def test_filter_pin_strategy_keeps_chosen_only(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_3", confidence=0.9)
        )
        decider = ToolSelectionDecider(
            provider,
            mode="filter",
            scopes=["chat"],
            threshold=0.75,
            top_k=3,
            min_tools=2,
            filter_strategy="pin",
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(5), "chat")
        assert selection.filtered_tools is not None
        names = [t["function"]["name"] for t in selection.filtered_tools]
        assert names == ["tool_3"]

    async def test_fastpath_zero_arg_tool(self) -> None:
        tools = _tools(3)
        high = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_2", confidence=0.9, probabilities={"tool_2": 0.9})
        )
        decider = ToolSelectionDecider(
            high, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool == "tool_2"
        assert selection.fastpath_args == {}

        low = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_2", confidence=0.5, probabilities={"tool_2": 0.5})
        )
        decider = ToolSelectionDecider(
            low, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None
        assert selection.fastpath_eligible is False
        assert selection.deferred_reason == "low_confidence"

        no_confidence = FakeDecisionProvider(DecisionAnswer(type="choice", choice="tool_2"))
        decider = ToolSelectionDecider(
            no_confidence, mode="fastpath", scopes=["chat"], threshold=0.0, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None

    async def test_fastpath_with_closed_set_args(self) -> None:
        tools = [
            _tool("tool_0"),
            _param_tool(
                "set_lights",
                {"room": {"enum": ["kitchen", "bedroom"]}, "on": {"type": "boolean"}},
                ["room", "on"],
            ),
            _tool("tool_2"),
        ]
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="set_lights", confidence=0.95),
            extra_answers={
                arg_question_id(1, "room"): DecisionAnswer(
                    type="choice", choice="kitchen", confidence=0.97
                ),
                arg_question_id(1, "on"): DecisionAnswer(type="noul", noul=0.93),
            },
        )
        decider = ToolSelectionDecider(
            provider, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool == "set_lights"
        assert selection.fastpath_args == {"room": "kitchen", "on": True}
        assert selection.args_certainty == 0.93

    async def test_fastpath_args_below_threshold_still_pins_for_llm(self) -> None:
        tools = [
            _tool("tool_0"),
            _param_tool(
                "set_lights",
                {"room": {"enum": ["kitchen", "bedroom"]}, "on": {"type": "boolean"}},
                ["room", "on"],
            ),
            _tool("tool_2"),
        ]
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="set_lights", confidence=0.95),
            extra_answers={
                arg_question_id(1, "room"): DecisionAnswer(
                    type="choice", choice="kitchen", confidence=0.5
                ),
                arg_question_id(1, "on"): DecisionAnswer(type="noul", noul=0.99),
            },
        )
        decider = ToolSelectionDecider(
            provider, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None
        assert selection.fastpath_eligible is False
        assert selection.deferred_reason == "args_below_threshold"
        # The filter side still acted: the LLM sees only the pinned tool.
        assert selection.filtered_tools is not None
        assert [t["function"]["name"] for t in selection.filtered_tools] == ["set_lights"]

    async def test_open_args_never_fastpath_but_pin(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_1", confidence=0.99)
        )
        tools = [_param_tool(f"tool_{i}", {"path": {"type": "string"}}, ["path"]) for i in range(3)]
        decider = ToolSelectionDecider(
            provider, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], tools, "chat")
        assert selection.fastpath_tool is None
        assert selection.fastpath_eligible is False
        assert selection.deferred_reason is None
        assert selection.filtered_tools is not None
        assert [t["function"]["name"] for t in selection.filtered_tools] == ["tool_1"]

    async def test_extract_args_disabled_allows_const_only(self) -> None:
        enum_tool = _param_tool("pick", {"mode": {"enum": ["a", "b"]}}, ["mode"])
        const_tool = _param_tool("fixed", {"mode": {"const": "auto"}}, ["mode"])
        plain_tool = _tool("plain")
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="pick", confidence=0.99)
        )
        decider = ToolSelectionDecider(
            provider,
            mode="fastpath",
            scopes=["chat"],
            threshold=0.75,
            top_k=5,
            min_tools=2,
            extract_args=False,
        )
        selection = await decider.select(
            [{"role": "user", "content": "go"}], [enum_tool, const_tool, plain_tool], "chat"
        )
        assert selection.fastpath_eligible is False
        assert provider.calls[0]["questions"].keys() == {"tool_selection", "needs_tool"}

        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="fixed", confidence=0.99)
        )
        decider = ToolSelectionDecider(
            provider,
            mode="fastpath",
            scopes=["chat"],
            threshold=0.75,
            top_k=5,
            min_tools=2,
            extract_args=False,
        )
        selection = await decider.select(
            [{"role": "user", "content": "go"}], [enum_tool, const_tool, plain_tool], "chat"
        )
        assert selection.fastpath_eligible is True
        assert selection.fastpath_args == {}

    async def test_needs_tool_disagreement_defers(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_1", confidence=0.99), needs=0.1
        )
        decider = ToolSelectionDecider(
            provider, mode="fastpath", scopes=["chat"], threshold=0.75, top_k=5, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(3), "chat")
        assert selection.deferred_reason == "answers_disagree"
        assert selection.filtered_tools is None
        assert selection.fastpath_eligible is False
        assert selection.needs_tool == 0.1

    async def test_none_disagreement_defers(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="none", confidence=0.99), needs=0.9
        )
        decider = ToolSelectionDecider(
            provider, mode="filter", scopes=["chat"], threshold=0.75, top_k=3, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(3), "chat")
        assert selection.deferred_reason == "answers_disagree"

    async def test_missing_needs_tool_defers_action(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_1", confidence=0.99), needs=None
        )
        decider = ToolSelectionDecider(
            provider, mode="filter", scopes=["chat"], threshold=0.75, top_k=3, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(3), "chat")
        assert selection.deferred_reason == "needs_tool_missing"
        assert selection.filtered_tools is None

    async def test_none_choice_disables_filter(self) -> None:
        provider = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="none", confidence=0.99), needs=0.1
        )
        decider = ToolSelectionDecider(
            provider, mode="filter", scopes=["chat"], threshold=0.75, top_k=3, min_tools=2
        )
        selection = await decider.select([{"role": "user", "content": "go"}], _tools(3), "chat")
        assert selection.chosen == "none"
        assert selection.filtered_tools is None
        assert selection.deferred_reason is None
