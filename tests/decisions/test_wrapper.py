"""Tests for the decision provider wrapper (shadow / filter / fastpath / dual)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.decisions.selector import ToolSelectionDecider
from nanobot.decisions.store import DecisionsStore
from nanobot.decisions.types import DecisionAnswer, DecisionResult
from nanobot.decisions.wrapper import DecisionProviderWrapper
from nanobot.llm_usage.context import llm_usage_source
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_MESSAGES = [{"role": "user", "content": "list my files please"}]


def _tool(name: str, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} does things",
            "parameters": {"type": "object", "properties": {}, "required": required or []},
        },
    }


def _tools() -> list[dict[str, Any]]:
    return [_tool(f"tool_{i}", required=["path"]) for i in range(4)] + [_tool("status", required=[])]


class FakeInnerProvider(LLMProvider):
    """Single-attempt fake mirroring what the retry machinery calls."""

    def __init__(self, *, with_tool_call: bool = True) -> None:
        super().__init__(provider_name="fake-llm")
        self.with_tool_call = with_tool_call
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "model": model, "stream": False})
        return self._response()

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "model": model, "stream": True})
        return self._response()

    def _response(self) -> LLMResponse:
        if self.with_tool_call:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="llm-1", name="tool_0", arguments={"path": "/tmp"})],
                finish_reason="tool_calls",
                usage=None,
            )
        return LLMResponse(content="all done", finish_reason="stop")

    def get_default_model(self) -> str:
        return "fake-model"


class FakeDecisionProvider:
    def __init__(self, answer: DecisionAnswer | None = None, error: str | None = None) -> None:
        self.answer = answer or DecisionAnswer(type="choice", choice="none")
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def decide(self, state: Any, questions: dict[str, Any]) -> DecisionResult:
        self.calls.append({"state": state})
        return DecisionResult(
            provider="fake",
            model="fake-1.13",
            answers={} if self.error is not None else {"tool_selection": self.answer},
            error=self.error,
            latency_ms=42,
            input_tokens=120,
            output_tokens=8,
            cost_usd=0.000002,
        )


def _wrapper(
    decision_provider: FakeDecisionProvider,
    store: DecisionsStore,
    *,
    mode: str,
    scopes: list[str] | None = None,
    threshold: float = 0.75,
    top_k: int = 3,
    inner: FakeInnerProvider | None = None,
) -> tuple[DecisionProviderWrapper, FakeInnerProvider]:
    decider = ToolSelectionDecider(
        decision_provider,
        mode=mode,
        scopes=scopes or ["chat"],
        threshold=threshold,
        top_k=top_k,
        min_tools=2,
    )
    fake_inner = inner or FakeInnerProvider()
    wrapper = DecisionProviderWrapper(fake_inner, decider=decider, store=store, mode=mode)
    return wrapper, fake_inner


@contextmanager
def chat_scope() -> Iterator[None]:
    ctx = RequestContext(channel="websocket", chat_id="c1", session_key="chat-key")
    with request_context(ctx), llm_usage_source("user"):
        yield


class TestShadowMode:
    async def test_llm_sees_full_tools_and_both_sides_logged(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(
                type="choice",
                choice="tool_0",
                probabilities={"tool_0": 0.85, "status": 0.15},
                confidence=0.9,
            )
        )
        wrapper, inner = _wrapper(decision, store, mode="shadow")
        with chat_scope():
            response = await wrapper.chat_stream_with_retry(
                messages=_MESSAGES, tools=_tools(), model="fake-model"
            )
        # LLM answered (shadow never intercepts)
        assert len(inner.calls) == 1
        assert response.content == "" and response.tool_calls[0].name == "tool_0"
        # Full tool list passed through
        assert len(inner.calls[0]["tools"]) == 5
        records = list(store.iter_records())
        assert len(records) == 1
        record = records[0]
        assert record.chosen == "tool_0"
        assert record.llm_tool_names == ["tool_0"]
        assert record.mode == "shadow"
        assert record.decision_cost_usd == 0.000002
        assert record.fastpath_executed is False
        # tool_0 requires args -> not fastpath eligible despite high confidence
        assert record.fastpath_eligible is False

    async def test_scope_outside_config_is_not_recorded(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(DecisionAnswer(type="choice", choice="status", confidence=0.99))
        wrapper, inner = _wrapper(decision, store, mode="shadow", scopes=["cron"])
        with chat_scope():
            await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        assert decision.calls == []
        assert len(inner.calls) == 1
        assert list(store.iter_records()) == []


class TestFilterMode:
    async def test_llm_sees_narrowed_list(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(
                type="choice",
                choice="status",
                probabilities={"status": 0.8, "tool_2": 0.2},
                confidence=0.5,
            )
        )
        wrapper, inner = _wrapper(decision, store, mode="filter", top_k=2)
        with chat_scope():
            await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        seen = inner.calls[0]["tools"]
        names = [t["function"]["name"] for t in seen]
        assert names[0] == "status"
        assert names[1] == "tool_2"
        assert len(names) == 2
        record = list(store.iter_records())[0]
        assert record.filtered_tool_count == 2
        assert record.total_tool_count == 5


class TestFastpathMode:
    async def test_zero_arg_tool_executes_without_llm(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="status", confidence=0.95, probabilities={"status": 0.95})
        )
        wrapper, inner = _wrapper(decision, store, mode="fastpath")
        with chat_scope():
            response = await wrapper.chat_stream_with_retry(
                messages=_MESSAGES, tools=_tools(), model="fake-model"
            )
        assert inner.calls == []
        assert response.should_execute_tools is True
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "status"
        assert call.arguments == {}
        assert call.id.startswith("decision-")
        assert response.usage is not None and response.usage.request_count == 1
        record = list(store.iter_records())[0]
        assert record.fastpath_executed is True
        assert record.llm_tool_names is None

    async def test_required_args_tool_falls_back_to_llm(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="tool_1", confidence=0.99)
        )
        wrapper, inner = _wrapper(decision, store, mode="fastpath")
        with chat_scope():
            response = await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        assert len(inner.calls) == 1
        assert response.tool_calls[0].name == "tool_0"
        record = list(store.iter_records())[0]
        assert record.fastpath_executed is False
        assert record.fastpath_eligible is False

    async def test_low_confidence_falls_back(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="status", confidence=0.6)
        )
        wrapper, inner = _wrapper(decision, store, mode="fastpath", threshold=0.75)
        with chat_scope():
            await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        assert len(inner.calls) == 1
        record = list(store.iter_records())[0]
        assert record.fastpath_eligible is False


class TestDualMode:
    async def test_llm_called_for_measurement_but_synthetic_returned(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="status", confidence=0.95, probabilities={"status": 0.95})
        )
        wrapper, inner = _wrapper(decision, store, mode="dual")
        with chat_scope():
            response = await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        # LLM was invoked (measurement) but its output discarded
        assert len(inner.calls) == 1
        assert response.tool_calls[0].name == "status"
        assert response.tool_calls[0].id.startswith("decision-")
        record = list(store.iter_records())[0]
        assert record.fastpath_executed is True
        assert record.llm_tool_names == ["tool_0"]


class TestFailOpen:
    async def test_decision_error_passes_through(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(error="request failed")
        wrapper, inner = _wrapper(decision, store, mode="fastpath")
        with chat_scope():
            response = await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        assert len(inner.calls) == 1
        assert len(inner.calls[0]["tools"]) == 5
        assert response.tool_calls[0].name == "tool_0"
        record = list(store.iter_records())[0]
        assert record.skipped_reason == "error:request failed"

    async def test_decision_exception_passes_through(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)

        class Exploding:
            async def decide(self, state: Any, questions: dict[str, Any]) -> DecisionResult:
                raise RuntimeError("boom")

        decider = ToolSelectionDecider(
            Exploding(), mode="fastpath", scopes=["chat"], threshold=0.75, top_k=3, min_tools=2
        )
        inner = FakeInnerProvider()
        wrapper = DecisionProviderWrapper(inner, decider=decider, store=store, mode="fastpath")
        with chat_scope():
            response = await wrapper.chat_stream_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        assert len(inner.calls) == 1
        assert response.tool_calls[0].name == "tool_0"

    async def test_disabled_layers_never_wrap(self, tmp_path) -> None:
        from nanobot.config.schema import Config
        from nanobot.decisions.wrapper import wrap_provider_with_decisions

        config = Config()
        config.decisions.enabled = False
        inner = FakeInnerProvider()
        assert wrap_provider_with_decisions(inner, config) is inner


class TestDelegationSurface:
    def test_delegates_identity_and_generation(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider()
        wrapper, inner = _wrapper(decision, store, mode="shadow")
        assert wrapper.provider_name == "fake-llm"
        assert wrapper.get_default_model() == "fake-model"
        assert wrapper.generation is inner.generation
        wrapper.generation = wrapper.generation  # setter delegates
        assert wrapper.supports_native_compaction() == inner.supports_native_compaction()

    async def test_non_streaming_chat_path_intercepted(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        decision = FakeDecisionProvider(
            DecisionAnswer(type="choice", choice="status", confidence=0.95, probabilities={"status": 1.0})
        )
        wrapper, inner = _wrapper(decision, store, mode="fastpath")
        with chat_scope():
            response = await wrapper.chat_with_retry(messages=_MESSAGES, tools=_tools(), model="m")
        assert inner.calls == []
        assert response.tool_calls[0].name == "status"
