"""LLM provider wrapper implementing the decision layer.

Wraps the outermost chat provider (the same seam ``FallbackProvider`` uses)
so the agent core stays untouched. Per model round it can: narrow the tool
definitions the LLM sees (filter), answer with a synthetic tool call without
invoking the LLM (fastpath, zero-required-arg tools only), and log the
decision next to what the LLM actually did (shadow). Everything is fail-open:
any decision-layer failure defers to the wrapped provider unchanged.
"""

# pyright: reportIncompatibleMethodOverride=false, reportIncompatibleVariableOverride=false

from __future__ import annotations

import uuid
from typing import Any, cast

from loguru import logger

from nanobot.decisions.selector import (
    ToolSelection,
    ToolSelectionDecider,
    detect_decision_scope,
)
from nanobot.decisions.store import DecisionRecord, DecisionsStore, now_ms
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallRequest,
)

_FASTPATH_MARKER = "decision"


def _current_session_key() -> str:
    from nanobot.agent.tools.context import current_request_context

    ctx = current_request_context()
    if ctx is None:
        return "unknown"
    session_key = ctx.session_key
    return session_key if session_key else "unknown"


def _fastpath_response(
    tool_name: str,
    arguments: dict[str, Any],
    latency_ms: int | None,
) -> LLMResponse:
    """Synthesize the response the runner expects when the LLM requests a tool."""
    usage = LLMUsage.empty_request()
    if latency_ms is not None and latency_ms > 0:
        usage = usage.with_timing(generation_ms=latency_ms, ttft_ms=latency_ms)
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                id=f"{_FASTPATH_MARKER}-{uuid.uuid4().hex[:12]}",
                name=tool_name,
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
        usage=usage,
    )


def _llm_tool_names(response: LLMResponse | None) -> list[str] | None:
    if response is None:
        return None
    # ``name`` is typed str but a gateway can emit a degenerate call; the
    # truthiness filter drops those without changing the typed surface.
    names = [call.name for call in response.tool_calls if call.name]
    return names or None


class DecisionProviderWrapper(LLMProvider):
    """Delegating wrapper; intercepts single-attempt ``chat``/``chat_stream``.

    Retry orchestration, call observation and stream timing are inherited from
    the base class, so usage records keep the inner provider's identity and
    the decision layer adds exactly one decision call per physical attempt.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        decider: ToolSelectionDecider,
        store: DecisionsStore,
        mode: str,
    ) -> None:
        inner_generation = inner.generation
        self._inner = inner
        super().__init__(provider_name=inner.provider_name)
        self._inner.generation = inner_generation
        self.supports_stream_recover_callback = bool(
            getattr(inner, "supports_stream_recover_callback", False)
        )
        self._decider = decider
        self._store = store
        self._mode = mode

    # ------------------------------------------------------------------
    # Delegation surface (mirrors FallbackProvider)

    @property
    def inner(self) -> LLMProvider:
        """The wrapped provider, for attachment points that see through wrappers."""
        return self._inner

    @property
    def generation(self) -> GenerationSettings:
        return self._inner.generation

    @generation.setter
    def generation(self, value: GenerationSettings) -> None:
        self._inner.generation = value

    def get_default_model(self) -> str:
        return self._inner.get_default_model()

    def can_resume_conversation_state(self, *args: Any, **kwargs: Any) -> bool:
        return self._inner.can_resume_conversation_state(*args, **kwargs)

    def supports_native_compaction(self, *args: Any, **kwargs: Any) -> bool:
        return self._inner.supports_native_compaction(*args, **kwargs)

    def supports_pre_request_compaction(self, *args: Any, **kwargs: Any) -> bool:
        return self._inner.supports_pre_request_compaction(*args, **kwargs)

    # ------------------------------------------------------------------
    # Interception points (single-attempt calls made by the retry machinery)

    async def chat(self, **kwargs: Any) -> LLMResponse:
        return await self._dispatch(stream=False, kwargs=dict(kwargs))

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        return await self._dispatch(stream=True, kwargs=dict(kwargs))

    async def chat_with_context(self, *, provider_context: Any, **kwargs: Any) -> LLMResponse:
        return await self._dispatch(
            stream=False, kwargs={"provider_context": provider_context, **kwargs}
        )

    async def chat_stream_with_context(self, *, provider_context: Any, **kwargs: Any) -> LLMResponse:
        return await self._dispatch(
            stream=True, kwargs={"provider_context": provider_context, **kwargs}
        )

    # ------------------------------------------------------------------
    # Decision flow

    async def _call_inner(self, stream: bool, kwargs: dict[str, Any]) -> LLMResponse:
        provider_context = kwargs.pop("provider_context", None)
        if provider_context is not None:
            method = self._inner.chat_stream_with_context if stream else self._inner.chat_with_context
            return await method(provider_context=provider_context, **kwargs)
        method = self._inner.chat_stream if stream else self._inner.chat
        return await method(**kwargs)

    async def _dispatch(self, *, stream: bool, kwargs: dict[str, Any]) -> LLMResponse:
        messages = kwargs.get("messages")
        tools = kwargs.get("tools")
        model = kwargs.get("model")
        selection = await self._decide(messages, tools)

        # Scope skips mean the layer is intentionally absent for this turn
        # (e.g. a subagent round with scopes=["chat"]); recording them would
        # only produce per-round noise in the logs.
        if selection.skipped_reason is not None and selection.skipped_reason.startswith("scope:"):
            return await self._call_inner(stream, kwargs)

        if selection.fastpath_tool is not None:
            latency = selection.decision.latency_ms if selection.decision is not None else None
            arguments = selection.fastpath_args if selection.fastpath_args is not None else {}
            response = _fastpath_response(selection.fastpath_tool, arguments, latency)
            llm_response: LLMResponse | None = None
            if self._mode == "dual":
                # Quantification mode: also call the LLM (with the filtered
                # list, matching what the fallback path would have seen) and
                # discard its output, logging both sides of the A/B.
                dual_kwargs = self._apply_filter(kwargs, selection)
                llm_response = await self._call_inner(stream, dual_kwargs)
            self._record(selection, model, response=None, llm_response=llm_response)
            return response

        forwarded = self._apply_filter(kwargs, selection)
        response = await self._call_inner(stream, forwarded)
        self._record(selection, model, response=response, llm_response=response)
        return response

    async def _decide(
        self,
        messages: Any,
        tools: Any,
    ) -> ToolSelection:
        if not isinstance(messages, list) or (tools is not None and not isinstance(tools, list)):
            return ToolSelection(skipped_reason="malformed_request")
        try:
            scope = detect_decision_scope()
        except Exception as e:
            logger.warning("Decision scope detection failed; deferring: {}", e)
            return ToolSelection(skipped_reason="scope:error")
        try:
            typed_messages = cast("list[dict[str, Any]]", messages)
            typed_tools = cast("list[dict[str, Any]] | None", tools)
            return await self._decider.select(typed_messages, typed_tools, scope)
        except Exception as e:
            logger.exception("Decision call failed; deferring to LLM: {}", e)
            return ToolSelection(skipped_reason=f"error:{e}")

    def _apply_filter(
        self, kwargs: dict[str, Any], selection: ToolSelection
    ) -> dict[str, Any]:
        if selection.filtered_tools is None:
            return kwargs
        return {**kwargs, "tools": selection.filtered_tools}

    def _record(
        self,
        selection: ToolSelection,
        model: Any,
        *,
        response: LLMResponse | None,
        llm_response: LLMResponse | None,
    ) -> None:
        try:
            decision = selection.decision
            usage = None
            finish_reason = None
            source = llm_response if llm_response is not None else response
            if source is not None:
                usage = source.usage
                finish_reason = source.finish_reason
            self._store.record(
                DecisionRecord(
                    ts_ms=now_ms(),
                    session_key=_current_session_key(),
                    scope=selection.scope or "chat",
                    mode=self._mode,
                    provider=decision.provider if decision is not None else "",
                    model=decision.model if decision is not None else "",
                    chosen=selection.chosen,
                    confidence=selection.confidence,
                    probabilities=selection.probabilities,
                    skipped_reason=selection.skipped_reason,
                    error=decision.error if decision is not None else None,
                    decision_latency_ms=decision.latency_ms if decision is not None else None,
                    decision_input_tokens=decision.input_tokens if decision is not None else None,
                    decision_output_tokens=decision.output_tokens if decision is not None else None,
                    decision_cost_usd=decision.cost_usd if decision is not None else None,
                    llm_model=model if isinstance(model, str) else None,
                    llm_tool_names=_llm_tool_names(source),
                    llm_finish_reason=finish_reason,
                    llm_input_tokens=usage.input_tokens if usage is not None else None,
                    llm_output_tokens=usage.output_tokens if usage is not None else None,
                    llm_generation_ms=source.generation_ms if source is not None else None,
                    fastpath_eligible=selection.fastpath_eligible,
                    fastpath_executed=selection.fastpath_tool is not None,
                    fastpath_args=selection.fastpath_args,
                    args_certainty=selection.args_certainty,
                    needs_tool=selection.needs_tool,
                    deferred_reason=selection.deferred_reason,
                    filtered_tool_count=(
                        len(selection.filtered_tools) if selection.filtered_tools is not None else None
                    ),
                    total_tool_count=selection.total_tool_count,
                )
            )
        except Exception as e:  # pragma: no cover - fail-open by contract
            logger.warning("Decision record construction failed: {}", e)


def wrap_provider_with_decisions(provider: LLMProvider, config: Any) -> LLMProvider:
    """Wrap ``provider`` with the decision layer when enabled in config.

    Returns the provider unchanged when the layer is disabled or no decision
    provider can be built, so upstream behavior is the default path.
    """
    from nanobot.decisions.factory import make_decision_provider

    decision_provider = make_decision_provider(config)
    if decision_provider is None:
        return provider
    decisions = config.decisions
    decider = ToolSelectionDecider(
        decision_provider,
        mode=decisions.mode,
        scopes=list(decisions.scopes),
        threshold=decisions.threshold,
        top_k=decisions.top_k,
        min_tools=decisions.min_tools,
        arg_threshold=decisions.arg_threshold,
        filter_strategy=decisions.filter_strategy,
        extract_args=decisions.extract_args,
    )
    return DecisionProviderWrapper(
        provider,
        decider=decider,
        store=DecisionsStore(),
        mode=decisions.mode,
    )
