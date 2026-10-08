"""Tool-selection decision logic: state building, gating, and result policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from loguru import logger

from nanobot.decisions.base import DecisionProvider
from nanobot.decisions.types import ChoiceQuestion, DecisionResult

DecisionScope = Literal["chat", "subagent", "cron"]

_NONE_OPTION = "none"
_MAX_CHOICE_OPTIONS = 255
_RECENT_MESSAGES = 12
_MESSAGE_TEXT_LIMIT = 400
_LAST_USER_LIMIT = 2000
_DESCRIPTION_LIMIT = 300


def detect_decision_scope() -> DecisionScope:
    """Classify the current call site into a decisions scope.

    Subagent turns are identified by their ``subagent:`` session-key prefix;
    everything else follows the LLM usage source (cron triggers, dream
    consolidation and system turns all map to the ``cron`` automation bucket).
    """
    from nanobot.agent.tools.context import current_request_context
    from nanobot.llm_usage.context import current_llm_usage_source

    ctx = current_request_context()
    session_key = getattr(ctx, "session_key", None)
    if isinstance(session_key, str) and session_key.startswith("subagent:"):
        return "subagent"
    source = current_llm_usage_source()
    if source in ("cron", "dream", "system"):
        return "cron"
    return "chat"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        entries = cast("list[Any]", content)
        parts: list[str] = []
        for part in entries:
            if isinstance(part, dict):
                chunk = cast("dict[str, Any]", part)
                text = chunk.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""


def build_state(messages: list[dict[str, Any]], tool_catalog: list[dict[str, str]]) -> dict[str, Any]:
    """Assemble the decision ``state`` from the transcript and tool catalog."""
    recent: list[dict[str, str]] = []
    last_user: str | None = None
    for message in messages[-_RECENT_MESSAGES:]:
        role = message.get("role")
        if role not in ("user", "assistant", "tool"):
            continue
        text = _message_text(message.get("content"))[:_MESSAGE_TEXT_LIMIT]
        recent.append({"role": str(role), "content": text})
        if role == "user":
            last_user = text
    if last_user is None:
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = _message_text(message.get("content"))[:_LAST_USER_LIMIT]
                break
    return {
        "last_user_message": last_user,
        "recent_messages": recent,
        "available_tools": tool_catalog,
    }


def _tool_name(tool_def: dict[str, Any]) -> str | None:
    function = tool_def.get("function")
    if not isinstance(function, dict):
        return None
    entry = cast("dict[str, Any]", function)
    name = entry.get("name")
    return name if isinstance(name, str) and name else None


def tool_catalog(tools: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Extract ``{name, description}`` entries from OpenAI-style tool definitions."""
    catalog: list[dict[str, str]] = []
    if not tools:
        return catalog
    for tool_def in tools:
        name = _tool_name(tool_def)
        if name is None:
            continue
        function = tool_def["function"]
        description = function.get("description")
        catalog.append(
            {
                "name": name,
                "description": (
                    description[:_DESCRIPTION_LIMIT] if isinstance(description, str) else ""
                ),
            }
        )
    return catalog


def has_zero_required_params(tool_def: dict[str, Any] | None) -> bool:
    """Whether the tool can be called with an empty arguments object."""
    if tool_def is None:
        return False
    function = tool_def.get("function")
    if not isinstance(function, dict):
        return False
    entry = cast("dict[str, Any]", function)
    parameters = entry.get("parameters")
    if not isinstance(parameters, dict):
        return True
    schema = cast("dict[str, Any]", parameters)
    required = schema.get("required")
    return not required


def build_choice_question(catalog: list[dict[str, str]]) -> ChoiceQuestion:
    criteria: dict[str, str | None] = {
        entry["name"]: entry["description"] or None for entry in catalog
    }
    criteria[_NONE_OPTION] = "No tool call is needed; the assistant should answer directly."
    return ChoiceQuestion(
        type="choice",
        instructions=(
            "Which available tool, if any, should the assistant call next to make "
            "progress on the user's request? Judge strictly from the conversation "
            "and each tool's description."
        ),
        criteria=criteria,
    )


@dataclass(frozen=True)
class ToolSelection:
    """Outcome of one selection round, before any LLM call."""

    skipped_reason: str | None
    scope: DecisionScope | None = None
    decision: DecisionResult | None = None
    chosen: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    filtered_tools: list[dict[str, Any]] | None = field(default=None, repr=False)
    fastpath_eligible: bool = False
    fastpath_tool: str | None = None
    total_tool_count: int | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


class ToolSelectionDecider:
    """Applies config policy around one decision-model call per model round."""

    def __init__(
        self,
        provider: DecisionProvider,
        *,
        mode: str,
        scopes: list[str],
        threshold: float,
        top_k: int,
        min_tools: int,
    ) -> None:
        self._provider = provider
        self._mode = mode
        self._scopes = frozenset(scopes)
        self._threshold = threshold
        self._top_k = top_k
        self._min_tools = min_tools

    async def select(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        scope: DecisionScope,
    ) -> ToolSelection:
        tool_defs = list(tools or [])
        total = len(tool_defs)
        if scope not in self._scopes:
            return ToolSelection(skipped_reason=f"scope:{scope}", scope=scope, total_tool_count=total)
        if total == 0:
            return ToolSelection(skipped_reason="no_tools", scope=scope, total_tool_count=total)
        if total < self._min_tools:
            return ToolSelection(skipped_reason="min_tools", scope=scope, total_tool_count=total)
        if total + 1 > _MAX_CHOICE_OPTIONS:
            return ToolSelection(skipped_reason="too_many_tools", scope=scope, total_tool_count=total)

        catalog = tool_catalog(tool_defs)
        by_name = {_tool_name(tool_def): tool_def for tool_def in tool_defs if _tool_name(tool_def)}
        state = build_state(messages, catalog)
        result = await self._provider.decide(
            state, {"tool_selection": build_choice_question(catalog)}
        )
        if not result.ok:
            return ToolSelection(
                skipped_reason=f"error:{result.error}", scope=scope, decision=result, total_tool_count=total
            )
        answer = result.answer_for("tool_selection")
        if answer is None or answer.choice is None:
            return ToolSelection(
                skipped_reason="error:missing answer", scope=scope, decision=result, total_tool_count=total
            )

        chosen = answer.choice
        if chosen != _NONE_OPTION and chosen not in by_name:
            logger.warning("Decision model chose unknown tool {!r}; deferring", chosen)
            return ToolSelection(
                skipped_reason=f"unknown_tool:{chosen}",
                scope=scope,
                decision=result,
                chosen=chosen,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
                total_tool_count=total,
            )

        filtered: list[dict[str, Any]] | None = None
        if self._mode in ("filter", "fastpath", "dual") and chosen != _NONE_OPTION:
            filtered = self._rank_tools(tool_defs, chosen, answer.probabilities)

        # Eligibility is mode-independent: shadow and filter logs must be able
        # to answer "would the fast path have fired?" for calibration reports.
        eligible = bool(
            chosen != _NONE_OPTION
            and answer.confidence is not None
            and answer.confidence >= self._threshold
            and has_zero_required_params(by_name.get(chosen))
        )
        fastpath_tool = chosen if eligible and self._mode in ("fastpath", "dual") else None

        return ToolSelection(
            skipped_reason=None,
            scope=scope,
            decision=result,
            chosen=chosen,
            confidence=answer.confidence,
            probabilities=answer.probabilities,
            filtered_tools=filtered,
            fastpath_eligible=eligible,
            fastpath_tool=fastpath_tool,
            total_tool_count=total,
        )

    def _rank_tools(
        self,
        tool_defs: list[dict[str, Any]],
        chosen: str,
        probabilities: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """Chosen tool first, the rest by decision probability (stable otherwise)."""
        chosen_def = None
        rest: list[dict[str, Any]] = []
        for tool_def in tool_defs:
            if _tool_name(tool_def) == chosen:
                chosen_def = tool_def
            else:
                rest.append(tool_def)
        if probabilities:
            rest.sort(
                key=lambda tool_def: probabilities.get(_tool_name(tool_def) or "", -1.0),
                reverse=True,
            )
        ranked = ([chosen_def] if chosen_def is not None else []) + rest
        return ranked[: max(1, self._top_k)]
