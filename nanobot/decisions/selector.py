"""Tool-selection decision logic: state building, gating, and result policy.

v1.1 adds bounded extraction (after the jev-gateway design): the decision
model picks the tool, cross-checks that a tool is needed at all, and —
when every argument of the chosen tool has a closed set of allowed values
(const, enum, boolean) — fills those arguments in the same call, so the
fast path covers far more than zero-argument tools. Free-form arguments
(paths, commands, text) stay with the LLM, which then sees the pinned tool
only and acts as an argument filler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from loguru import logger

from nanobot.decisions.base import DecisionProvider
from nanobot.decisions.types import (
    ChoiceQuestion,
    DecisionAnswer,
    DecisionResult,
    NoulQuestion,
    Question,
)

DecisionScope = Literal["chat", "subagent", "cron"]

_NONE_OPTION = "none"
_MAX_CHOICE_OPTIONS = 255
_RECENT_MESSAGES = 12
_MESSAGE_TEXT_LIMIT = 400
_LAST_USER_LIMIT = 2000
_DESCRIPTION_LIMIT = 300
_PARAM_DESCRIPTION_LIMIT = 300

# Cross-check agreement band (jev-gateway values): choosing a tool requires
# the independent yes/no judgment to lean the same way, and vice versa.
_NEEDS_TOOL_MIN = 0.3
_NEEDS_TOOL_MAX = 0.7

# Cap on speculative argument questions fanned out in one decision call;
# optional parameters cost 2 (value + "is a value stated?"), required cost 1.
_MAX_ARG_QUESTIONS = 96

_TOOL_QUESTION = "tool_selection"
_NEEDS_TOOL_QUESTION = "needs_tool"


def arg_question_id(tool_index: int, param: str) -> str:
    return f"arg:{tool_index}:{param}"


def stated_question_id(tool_index: int, param: str) -> str:
    return f"stated:{tool_index}:{param}"


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


@dataclass
class ClosedParam:
    """A parameter whose allowed values form a finite, enumerable set."""

    name: str
    required: bool
    kind: Literal["const", "boolean", "enum"]
    value: Any = None  # const value
    values: dict[str, Any] = field(default_factory=dict)  # enum label -> wire value
    description: str | None = None


@dataclass
class ToolPlan:
    """Per-tool argument analysis.

    ``closed_params`` is ``None`` when at least one parameter is open-ended
    (free text, numbers, dates): the tool can be pinned for the LLM to fill,
    but never answered directly. An empty list means the tool takes no
    fillable arguments at all.
    """

    name: str | None
    closed_params: list[ClosedParam] | None = None


def _closed_param(name: str, schema: dict[str, Any], required: bool) -> ClosedParam | None:
    if "const" in schema:
        return ClosedParam(name=name, required=required, kind="const", value=schema["const"])
    raw_enum = schema.get("enum")
    if isinstance(raw_enum, list) and raw_enum:
        enum = cast("list[Any]", raw_enum)
        if len(enum) == 1:
            return ClosedParam(name=name, required=required, kind="const", value=enum[0])
        values: dict[str, Any] = {}
        for value in enum:
            if value is not None and isinstance(value, (dict, list)):
                return None
            label = str(value)
            if label in values:
                return None  # colliding labels ("1" vs 1) cannot be mapped back
            values[label] = value
        if len(values) > _MAX_CHOICE_OPTIONS:
            return None
        description = schema.get("description")
        return ClosedParam(
            name=name,
            required=required,
            kind="enum",
            values=values,
            description=description if isinstance(description, str) else None,
        )
    if schema.get("type") == "boolean":
        description = schema.get("description")
        return ClosedParam(
            name=name,
            required=required,
            kind="boolean",
            description=description if isinstance(description, str) else None,
        )
    return None


def plan_tool(tool_def: dict[str, Any]) -> ToolPlan:
    """Classify a tool's parameters for bounded extraction."""
    name = _tool_name(tool_def)
    function = tool_def.get("function")
    if name is None or not isinstance(function, dict):
        return ToolPlan(name=None, closed_params=None)
    entry = cast("dict[str, Any]", function)
    parameters = entry.get("parameters")
    if parameters is None:
        return ToolPlan(name=name, closed_params=[])
    if not isinstance(parameters, dict):
        return ToolPlan(name=None, closed_params=None)
    schema = cast("dict[str, Any]", parameters)
    schema_type = schema.get("type")
    if schema_type is not None and schema_type != "object":
        return ToolPlan(name=name, closed_params=None)
    raw_required = schema.get("required")
    required: set[str] = (
        {item for item in cast("list[Any]", raw_required) if isinstance(item, str)}
        if isinstance(raw_required, list)
        else set()
    )
    raw_properties = schema.get("properties")
    properties: dict[str, Any] = (
        cast("dict[str, Any]", raw_properties) if isinstance(raw_properties, dict) else {}
    )
    closed: list[ClosedParam] = []
    for param_name, raw_schema in properties.items():
        if not isinstance(raw_schema, dict):
            return ToolPlan(name=name, closed_params=None)
        param = _closed_param(param_name, cast("dict[str, Any]", raw_schema), param_name in required)
        if param is None:
            return ToolPlan(name=name, closed_params=None)
        closed.append(param)
    if required - set(properties):
        # A required parameter missing from ``properties`` says nothing safe
        # about the call; the LLM must handle the tool.
        return ToolPlan(name=name, closed_params=None)
    return ToolPlan(name=name, closed_params=closed)


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


def build_questions(
    tool_defs: list[dict[str, Any]],
    *,
    with_args: bool,
) -> tuple[dict[str, Question], list[ToolPlan]]:
    """Build the per-round question set: tool choice, needs-tool cross-check,
    and speculative argument questions for every fully closed tool.

    Extra questions barely change decision latency, so argument questions are
    fanned out for all closed tools at once and the relevant answers are picked
    afterwards. Tools whose questions would exceed the fan-out cap keep their
    tool plan open (pinnable, never direct).
    """
    plans = [plan_tool(tool_def) for tool_def in tool_defs]
    questions: dict[str, Question] = {
        _TOOL_QUESTION: build_choice_question(tool_catalog(tool_defs)),
        _NEEDS_TOOL_QUESTION: NoulQuestion(
            type="noul",
            instructions=(
                "Does the assistant need to call one of its tools now, rather "
                "than reply to the user in plain text?"
            ),
        ),
    }
    if not with_args:
        return questions, plans

    arg_questions = 0
    for index, plan in enumerate(plans):
        if plan.closed_params is None:
            continue
        asked = [param for param in plan.closed_params if param.kind != "const"]
        cost = sum(1 if param.required else 2 for param in asked)
        if arg_questions + cost > _MAX_ARG_QUESTIONS:
            plan.closed_params = None
            continue
        arg_questions += cost
        for param in asked:
            about = f'the "{param.name}" argument of the tool "{plan.name}"'
            if param.description:
                about = f"{about} ({param.description[:_PARAM_DESCRIPTION_LIMIT]})"
            if param.kind == "boolean":
                questions[arg_question_id(index, param.name)] = NoulQuestion(
                    type="noul",
                    instructions=(
                        f'If the assistant calls "{plan.name}" now, should {about} be true?'
                    ),
                )
            else:
                questions[arg_question_id(index, param.name)] = ChoiceQuestion(
                    type="choice",
                    instructions=(
                        f"If the assistant calls \"{plan.name}\" now, what value should {about} have?"
                    ),
                    criteria={label: None for label in param.values},
                )
            if not param.required:
                questions[stated_question_id(index, param.name)] = NoulQuestion(
                    type="noul",
                    instructions=f"Does the conversation state or clearly imply a value for {about}?",
                )
    return questions, plans


@dataclass(frozen=True)
class ArgsResolution:
    """Outcome of bounded extraction for one tool.

    ``args`` is ``None`` on failure, with ``deferred_reason`` saying why;
    on success ``certainty`` is the weakest judgment across all parameters.
    """

    args: dict[str, Any] | None = None
    certainty: float | None = None
    deferred_reason: str | None = None


def resolve_args(
    plan: ToolPlan,
    tool_index: int,
    answers: dict[str, DecisionAnswer],
    min_certainty: float,
) -> ArgsResolution:
    """Fill every closed-set argument from the decision answers.

    Optional parameters are only included when the conversation states a
    value for them (the ``stated`` question); they are omitted otherwise,
    never invented. The weakest judgment bounds the whole call.
    """
    args: dict[str, Any] = {}
    certainty = 1.0
    for param in plan.closed_params or []:
        if param.kind == "const":
            if param.required:
                args[param.name] = param.value
            continue
        if not param.required:
            stated = answers.get(stated_question_id(tool_index, param.name))
            if stated is None or stated.type != "noul" or stated.noul is None:
                return ArgsResolution(deferred_reason="args_unresolved")
            certainty = min(certainty, max(stated.noul, 1.0 - stated.noul))
            if stated.noul < 0.5:
                continue
        answer = answers.get(arg_question_id(tool_index, param.name))
        if param.kind == "boolean":
            if answer is None or answer.type != "noul" or answer.noul is None:
                return ArgsResolution(deferred_reason="args_unresolved")
            args[param.name] = answer.noul >= 0.5
            certainty = min(certainty, max(answer.noul, 1.0 - answer.noul))
        elif (
            answer is None
            or answer.type != "choice"
            or answer.choice is None
            or answer.confidence is None
            or answer.choice not in param.values
        ):
            # A choice answer without confidence cannot be scored: defer.
            return ArgsResolution(deferred_reason="args_unresolved")
        else:
            args[param.name] = param.values[answer.choice]
            certainty = min(certainty, answer.confidence)
    if certainty < min_certainty:
        return ArgsResolution(certainty=certainty, deferred_reason="args_below_threshold")
    return ArgsResolution(args=args, certainty=certainty)


@dataclass(frozen=True)
class ToolSelection:
    """Outcome of one selection round, before any LLM call.

    ``skipped_reason`` covers pre-call gates (scope, tool count, provider
    failure); ``deferred_reason`` covers post-answer refusals to act (low
    confidence, cross-check disagreement, unresolved arguments). Both being
    ``None`` means the round's policy was applied.
    """

    skipped_reason: str | None
    scope: DecisionScope | None = None
    decision: DecisionResult | None = None
    chosen: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    filtered_tools: list[dict[str, Any]] | None = field(default=None, repr=False)
    fastpath_eligible: bool = False
    fastpath_tool: str | None = None
    fastpath_args: dict[str, Any] | None = None
    args_certainty: float | None = None
    needs_tool: float | None = None
    deferred_reason: str | None = None
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
        arg_threshold: float = 0.8,
        filter_strategy: str = "pin",
        extract_args: bool = True,
    ) -> None:
        self._provider = provider
        self._mode = mode
        self._scopes = frozenset(scopes)
        self._threshold = threshold
        self._top_k = top_k
        self._min_tools = min_tools
        self._arg_threshold = arg_threshold
        self._filter_strategy = filter_strategy
        self._extract_args = extract_args

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
        questions, plans = build_questions(tool_defs, with_args=self._extract_args)
        result = await self._provider.decide(state, questions)
        if not result.ok:
            return ToolSelection(
                skipped_reason=f"error:{result.error}", scope=scope, decision=result, total_tool_count=total
            )
        answer = result.answer_for(_TOOL_QUESTION)
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

        wants_tool = chosen != _NONE_OPTION
        needs_answer = result.answer_for(_NEEDS_TOOL_QUESTION)
        needs_tool = (
            needs_answer.noul
            if needs_answer is not None and needs_answer.type == "noul" and needs_answer.noul is not None
            else None
        )

        deferred: str | None = None
        if answer.confidence is None or answer.confidence < self._threshold:
            deferred = "low_confidence"
        elif needs_tool is None:
            deferred = "needs_tool_missing"
        elif wants_tool and needs_tool < _NEEDS_TOOL_MIN:
            deferred = "answers_disagree"
        elif not wants_tool and needs_tool > _NEEDS_TOOL_MAX:
            deferred = "answers_disagree"

        filtered: list[dict[str, Any]] | None = None
        if deferred is None and wants_tool and self._mode in ("filter", "fastpath", "dual"):
            if self._filter_strategy == "pin":
                filtered = [by_name[chosen]]
            else:
                filtered = self._rank_tools(tool_defs, chosen, answer.probabilities)

        # Eligibility is mode-independent: shadow and filter logs must be able
        # to answer "would the fast path have fired?" for calibration reports.
        eligible = False
        fastpath_args: dict[str, Any] | None = None
        args_certainty: float | None = None
        if deferred is None and wants_tool:
            tool_index = next((i for i, plan in enumerate(plans) if plan.name == chosen), None)
            if tool_index is not None:
                plan = plans[tool_index]
                if plan.closed_params is None:
                    pass  # open-argument tool: pinnable, but the LLM must fill it
                elif not self._extract_args:
                    # No argument questions were asked; only fully determined
                    # (const-only or argument-less) tools can execute directly.
                    eligible = all(param.kind == "const" for param in plan.closed_params)
                    if eligible:
                        fastpath_args = {}
                else:
                    resolution = resolve_args(plan, tool_index, result.answers, self._arg_threshold)
                    if resolution.args is None:
                        deferred = resolution.deferred_reason
                    else:
                        eligible = True
                        fastpath_args = resolution.args
                        args_certainty = resolution.certainty
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
            fastpath_args=fastpath_args,
            args_certainty=args_certainty,
            needs_tool=needs_tool,
            deferred_reason=deferred,
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
