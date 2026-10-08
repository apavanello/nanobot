"""Wire types for System One decision models (TypeSafe Jev contract).

The TypeSafe ``/v1/systemone`` API is the de facto contract for decision
models: a request carries a ``state`` plus typed ``questions`` (choice /
score / noul), and the response carries one typed answer per question with
per-option probabilities and a confidence value. OpenRouter's Decisions API
(``/api/alpha/decisions``) speaks the same schema with vendor-prefixed model
ids and an extra ``usage.cost`` field.

``probabilities`` and ``confidence`` are optional on the wire (not every
decision model returns them). They are parsed as ``None`` when absent —
callers must treat ``None`` as "cannot act on this answer" (defer), never
as a silent default. A missing value field or an unexpected ``type`` is a
parse error, per the Decisions API guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Required, TypedDict, cast

QuestionType = Literal["choice", "score", "noul"]


class ChoiceQuestion(TypedDict):
    """Pick one option from a named set (up to 255 options)."""

    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str | None]


class ScoreQuestion(TypedDict):
    """Rate the state against an ordered rubric (2-10 levels, lowest first)."""

    type: Literal["score"]
    instructions: str
    criteria: list[str]


class NoulQuestion(TypedDict):
    """Yes/no judgment returning a single 0-1 probability."""

    type: Literal["noul"]
    instructions: str
    criteria: NotRequired[dict[str, str]]


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


class DecisionRequestBody(TypedDict, total=False):
    """Request body shared by both decision endpoints."""

    model: Required[str]
    state: Required[Any]
    questions: Required[dict[str, Question]]
    session_id: str
    user: str


@dataclass(frozen=True)
class DecisionAnswer:
    """One typed answer, normalized across providers.

    ``probabilities``/``confidence``/``legend`` stay ``None`` when the wire
    response omitted them; the semantic layer must defer instead of guessing.
    """

    type: QuestionType
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    legend: dict[str, str] | None = None


@dataclass(frozen=True)
class DecisionResult:
    """Normalized outcome of one decision call (fail-open).

    ``error`` is set instead of raising: an operational failure means the
    caller should defer to the LLM, not abort the agent turn.
    """

    provider: str
    model: str = ""
    answers: dict[str, DecisionAnswer] = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def answer_for(self, question_id: str) -> DecisionAnswer | None:
        return self.answers.get(question_id)


class DecisionParseError(ValueError):
    """The response body did not match the System One answer contract."""


def _parse_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionParseError(f"{label} must be a number, got {value!r}")
    return float(value)


def _parse_probabilities(value: Any, label: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DecisionParseError(f"{label} must be an object when present")
    entries = cast("dict[str, Any]", value)
    parsed: dict[str, float] = {}
    for option, prob in entries.items():
        parsed[option] = _parse_float(prob, f"{label}[{option!r}]")
    return parsed


def _parse_legend(value: Any, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DecisionParseError(f"{label} must be an object when present")
    entries = cast("dict[str, Any]", value)
    legend: dict[str, str] = {}
    for level, description in entries.items():
        if not isinstance(description, str):
            raise DecisionParseError(f"{label}[{level!r}] must be a string")
        legend[level] = description
    return legend


def parse_decision_answer(payload: Any, label: str) -> DecisionAnswer:
    """Parse one answer object, rejecting missing value fields and unknown types."""
    if not isinstance(payload, dict):
        raise DecisionParseError(f"{label} must be an object, got {type(payload).__name__}")
    data = cast("dict[str, Any]", payload)
    answer_type = data.get("type")
    if answer_type == "choice":
        choice = data.get("choice")
        if not isinstance(choice, str) or not choice:
            raise DecisionParseError(f"{label}.choice must be a non-empty string")
        return DecisionAnswer(
            type="choice",
            choice=choice,
            probabilities=_parse_probabilities(data.get("probabilities"), f"{label}.probabilities"),
            confidence=(
                None
                if data.get("confidence") is None
                else _parse_float(data["confidence"], f"{label}.confidence")
            ),
        )
    if answer_type == "score":
        return DecisionAnswer(
            type="score",
            score=_parse_float(data.get("score"), f"{label}.score"),
            probabilities=_parse_probabilities(data.get("probabilities"), f"{label}.probabilities"),
            confidence=(
                None
                if data.get("confidence") is None
                else _parse_float(data["confidence"], f"{label}.confidence")
            ),
            legend=_parse_legend(data.get("legend"), f"{label}.legend"),
        )
    if answer_type == "noul":
        return DecisionAnswer(
            type="noul",
            noul=_parse_float(data.get("noul"), f"{label}.noul"),
        )
    raise DecisionParseError(f"{label}.type must be 'choice', 'score' or 'noul', got {answer_type!r}")


def parse_decision_response(
    payload: Any,
    *,
    provider: str,
) -> DecisionResult:
    """Parse a full decision response body into a normalized result."""
    if not isinstance(payload, dict):
        raise DecisionParseError(f"response must be an object, got {type(payload).__name__}")
    data = cast("dict[str, Any]", payload)
    raw_answers = data.get("answers")
    if not isinstance(raw_answers, dict):
        raise DecisionParseError("response.answers must be an object")
    answer_entries = cast("dict[str, Any]", raw_answers)
    answers = {
        question_id: parse_decision_answer(answer, f"answers[{question_id!r}]")
        for question_id, answer in answer_entries.items()
    }
    usage = data.get("usage")
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    if isinstance(usage, dict):
        usage_data = cast("dict[str, Any]", usage)
        raw_input = usage_data.get("input_tokens")
        raw_output = usage_data.get("output_tokens")
        input_tokens = raw_input if isinstance(raw_input, int) and not isinstance(raw_input, bool) else None
        output_tokens = raw_output if isinstance(raw_output, int) and not isinstance(raw_output, bool) else None
        raw_cost = usage_data.get("cost")
        cost_usd = None if raw_cost is None else _parse_float(raw_cost, "usage.cost")
    model = data.get("model")
    return DecisionResult(
        provider=provider,
        model=model if isinstance(model, str) else "",
        answers=answers,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
