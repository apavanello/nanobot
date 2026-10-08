"""Tests for System One decision wire-type parsing."""

from __future__ import annotations

import pytest

from nanobot.decisions.types import (
    DecisionParseError,
    DecisionResult,
    parse_decision_answer,
    parse_decision_response,
)


def _choice_answer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.88, "technical": 0.12},
        "confidence": 0.81,
    }
    payload.update(overrides)
    return payload


class TestParseChoiceAnswer:
    def test_full_answer(self) -> None:
        answer = parse_decision_answer(_choice_answer(), "answers[team]")
        assert answer.type == "choice"
        assert answer.choice == "billing"
        assert answer.probabilities == {"billing": 0.88, "technical": 0.12}
        assert answer.confidence == 0.81

    def test_missing_probabilities_stays_none(self) -> None:
        payload = _choice_answer()
        del payload["probabilities"]
        answer = parse_decision_answer(payload, "a")
        assert answer.probabilities is None
        assert answer.confidence == 0.81

    def test_missing_confidence_stays_none(self) -> None:
        payload = _choice_answer()
        del payload["confidence"]
        answer = parse_decision_answer(payload, "a")
        assert answer.confidence is None
        assert answer.probabilities is not None

    def test_missing_choice_is_parse_error(self) -> None:
        payload = _choice_answer()
        del payload["choice"]
        with pytest.raises(DecisionParseError, match="choice"):
            parse_decision_answer(payload, "a")

    def test_non_string_choice_is_parse_error(self) -> None:
        with pytest.raises(DecisionParseError):
            parse_decision_answer(_choice_answer(choice=42), "a")

    def test_unexpected_type_is_parse_error(self) -> None:
        with pytest.raises(DecisionParseError, match="choice", ):
            parse_decision_answer({"type": "prose", "text": "hi"}, "a")


class TestParseOtherAnswers:
    def test_noul(self) -> None:
        answer = parse_decision_answer({"type": "noul", "noul": 0.96}, "a")
        assert answer.type == "noul"
        assert answer.noul == 0.96
        assert answer.confidence is None

    def test_noul_missing_value_is_parse_error(self) -> None:
        with pytest.raises(DecisionParseError, match="noul"):
            parse_decision_answer({"type": "noul"}, "a")

    def test_score_with_legend(self) -> None:
        payload = {
            "type": "score",
            "score": 1.99,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0, "1": 0, "2": 1},
            "confidence": 0.99,
        }
        answer = parse_decision_answer(payload, "a")
        assert answer.score == 1.99
        assert answer.legend == {"0": "low", "1": "mid", "2": "high"}

    def test_non_object_payload_is_parse_error(self) -> None:
        with pytest.raises(DecisionParseError):
            parse_decision_answer("billing", "a")


class TestParseResponse:
    def test_full_response(self) -> None:
        payload = {
            "model": "jev-1.13.0",
            "answers": {"team": _choice_answer()},
            "usage": {"input_tokens": 318, "output_tokens": 34},
        }
        result = parse_decision_response(payload, provider="typesafe")
        assert result.ok
        assert result.model == "jev-1.13.0"
        assert result.answer_for("team") is not None
        assert result.answer_for("team").choice == "billing"  # type: ignore[union-attr]
        assert result.input_tokens == 318
        assert result.output_tokens == 34
        assert result.cost_usd is None

    def test_openrouter_usage_cost(self) -> None:
        payload = {
            "id": "gen-dec-1",
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "answers": {"is_bug": {"type": "noul", "noul": 0.96}},
            "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
        }
        result = parse_decision_response(payload, provider="openrouter")
        assert result.cost_usd == pytest.approx(0.000019992)
        assert result.answer_for("is_bug").noul == 0.96  # type: ignore[union-attr]

    def test_missing_answers_is_parse_error(self) -> None:
        with pytest.raises(DecisionParseError):
            parse_decision_response({"model": "jev"}, provider="typesafe")

    def test_one_bad_answer_fails_the_parse(self) -> None:
        payload = {
            "answers": {"good": _choice_answer(), "bad": {"type": "choice"}},
        }
        with pytest.raises(DecisionParseError):
            parse_decision_response(payload, provider="typesafe")

    def test_error_result_is_not_ok(self) -> None:
        result = DecisionResult(provider="typesafe", error="request failed")
        assert not result.ok
        assert result.answer_for("team") is None
