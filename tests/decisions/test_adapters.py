"""Tests for decision adapters over a mocked HTTP transport."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nanobot.decisions.openrouter_provider import OpenRouterDecisionProvider
from nanobot.decisions.typesafe_provider import TypeSafeDecisionProvider

_QUESTION: dict[str, Any] = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments", "technical": "Bugs"},
    }
}


def _ok_payload() -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "technical": 0.1},
                "confidence": 0.8,
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


class TestTypeSafeAdapter:
    async def test_request_shape_and_parse(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_ok_payload())

        provider = TypeSafeDecisionProvider(
            api_key="ts-key", api_base="https://ts.example", model="jev-latest", transport=httpx.MockTransport(handler)
        )
        result = await provider.decide({"ticket": "broken checkout"}, _QUESTION)

        assert seen["url"] == "https://ts.example/v1/systemone"
        assert seen["auth"] == "Bearer ts-key"
        assert seen["body"]["model"] == "jev-latest"
        assert seen["body"]["questions"]["team"]["type"] == "choice"
        assert result.ok
        assert result.answer_for("team").choice == "billing"  # type: ignore[union-attr]
        assert result.input_tokens == 10
        assert result.latency_ms is not None

    async def test_missing_key_defers_without_call(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not be called")

        provider = TypeSafeDecisionProvider(api_key=None, transport=httpx.MockTransport(handler))
        result = await provider.decide("state", _QUESTION)
        assert not result.ok
        assert "TYPESAFE_API_KEY" in result.error

    async def test_401_fails_fast_without_retry(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, json={"error": "bad key"})

        provider = TypeSafeDecisionProvider(api_key="bad", transport=httpx.MockTransport(handler))
        result = await provider.decide("state", _QUESTION)
        assert calls["n"] == 1
        assert not result.ok

    async def test_429_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": "slow down"})
            return httpx.Response(200, json=_ok_payload())

        provider = TypeSafeDecisionProvider(api_key="k", transport=httpx.MockTransport(handler))
        with patch("nanobot.decisions._http.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await provider.decide("state", _QUESTION)
        assert sleep.await_count == 1
        assert calls["n"] == 2
        assert result.ok


class TestOpenRouterAdapter:
    async def test_request_shape_and_parse(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "gen-dec-1",
                    "model": "typesafe/jev-1.13-20260917",
                    "provider": "TypeSafe",
                    "answers": _ok_payload()["answers"],
                    "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002},
                },
            )

        provider = OpenRouterDecisionProvider(
            api_key="or-key", model="typesafe/jev-1.13", transport=httpx.MockTransport(handler)
        )
        result = await provider.decide({"ticket": "x"}, _QUESTION)

        assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
        assert seen["body"]["model"] == "typesafe/jev-1.13"
        assert result.cost_usd == pytest.approx(0.00002)
        assert result.model == "typesafe/jev-1.13-20260917"

    async def test_missing_key_defers(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not be called")

        provider = OpenRouterDecisionProvider(api_key=None, transport=httpx.MockTransport(handler))
        result = await provider.decide("state", _QUESTION)
        assert not result.ok
        assert "OPENROUTER_API_KEY" in result.error

    async def test_malformed_answers_parse_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"answers": {"team": {"type": "choice"}}})

        provider = OpenRouterDecisionProvider(api_key="k", transport=httpx.MockTransport(handler))
        result = await provider.decide("state", _QUESTION)
        assert not result.ok
        assert "parse error" in result.error
