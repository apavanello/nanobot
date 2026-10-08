"""TypeSafe (Jev) decision-model adapter — first-party System One endpoint."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from loguru import logger

from nanobot.decisions._http import post_decision_with_retry
from nanobot.decisions.types import (
    DecisionRequestBody,
    DecisionResult,
    Question,
    parse_decision_response,
)

_DEFAULT_API_BASE = "https://api.typesafe.ai"
_DECISION_PATH = "/v1/systemone"
_TIMEOUT_S = 30.0


class TypeSafeDecisionProvider:
    """Calls ``POST {api_base}/v1/systemone`` with Bearer auth.

    Model ids are bare (``jev-latest``); responses report the resolved build
    (e.g. ``jev-1.13.0``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_name = "typesafe"
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.api_url = f"{(api_base or os.environ.get('TYPESAFE_BASE_URL') or _DEFAULT_API_BASE).rstrip('/')}{_DECISION_PATH}"
        self.model = model or "jev-latest"
        self._transport = transport
        logger.debug("TypeSafe decision endpoint: {}", self.api_url)

    async def decide(self, state: Any, questions: dict[str, Question]) -> DecisionResult:
        if not self.api_key:
            return DecisionResult(provider=self.provider_name, error="TYPESAFE_API_KEY not configured")
        body: DecisionRequestBody = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, transport=self._transport) as client:
            payload = await post_decision_with_retry(
                client,
                self.api_url,
                provider_label="TypeSafe",
                api_key=self.api_key,
                body=dict(body),
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        if payload is None:
            return DecisionResult(provider=self.provider_name, model=self.model, latency_ms=latency_ms, error="request failed")
        try:
            result = parse_decision_response(payload, provider=self.provider_name)
        except ValueError as e:
            logger.error("TypeSafe decision response parse error: {}", e)
            return DecisionResult(
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                error=f"parse error: {e}",
            )
        return DecisionResult(
            provider=result.provider,
            model=result.model,
            answers=result.answers,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=latency_ms,
        )
