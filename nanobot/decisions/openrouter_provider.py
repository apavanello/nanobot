"""OpenRouter Decisions API adapter — the System One contract behind one key.

Calls ``POST {api_base}/api/alpha/decisions``. Model ids are vendor-prefixed
canonical slugs (e.g. ``typesafe/jev-1.13``) and must be pinned: aliases can
shift probabilities between builds, and thresholds calibrated on one build do
not transfer to another. The live catalog is
``GET /api/v1/models?output_modalities=decisions`` — never pick from memory.
Responses additionally carry ``usage.cost`` in USD.
"""

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

_DEFAULT_API_BASE = "https://openrouter.ai"
_DECISION_PATH = "/api/alpha/decisions"
_TIMEOUT_S = 30.0


class OpenRouterDecisionProvider:
    """Decisions endpoint client reusing the shared OpenRouter credential."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_name = "openrouter"
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.api_url = f"{(api_base or os.environ.get('OPENROUTER_BASE_URL') or _DEFAULT_API_BASE).rstrip('/')}{_DECISION_PATH}"
        self.model = model or "typesafe/jev-1.13"
        self._transport = transport
        logger.debug("OpenRouter decision endpoint: {}", self.api_url)

    async def decide(self, state: Any, questions: dict[str, Question]) -> DecisionResult:
        if not self.api_key:
            return DecisionResult(provider=self.provider_name, error="OPENROUTER_API_KEY not configured")
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
                provider_label="OpenRouter",
                api_key=self.api_key,
                body=dict(body),
            )
        latency_ms = int((time.monotonic() - started) * 1000)
        if payload is None:
            return DecisionResult(
                provider=self.provider_name, model=self.model, latency_ms=latency_ms, error="request failed"
            )
        try:
            result = parse_decision_response(payload, provider=self.provider_name)
        except ValueError as e:
            logger.error("OpenRouter decision response parse error: {}", e)
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
