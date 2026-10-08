"""Shared HTTP helper for decision-model adapters.

Mirrors the retry discipline of ``nanobot.providers.transcription``: up to 3
retries with exponential backoff on transient failures (408/429/5xx and
connect/read/timeout errors). Other errors — including 401/403, whose cause
is a misconfigured key — return ``None`` immediately so the decision layer
can defer to the LLM instead of burning quota on a hopeless retry loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
from loguru import logger

_MAX_RETRIES = 3
_BACKOFF_S = (1.0, 2.0, 4.0)
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


async def post_decision_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider_label: str,
    api_key: str | None,
    body: dict[str, Any],
) -> dict[str, Any] | None:
    """POST a decision request, retrying transient failures.

    Returns the parsed JSON object, or ``None`` when the call failed after
    retries or failed fast on a non-retryable error.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await client.post(url, json=body, headers=headers)
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "{} decision call transient error (attempt {}/{}): {}",
                    provider_label,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    e,
                )
                await asyncio.sleep(_BACKOFF_S[attempt])
                continue
            logger.exception(
                "{} decision call error after {} attempts: {}",
                provider_label,
                _MAX_RETRIES + 1,
                e,
            )
            return None
        except Exception as e:
            logger.exception("{} decision call error: {}", provider_label, e)
            return None

        if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
            logger.warning(
                "{} decision call transient HTTP {} (attempt {}/{})",
                provider_label,
                response.status_code,
                attempt + 1,
                _MAX_RETRIES + 1,
            )
            await asyncio.sleep(_BACKOFF_S[attempt])
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            detail = response.text.strip().replace("\n", " ")[:500]
            logger.error(
                "{} decision call HTTP {}{}{}",
                provider_label,
                response.status_code,
                f" {response.reason_phrase}" if response.reason_phrase else "",
                f": {detail}" if detail else "",
            )
            return None
        except Exception as e:
            logger.exception("{} decision call error: {}", provider_label, e)
            return None

        try:
            payload = response.json()
        except Exception as e:
            logger.exception("{} decision call error: malformed response body: {}", provider_label, e)
            return None
        if not isinstance(payload, dict):
            logger.error(
                "{} decision call error: unexpected response shape: {!r}",
                provider_label,
                type(payload).__name__,
            )
            return None
        return cast("dict[str, Any]", payload)
    return None
