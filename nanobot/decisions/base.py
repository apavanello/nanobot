"""Runtime protocol implemented by decision-model adapters."""

from __future__ import annotations

from typing import Any, Protocol

from nanobot.decisions.types import DecisionResult, Question


class DecisionProvider(Protocol):
    """A System One decision model endpoint.

    Adapters are constructed with ``(api_key, api_base, model)`` credentials
    (each falling back to a provider-specific environment variable) and expose
    a single ``decide`` call. Operational failures return a ``DecisionResult``
    with ``error`` set rather than raising, so the agent turn always proceeds.
    """

    provider_name: str

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None: ...

    async def decide(self, state: Any, questions: dict[str, Question]) -> DecisionResult: ...
