"""Registry for System One decision-model providers.

Adapter classes live in ``nanobot.decisions.typesafe_provider`` and
``nanobot.decisions.openrouter_provider``. This module is the app-level
source of truth for provider names, default models, credential environment
variables, API bases, and endpoint paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from nanobot.decisions.base import DecisionProvider


class DecisionProviderAdapter(Protocol):
    """Construction surface required of every decision adapter class."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class DecisionProviderSpec:
    name: str
    adapter: str
    default_model: str
    env_key: str
    default_api_base: str
    decision_path: str

    def load_adapter(self) -> type[DecisionProviderAdapter]:
        module_name, _, class_name = self.adapter.partition(":")
        if not module_name or not class_name:
            raise RuntimeError(f"Invalid decision adapter path: {self.adapter}")
        adapter = getattr(import_module(module_name), class_name)
        return adapter


DECISION_PROVIDERS: tuple[DecisionProviderSpec, ...] = (
    DecisionProviderSpec(
        name="typesafe",
        adapter="nanobot.decisions.typesafe_provider:TypeSafeDecisionProvider",
        default_model="jev-latest",
        env_key="TYPESAFE_API_KEY",
        default_api_base="https://api.typesafe.ai",
        decision_path="/v1/systemone",
    ),
    DecisionProviderSpec(
        name="openrouter",
        adapter="nanobot.decisions.openrouter_provider:OpenRouterDecisionProvider",
        default_model="typesafe/jev-1.13",
        env_key="OPENROUTER_API_KEY",
        default_api_base="https://openrouter.ai",
        decision_path="/api/alpha/decisions",
    ),
)

DEFAULT_DECISION_PROVIDER = "openrouter"

_BY_NAME = {spec.name: spec for spec in DECISION_PROVIDERS}


def decision_provider_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in DECISION_PROVIDERS)


def get_decision_provider(name: str) -> DecisionProviderSpec | None:
    return _BY_NAME.get(name)


def resolve_decision_provider(value: Any) -> DecisionProviderSpec | None:
    if not isinstance(value, str):
        return None
    return _BY_NAME.get(value.strip().lower())


def cast_decision_provider(adapter: DecisionProviderAdapter) -> DecisionProvider:
    """Narrow an adapter instance to the runtime protocol (structural match)."""
    return adapter  # type: ignore[return-value]
