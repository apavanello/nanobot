"""Factory resolving the configured decision provider from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from nanobot.decisions.registry import (
    DEFAULT_DECISION_PROVIDER,
    decision_provider_names,
    resolve_decision_provider,
)

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.decisions.base import DecisionProvider


def make_decision_provider(config: Config) -> DecisionProvider | None:
    """Build the decision provider described by ``config.decisions``.

    Returns ``None`` when the layer is disabled or the provider name is not
    resolvable (warn + fail open, mirroring the transcription factory). A
    missing API key does not disable the provider: the adapter reports it as
    a per-call error so the agent keeps running and the gap stays visible.
    """
    decisions = config.decisions
    if not decisions.enabled:
        return None

    name = (decisions.provider or DEFAULT_DECISION_PROVIDER).strip().lower()
    spec = resolve_decision_provider(name)
    if spec is None:
        logger.warning(
            "Unknown decisions provider {!r}; falling back to {!r} (available: {})",
            name,
            DEFAULT_DECISION_PROVIDER,
            ", ".join(decision_provider_names()),
        )
        spec = resolve_decision_provider(DEFAULT_DECISION_PROVIDER)
        if spec is None:  # pragma: no cover - registry always defines the default
            return None

    provider_config = getattr(config.providers, spec.name, None)
    api_key = getattr(provider_config, "api_key", None) if provider_config is not None else None
    api_base = getattr(provider_config, "api_base", None) if provider_config is not None else None
    model = decisions.model or spec.default_model
    if decisions.model is None:
        logger.info(
            "Decisions layer: provider {}, model {} (pin a canonical slug in decisions.model to freeze probabilities)",
            spec.name,
            model,
        )
    adapter = spec.load_adapter()
    provider = adapter(api_key=api_key, api_base=api_base, model=model)
    return provider  # type: ignore[return-value]
