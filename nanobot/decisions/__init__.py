"""System One decision layer for tool selection.

Small typed decision models (TypeSafe Jev, the OpenRouter Decisions catalog)
choose tools and narrow the tool list the LLM sees, with a confidence-gated
fast path and shadow logging for calibration. See
``brainstorms/2026-10-07-nanobot-fork-jev-system-one.md`` for the design
record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import Config
    from nanobot.decisions.base import DecisionProvider


def make_decision_provider(config: Config) -> DecisionProvider | None:
    """Lazily import the factory to keep decision adapters out of startup."""
    from nanobot.decisions.factory import make_decision_provider as _make

    return _make(config)
