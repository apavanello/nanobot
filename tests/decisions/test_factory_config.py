"""Tests for the decision factory and DecisionsConfig schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, DecisionsConfig
from nanobot.decisions.factory import make_decision_provider
from nanobot.decisions.openrouter_provider import OpenRouterDecisionProvider
from nanobot.decisions.registry import decision_provider_names, resolve_decision_provider
from nanobot.decisions.typesafe_provider import TypeSafeDecisionProvider


class TestDecisionsConfig:
    def test_defaults_are_inert(self) -> None:
        cfg = DecisionsConfig()
        assert cfg.enabled is False
        assert cfg.mode == "shadow"
        assert cfg.threshold == 0.75
        assert cfg.scopes == ["chat"]

    def test_config_root_has_decisions_section(self) -> None:
        config = Config()
        assert config.decisions.enabled is False

    def test_camelcase_roundtrip(self) -> None:
        config = Config.model_validate(
            {"decisions": {"enabled": True, "provider": "openrouter", "topK": 9, "minTools": 3}}
        )
        assert config.decisions.enabled is True
        assert config.decisions.top_k == 9
        assert config.decisions.min_tools == 3
        dumped = config.model_dump(mode="json", by_alias=True)
        assert dumped["decisions"]["topK"] == 9

    def test_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DecisionsConfig(threshold=1.5)
        with pytest.raises(ValidationError):
            DecisionsConfig(mode="aggressive")

    def test_invalid_scope_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecisionsConfig(scopes=["dream"])


class TestRegistry:
    def test_known_providers(self) -> None:
        assert set(decision_provider_names()) == {"typesafe", "openrouter"}
        assert resolve_decision_provider("TypeSafe").name == "typesafe"
        assert resolve_decision_provider(None) is None
        assert resolve_decision_provider("nope") is None

    def test_adapters_load(self) -> None:
        for name in decision_provider_names():
            spec = resolve_decision_provider(name)
            assert spec.load_adapter() is not None


class TestFactory:
    def test_disabled_returns_none(self) -> None:
        config = Config()
        config.decisions.enabled = False
        assert make_decision_provider(config) is None

    def test_default_provider_is_openrouter(self) -> None:
        config = Config()
        config.decisions.enabled = True
        provider = make_decision_provider(config)
        assert isinstance(provider, OpenRouterDecisionProvider)
        assert provider.model == "typesafe/jev-1.13"

    def test_typesafe_provider_with_credentials(self) -> None:
        config = Config()
        config.decisions.enabled = True
        config.decisions.provider = "typesafe"
        config.decisions.model = "jev-latest"
        config.providers.typesafe.api_key = "cfg-key"
        config.providers.typesafe.api_base = "https://ts.example"
        provider = make_decision_provider(config)
        assert isinstance(provider, TypeSafeDecisionProvider)
        assert provider.api_key == "cfg-key"
        assert provider.api_url == "https://ts.example/v1/systemone"
        assert provider.model == "jev-latest"

    def test_unknown_provider_warns_and_falls_back(self) -> None:
        config = Config()
        config.decisions.enabled = True
        config.decisions.provider = "nonexistent"
        provider = make_decision_provider(config)
        assert isinstance(provider, OpenRouterDecisionProvider)

    def test_openrouter_reuses_provider_credentials(self) -> None:
        config = Config()
        config.decisions.enabled = True
        config.providers.openrouter.api_key = "sk-or-test"
        provider = make_decision_provider(config)
        assert isinstance(provider, OpenRouterDecisionProvider)
        assert provider.api_key == "sk-or-test"
