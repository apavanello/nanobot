"""Tests for the ``nanobot decisions report`` CLI command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.decisions.store import DecisionRecord, DecisionsStore

runner = CliRunner()


def _record(**overrides: object) -> DecisionRecord:
    values: dict[str, object] = {
        "ts_ms": __import__("time").time_ns() // 1_000_000,
        "session_key": "chat-key",
        "scope": "chat",
        "mode": "shadow",
        "provider": "openrouter",
        "model": "typesafe/jev-1.13",
        "chosen": "status",
        "confidence": 0.9,
        "llm_tool_names": ["status"],
        "llm_finish_reason": "tool_calls",
        "llm_input_tokens": 800,
        "llm_output_tokens": 120,
        "fastpath_eligible": True,
        "decision_cost_usd": 0.000002,
    }
    values.update(overrides)
    return DecisionRecord(**values)


def _setup(tmp_path, monkeypatch, records: list[DecisionRecord]) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("nanobot.decisions.store.get_decisions_dir", lambda: tmp_path / "decisions")
    store = DecisionsStore(root=tmp_path / "decisions")
    for record in records:
        store.record(record)


def test_report_table(tmp_path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch, [_record(), _record(chosen="exec")])
    result = runner.invoke(app, ["decisions", "report", "--config", str(tmp_path / "config.json")])
    assert result.exit_code == 0, result.output
    assert "Decision layer report" in result.output
    assert "Fast-path eligible" in result.output


def test_report_json(tmp_path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch, [_record()])
    result = runner.invoke(
        app, ["decisions", "report", "--config", str(tmp_path / "config.json"), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["records"] == 1
    assert payload["agreement_rate"] == 1.0
    assert payload["potential_tokens_saved"] == 920


def test_report_empty(tmp_path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch, [])
    result = runner.invoke(app, ["decisions", "report", "--config", str(tmp_path / "config.json")])
    assert result.exit_code == 0, result.output
    assert "No decision records found" in result.output


def test_report_session_filter(tmp_path, monkeypatch) -> None:
    _setup(tmp_path, monkeypatch, [_record(session_key="a"), _record(session_key="b")])
    result = runner.invoke(
        app,
        [
            "decisions",
            "report",
            "--config",
            str(tmp_path / "config.json"),
            "--session",
            "a",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["records"] == 1
    assert payload["sessions"] == 1
