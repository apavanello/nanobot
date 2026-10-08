"""Tests for the decisions JSONL store."""

from __future__ import annotations

from nanobot.decisions.store import (
    DecisionRecord,
    DecisionsStore,
    safe_session_filename,
)


def _record(**overrides: object) -> DecisionRecord:
    values: dict[str, object] = {
        "ts_ms": 1_000,
        "session_key": "chat-key",
        "scope": "chat",
        "mode": "shadow",
        "provider": "openrouter",
        "model": "typesafe/jev-1.13",
        "chosen": "list_files",
        "confidence": 0.9,
        "llm_tool_names": ["list_files"],
    }
    values.update(overrides)
    return DecisionRecord(**values)


class TestSafeFilename:
    def test_sanitizes_separators(self) -> None:
        assert safe_session_filename("telegram:12345:chat") == "telegram_12345_chat.jsonl"

    def test_none_and_empty_fallback(self) -> None:
        assert safe_session_filename(None) == "unknown.jsonl"
        assert safe_session_filename("///") == "unknown.jsonl"

    def test_long_keys_truncated(self) -> None:
        assert len(safe_session_filename("a" * 500)) <= 80 + len(".jsonl")


class TestStore:
    def test_record_and_iter_roundtrip(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record())
        store.record(_record(ts_ms=2_000, chosen="exec"))
        records = list(store.iter_records())
        assert [r.chosen for r in records] == ["list_files", "exec"]
        assert records[0].confidence == 0.9
        assert records[0].llm_tool_names == ["list_files"]

    def test_sessions_are_separate_files(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(session_key="chat-a"))
        store.record(_record(session_key="chat-b", chosen="exec"))
        assert (tmp_path / "chat-a.jsonl").exists()
        assert (tmp_path / "chat-b.jsonl").exists()
        only_a = list(store.iter_records(session_key="chat-a"))
        assert len(only_a) == 1 and only_a[0].session_key == "chat-a"

    def test_since_ms_filters_old_records(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(ts_ms=100))
        store.record(_record(ts_ms=200))
        records = list(store.iter_records(since_ms=150))
        assert [r.ts_ms for r in records] == [200]

    def test_malformed_lines_skipped(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record())
        with (tmp_path / "chat-key.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("{not json}\n")
            fh.write('"just a string"\n')
        records = list(store.iter_records())
        assert len(records) == 1

    def test_missing_session_file_yields_nothing(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        assert list(store.iter_records(session_key="ghost")) == []

    def test_default_root_uses_decisions_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("nanobot.decisions.store.get_decisions_dir", lambda: tmp_path)
        store = DecisionsStore()
        store.record(_record())
        assert (tmp_path / "chat-key.jsonl").exists()
