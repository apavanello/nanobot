"""Append-only JSONL store for decision-layer records.

One file per session under ``<data_dir>/decisions/<safe-key>.jsonl``, written
fail-open: a store failure logs and never breaks the agent turn. The files
live outside the sessions directory on purpose — the session lister globs
``*.jsonl`` there and would ingest siblings as corrupted sessions.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

from loguru import logger

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SESSION_KEY_LEN = 80


def get_decisions_dir() -> Path:
    """Resolve the decisions data directory (patchable in tests)."""
    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("decisions")


def safe_session_filename(session_key: str | None) -> str:
    """Sanitize a session key into a stable, filesystem-safe stem."""
    key = (session_key or "unknown").strip() or "unknown"
    stem = _UNSAFE_FILENAME.sub("_", key)[:_MAX_SESSION_KEY_LEN].strip("._-") or "unknown"
    return f"{stem}.jsonl"


@dataclass(frozen=True)
class DecisionRecord:
    """One decision-layer observation for a single model round.

    The decision side captures the System One answer (choice, probabilities,
    confidence, usage). The ``llm_*`` side captures what the LLM actually did
    on the same round, so shadow logs directly calibrate the fast path.
    """

    ts_ms: int
    session_key: str
    scope: str
    mode: str
    provider: str
    model: str
    chosen: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    skipped_reason: str | None = None
    error: str | None = None
    decision_latency_ms: int | None = None
    decision_input_tokens: int | None = None
    decision_output_tokens: int | None = None
    decision_cost_usd: float | None = None
    llm_model: str | None = None
    llm_tool_names: list[str] | None = None
    llm_finish_reason: str | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_generation_ms: int | None = None
    fastpath_eligible: bool = False
    fastpath_executed: bool = False
    filtered_tool_count: int | None = None
    total_tool_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecisionRecord:
        known = {f.name for f in fields(cls)}
        values: dict[str, Any] = {k: v for k, v in payload.items() if k in known}
        return cls(**values)


class DecisionsStore:
    """Append-only JSONL store; process-safe enough for the single-process agent."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._lock = threading.Lock()

    def _resolve_root(self) -> Path:
        return self._root if self._root is not None else get_decisions_dir()

    def session_path(self, session_key: str | None) -> Path:
        return self._resolve_root() / safe_session_filename(session_key)

    def record(self, entry: DecisionRecord) -> None:
        """Append one record; logging-only failure (never breaks the turn)."""
        try:
            path = self.session_path(entry.session_key)
            line = json.dumps(entry.to_dict(), ensure_ascii=False, default=str)
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as e:  # pragma: no cover - fail-open by contract
            logger.warning("Decision record write failed: {}", e)

    def iter_records(
        self,
        *,
        session_key: str | None = None,
        since_ms: int | None = None,
    ) -> Iterator[DecisionRecord]:
        """Yield records across sessions (or one session), oldest first."""
        root = self._resolve_root()
        paths = (
            [self.session_path(session_key)]
            if session_key is not None
            else sorted(root.glob("*.jsonl"))
        )
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning("Skipping unreadable decision log {}: {}", path.name, e)
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    logger.warning("Skipping malformed decision log line in {}", path.name)
                    continue
                if not isinstance(payload, dict):
                    continue
                record = DecisionRecord.from_dict(cast("dict[str, Any]", payload))
                if since_ms is not None and record.ts_ms < since_ms:
                    continue
                yield record


def now_ms() -> int:
    return time.time_ns() // 1_000_000
