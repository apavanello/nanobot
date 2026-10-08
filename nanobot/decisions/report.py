"""Aggregations over decision records for the ``decisions report`` command.

Savings semantics:
- ``potential_tokens_saved`` — rounds where the fast path was eligible but the
  LLM round happened anyway (shadow / filter / dual). This is the directly
  measured "would have skipped" total.
- ``estimated_tokens_saved`` — fastpath rounds where the LLM was truly
  skipped, valued at the average tokens of the measured LLM rounds. This is
  an estimate: no LLM call exists to measure on those rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nanobot.decisions.store import DecisionRecord, DecisionsStore

_MS_PER_DAY = 86_400_000


@dataclass
class DecisionsReport:
    """Aggregated decision-layer activity (accumulated while scanning records)."""

    records: int = 0
    sessions: int = 0
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None
    by_mode: dict[str, int] = field(default_factory=dict)
    by_scope: dict[str, int] = field(default_factory=dict)
    decided: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    agreement_total: int = 0
    agreement_matches: int = 0
    fastpath_eligible: int = 0
    fastpath_executed: int = 0
    llm_rounds_measured: int = 0
    llm_tokens_measured: int = 0
    potential_tokens_saved: int = 0
    estimated_tokens_saved: int = 0
    decision_input_tokens: int = 0
    decision_output_tokens: int = 0
    decision_cost_usd: float = 0.0
    latency_samples: int = 0
    latency_total_ms: int = 0
    confidence_samples: int = 0
    confidence_total: float = 0.0
    chosen_counts: dict[str, int] = field(default_factory=dict)

    @property
    def agreement_rate(self) -> float | None:
        if self.agreement_total == 0:
            return None
        return self.agreement_matches / self.agreement_total

    @property
    def avg_decision_latency_ms(self) -> float | None:
        if self.latency_samples == 0:
            return None
        return self.latency_total_ms / self.latency_samples

    @property
    def avg_confidence(self) -> float | None:
        if self.confidence_samples == 0:
            return None
        return self.confidence_total / self.confidence_samples

    @property
    def total_tokens_saved_estimate(self) -> int:
        return self.potential_tokens_saved + self.estimated_tokens_saved

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "sessions": self.sessions,
            "by_mode": self.by_mode,
            "by_scope": self.by_scope,
            "decided": self.decided,
            "skipped_by_reason": self.skipped_by_reason,
            "agreement_rate": self.agreement_rate,
            "fastpath_eligible": self.fastpath_eligible,
            "fastpath_executed": self.fastpath_executed,
            "llm_rounds_measured": self.llm_rounds_measured,
            "llm_tokens_measured": self.llm_tokens_measured,
            "potential_tokens_saved": self.potential_tokens_saved,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "decision_input_tokens": self.decision_input_tokens,
            "decision_output_tokens": self.decision_output_tokens,
            "decision_cost_usd": self.decision_cost_usd,
            "avg_decision_latency_ms": self.avg_decision_latency_ms,
            "avg_confidence": self.avg_confidence,
            "chosen_counts": self.chosen_counts,
        }


def _skip_reason_bucket(reason: str) -> str:
    return reason.split(":", 1)[0]


def _llm_tokens(record: DecisionRecord) -> int:
    tokens = 0
    if record.llm_input_tokens is not None:
        tokens += record.llm_input_tokens
    if record.llm_output_tokens is not None:
        tokens += record.llm_output_tokens
    return tokens


def build_report(
    store: DecisionsStore,
    *,
    days: int | None = None,
    session_key: str | None = None,
) -> DecisionsReport:
    """Aggregate decision records into a report."""
    since_ms = None if days is None else _now_ms() - days * _MS_PER_DAY
    report = DecisionsReport()
    sessions: set[str] = set()
    executed_without_llm = 0

    for record in store.iter_records(session_key=session_key, since_ms=since_ms):
        report.records += 1
        sessions.add(record.session_key)
        if report.first_ts_ms is None or record.ts_ms < report.first_ts_ms:
            report.first_ts_ms = record.ts_ms
        if report.last_ts_ms is None or record.ts_ms > report.last_ts_ms:
            report.last_ts_ms = record.ts_ms
        report.by_mode[record.mode] = report.by_mode.get(record.mode, 0) + 1
        report.by_scope[record.scope] = report.by_scope.get(record.scope, 0) + 1

        if record.skipped_reason is not None:
            bucket = _skip_reason_bucket(record.skipped_reason)
            report.skipped_by_reason[bucket] = report.skipped_by_reason.get(bucket, 0) + 1
        elif record.chosen is not None:
            report.decided += 1
            report.chosen_counts[record.chosen] = report.chosen_counts.get(record.chosen, 0) + 1

        if _decision_is_agreement_pair(record):
            report.agreement_total += 1
            if _agrees(record):
                report.agreement_matches += 1

        if record.fastpath_eligible:
            report.fastpath_eligible += 1
        if record.fastpath_executed:
            report.fastpath_executed += 1

        tokens = _llm_tokens(record)
        if tokens > 0:
            report.llm_rounds_measured += 1
            report.llm_tokens_measured += tokens
        if record.fastpath_eligible and tokens > 0:
            report.potential_tokens_saved += tokens
        if record.fastpath_executed and tokens == 0:
            executed_without_llm += 1

        if record.decision_input_tokens is not None:
            report.decision_input_tokens += record.decision_input_tokens
        if record.decision_output_tokens is not None:
            report.decision_output_tokens += record.decision_output_tokens
        if record.decision_cost_usd is not None:
            report.decision_cost_usd += record.decision_cost_usd
        if record.decision_latency_ms is not None:
            report.latency_samples += 1
            report.latency_total_ms += record.decision_latency_ms
        if record.confidence is not None:
            report.confidence_samples += 1
            report.confidence_total += record.confidence

    report.sessions = len(sessions)
    if report.llm_rounds_measured > 0 and executed_without_llm > 0:
        avg = report.llm_tokens_measured / report.llm_rounds_measured
        report.estimated_tokens_saved = int(avg * executed_without_llm)
    return report


def _decision_is_agreement_pair(record: DecisionRecord) -> bool:
    """Both sides observed: a decision answer and an LLM response on the same round."""
    return record.chosen is not None and record.llm_finish_reason is not None


def _agrees(record: DecisionRecord) -> bool:
    assert record.chosen is not None
    if record.chosen == "none":
        return record.llm_tool_names is None
    return bool(record.llm_tool_names) and record.chosen in (record.llm_tool_names or [])


def _now_ms() -> int:
    import time

    return time.time_ns() // 1_000_000
