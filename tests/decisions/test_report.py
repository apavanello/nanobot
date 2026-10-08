"""Tests for decision report aggregations."""

from __future__ import annotations

import time

from nanobot.decisions.report import build_report
from nanobot.decisions.store import DecisionRecord, DecisionsStore

_NOW_MS = time.time_ns() // 1_000_000
_DAY_MS = 86_400_000


def _record(**overrides: object) -> DecisionRecord:
    values: dict[str, object] = {
        "ts_ms": _NOW_MS,
        "session_key": "s1",
        "scope": "chat",
        "mode": "shadow",
        "provider": "openrouter",
        "model": "typesafe/jev-1.13",
    }
    values.update(overrides)
    return DecisionRecord(**values)


class TestBuildReport:
    def test_empty_store(self, tmp_path) -> None:
        report = build_report(DecisionsStore(root=tmp_path))
        assert report.records == 0
        assert report.agreement_rate is None
        assert report.total_tokens_saved_estimate == 0

    def test_agreement_counting(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(chosen="tool_a", llm_tool_names=["tool_a"], llm_finish_reason="tool_calls"))
        store.record(_record(chosen="tool_b", llm_tool_names=["tool_a"], llm_finish_reason="tool_calls"))
        # decision said none, LLM called nothing -> agreement
        store.record(_record(chosen="none", llm_tool_names=None, llm_finish_reason="stop"))
        # decision said none, LLM called something -> disagreement
        store.record(_record(chosen="none", llm_tool_names=["tool_a"], llm_finish_reason="tool_calls"))
        report = build_report(store)
        assert report.agreement_total == 4
        assert report.agreement_matches == 2
        assert report.agreement_rate == 0.5

    def test_savings_potential_and_estimated(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        # shadow: eligible, LLM ran anyway -> potential
        store.record(
            _record(
                chosen="status",
                fastpath_eligible=True,
                llm_input_tokens=1000,
                llm_output_tokens=200,
                llm_finish_reason="stop",
            )
        )
        # shadow: not eligible, LLM ran (measured, not savings)
        store.record(
            _record(chosen="tool_x", llm_input_tokens=500, llm_output_tokens=100, llm_finish_reason="stop")
        )
        # fastpath executed: LLM skipped entirely
        store.record(_record(mode="fastpath", chosen="status", fastpath_eligible=True, fastpath_executed=True))
        report = build_report(store)
        assert report.potential_tokens_saved == 1200
        assert report.llm_rounds_measured == 2
        assert report.llm_tokens_measured == 1800
        # avg 900 per measured round * 1 executed-without-llm round
        assert report.estimated_tokens_saved == 900
        assert report.total_tokens_saved_estimate == 2100

    def test_dual_rounds_count_as_measured(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(
            _record(
                mode="dual",
                chosen="status",
                fastpath_eligible=True,
                fastpath_executed=True,
                llm_input_tokens=400,
                llm_output_tokens=100,
                llm_finish_reason="tool_calls",
                llm_tool_names=["tool_y"],
            )
        )
        report = build_report(store)
        # executed but LLM tokens present -> not "without llm", counted as potential
        assert report.estimated_tokens_saved == 0
        assert report.potential_tokens_saved == 500

    def test_decision_usage_and_costs(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(decision_latency_ms=100, decision_input_tokens=50, decision_output_tokens=5, decision_cost_usd=0.000002, confidence=0.9))
        store.record(_record(decision_latency_ms=200, decision_input_tokens=30, decision_output_tokens=3, decision_cost_usd=0.000001, confidence=0.7))
        report = build_report(store)
        assert report.decision_input_tokens == 80
        assert report.decision_output_tokens == 8
        assert report.decision_cost_usd == 0.000003
        assert report.avg_decision_latency_ms == 150.0
        assert report.avg_confidence == 0.8

    def test_skips_bucketed_by_prefix(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(skipped_reason="min_tools"))
        store.record(_record(skipped_reason="error:request failed"))
        store.record(_record(skipped_reason="error:parse error: x"))
        report = build_report(store)
        assert report.skipped_by_reason == {"min_tools": 1, "error": 2}
        assert report.decided == 0

    def test_days_filter_excludes_old_records(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(ts_ms=_NOW_MS - 10 * _DAY_MS, chosen="old_tool"))
        store.record(_record(chosen="new_tool"))
        assert build_report(store, days=5).records == 1
        assert build_report(store, days=30).records == 2

    def test_session_filter(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(session_key="a", chosen="t"))
        store.record(_record(session_key="b", chosen="t"))
        assert build_report(store, session_key="a").sessions == 1

    def test_chosen_counts_sorted_by_frequency(self, tmp_path) -> None:
        store = DecisionsStore(root=tmp_path)
        store.record(_record(chosen="a"))
        store.record(_record(chosen="b"))
        store.record(_record(chosen="b"))
        report = build_report(store)
        assert report.chosen_counts == {"a": 1, "b": 2}
