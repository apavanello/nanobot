"""Typer commands for the System One decision layer."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
decisions_app = typer.Typer(help="Inspect the decision layer (System One tool selection)")


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{100 * numerator / denominator:.1f}%"


def _activate_config(config: str | None, *, quiet: bool) -> None:
    """Bind the config path so the decisions store resolves to its data dir."""
    from nanobot.config.errors import ConfigLoadError
    from nanobot.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve(strict=False)
        set_config_path(config_path)
        if not quiet:
            console.print(f"[dim]Using config: {config_path}[/dim]")
    try:
        load_config(config_path)
    except ConfigLoadError as exc:
        from nanobot.cli.runtime_config import _print_config_error

        _print_config_error(exc)
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@decisions_app.command("report")
def report(
    days: int = typer.Option(30, "--days", "-d", help="Look back this many days"),
    session: str = typer.Option(None, "--session", "-s", help="Restrict to one session key"),
    config: str = typer.Option(None, "--config", "-c", help="Path to config.json"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON instead of a table"),
) -> None:
    """Aggregate decision-layer activity, agreement, and token savings."""
    from nanobot.decisions.report import build_report
    from nanobot.decisions.store import DecisionsStore

    _activate_config(config, quiet=as_json)
    store = DecisionsStore()
    result = build_report(store, days=days, session_key=session)

    if result.records == 0:
        console.print("[dim]No decision records found for the given range.[/dim]")
        raise typer.Exit(0)

    if as_json:
        import json

        console.print_json(json.dumps(result.to_dict()))
        raise typer.Exit(0)

    table = Table(title=f"Decision layer report (last {days} days)", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Records", str(result.records))
    table.add_row("Sessions", str(result.sessions))
    table.add_row("Decided (not skipped)", str(result.decided))
    table.add_row(
        "Skipped",
        ", ".join(f"{k}={v}" for k, v in sorted(result.skipped_by_reason.items())) or "none",
    )
    agreement = "—" if result.agreement_rate is None else f"{result.agreement_rate:.1%}"
    table.add_row(
        f"Decision↔LLM agreement ({result.agreement_matches}/{result.agreement_total})", agreement
    )
    table.add_row(
        "Fast-path eligible",
        f"{result.fastpath_eligible} ({_pct(result.fastpath_eligible, result.records)} of rounds)",
    )
    table.add_row("Fast-path executed", str(result.fastpath_executed))
    table.add_row("LLM rounds measured", str(result.llm_rounds_measured))
    table.add_row("LLM tokens on measured rounds", f"{result.llm_tokens_measured:,}")
    table.add_row("Potential tokens saved (measured)", f"{result.potential_tokens_saved:,}")
    table.add_row("Estimated tokens saved (executed)", f"{result.estimated_tokens_saved:,}")
    table.add_row("Decision tokens (in/out)", f"{result.decision_input_tokens:,} / {result.decision_output_tokens:,}")
    table.add_row("Decision cost (USD)", f"{result.decision_cost_usd:.6f}")
    latency = "—" if result.avg_decision_latency_ms is None else f"{result.avg_decision_latency_ms:.0f} ms"
    table.add_row("Avg decision latency", latency)
    confidence = "—" if result.avg_confidence is None else f"{result.avg_confidence:.3f}"
    table.add_row("Avg confidence", confidence)
    console.print(table)

    if result.chosen_counts:
        tools = Table(title="Chosen tools", show_lines=False)
        tools.add_column("Tool", style="bold")
        tools.add_column("Times", justify="right")
        for name, count in sorted(result.chosen_counts.items(), key=lambda kv: -kv[1]):
            tools.add_row(name, str(count))
        console.print(tools)

    console.print(
        "[dim]Savings: 'potential' counts eligible rounds where the LLM still ran "
        "(shadow/filter/dual); 'estimated' values executed fast-path rounds at the "
        "measured average tokens per LLM round.[/dim]"
    )
