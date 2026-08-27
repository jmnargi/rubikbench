"""Results screen: aggregates, per-solve table, detail tabs, export."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, RichLog, TabbedContent, TabPane

from ..benchmark import BenchmarkResult, export_jsonl
from ..scramble import scramble_to_string


class ResultsScreen(Screen):
    CSS = """
    #agg-row { height: 3; padding: 0 1; }
    .agg { width: 1fr; content-align: center middle; }
    #body { height: 1fr; padding: 0 1; }
    #table { height: 1fr; border: round $panel; }
    #detail { width: 46; height: 1fr; border: round $panel; }
    #detail ScrollableContainer { height: 1fr; }
    #buttons { height: auto; padding: 1; }
    """

    def __init__(self, result: BenchmarkResult) -> None:
        super().__init__()
        self.result = result
        self.selected: int | None = None

    def compose(self) -> ComposeResult:
        agg = self.result.aggregates()
        with Horizontal(id="agg-row"):
            yield Label(f"solved: {agg.get('solve_rate', '—')}", id="agg-rate", classes="agg")
            yield Label(f"avg score: {agg.get('avg_score', '—')}", id="agg-score", classes="agg")
            yield Label(f"avg moves: {agg.get('avg_moves', '—')}", id="agg-moves", classes="agg")
            yield Label(f"avg turns: {agg.get('avg_turns', '—')}", id="agg-turns", classes="agg")
            yield Label(f"avg tools: {agg.get('avg_tool_calls', '—')}", id="agg-tools", classes="agg")
            yield Label(f"avg time: {agg.get('avg_time', '—')}s", id="agg-time", classes="agg")
        with Horizontal(id="body"):
            yield DataTable(id="table")
            with TabbedContent(id="detail"):
                with TabPane("Summary", id="tab-summary"):
                    yield RichLog(id="detail-summary", highlight=True, markup=True, wrap=True)
                with TabPane("Moves", id="tab-moves"):
                    yield RichLog(id="detail-moves", markup=True, wrap=True)
                with TabPane("Transcript", id="tab-transcript"):
                    yield RichLog(id="detail-transcript", markup=True, wrap=True)
        with Horizontal(id="buttons"):
            yield Button("Export JSONL", id="export-btn")
            yield Button("Replay", id="replay-btn")
            yield Button("Run again", id="rerun-btn", variant="primary")
    @on(Button.Pressed, "#replay-btn")
    def _on_replay(self) -> None:
        self.app.show_replay(self.result, self.selected or 0)  # type: ignore[attr-defined]


    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.add_columns("#", "Solved", "Turns", "Tools", "Text", "Moves", "Par", "Time (s)", "Score")
        table.cursor_type = "row"
        for s in self.result.solves:
            table.add_row(
                str(s.index + 1),
                "✓" if s.solved else ("✗" if s.error else "·"),
                str(s.turns),
                str(s.tool_calls),
                str(s.text_actions),
                str(s.total_moves),
                str(s.par),
                f"{s.elapsed:.1f}",
                f"{s.score:.0f}",
                key=s.index,
            )
        if self.result.solves:
            self.set_timer(0.05, lambda: self._show_detail(0))
        else:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "0")

    def _richlog(self, id_: str) -> RichLog | None:
        try:
            return self.query_one(f"#{id_}", RichLog)
        except Exception:  # noqa: BLE001 - TabPane content may not be mounted yet
            return None

    def _show_detail(self, row_index: int) -> None:
        if not 0 <= row_index < len(self.result.solves):
            return
        solve = self.result.solves[row_index]
        self.selected = solve.index

        summary = self._richlog("detail-summary")
        if summary is not None:
            summary.clear()
            summary.write(f"[bold]{'SOLVED' if solve.solved else 'UNSOLVED'}[/bold] " + (f"[dim]({solve.error})[/dim]" if solve.error else ""))
            summary.write(f"score: [bold]{solve.score}[/bold] / 1000")
            if solve.finish_reasons:
                finish = ", ".join(solve.finish_reasons)
                if solve.truncated:
                    finish += " (truncated)"
                summary.write(f"finish: {finish}")
            summary.write(
                f"actions: {solve.tool_calls} tool(s), {solve.text_actions} text action(s)"
            )
            summary.write(
                f"tokens: reasoning {solve.reasoning_tokens} (est {solve.estimated_reasoning_tokens}), "
                f"cached {solve.cached_tokens} (est cacheable {solve.estimated_cacheable_tokens})"
            )

        moves = self._richlog("detail-moves")
        if moves is not None:
            moves.clear()
            moves.write(f"[dim]scramble ({len(solve.scramble)}):[/dim] {scramble_to_string(solve.scramble)}")
            moves.write(f"[dim]solution par: {solve.par} moves[/dim]")
            applied = []
            for entry in solve.transcript:
                if entry.get("role") == "tool" and entry.get("name") == "apply_moves":
                    moves_text = (entry.get("arguments") or {}).get("moves", "")
                    if moves_text:
                        applied.append(moves_text)
            if applied:
                moves.write(f"[bold]{len(applied)} apply_moves call(s):[/bold]")
                for i, a in enumerate(applied, 1):
                    moves.write(f"  {i}. {a}")
            else:
                moves.write("[dim]no apply_moves calls; check the transcript for text moves[/dim]")

        transcript = self._richlog("detail-transcript")
        if transcript is None:
            return
        transcript.clear()
        for entry in solve.transcript:
            turn = entry.get("turn", "?")
            role = entry.get("role")
            if role == "assistant":
                calls = ", ".join(t["name"] for t in entry.get("tool_calls") or [])
                content = (entry.get("content") or "").strip().replace("\n", " ")
                line = f"[yellow][t{turn}][/yellow] assistant"
                if calls:
                    line += f" -> {calls}"
                if content:
                    line += f": {content[:120]}"
                transcript.write(line)
            elif role == "tool":
                first = (entry.get("content") or "").splitlines()
                head = first[0][:110] if first else ""
                transcript.write(f"[cyan][t{turn}][/cyan] tool {entry.get('name')}: {head}")

    # ------------------------------------------------------------------ handlers
    @on(DataTable.RowSelected, "#table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None:
            return
        self._show_detail(event.cursor_row)

    @on(Button.Pressed, "#export-btn")
    def _on_export(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path("rubikbench_results") / f"bench_{stamp}.jsonl"
        try:
            export_jsonl(self.result, out)
        except OSError as exc:
            self.notify(f"Export failed: {exc}", severity="error", timeout=6)
            return
        self.notify(f"Exported {len(self.result.solves)} solve(s) to {out}")

    @on(Button.Pressed, "#rerun-btn")
    def _on_rerun(self) -> None:
        self.app.start_run()  # type: ignore[attr-defined]
