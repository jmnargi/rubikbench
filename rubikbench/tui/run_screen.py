"""Live run screen: cube net, stats, transcript log, progress, abort."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from textual import on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, ProgressBar, RichLog, Static

from ..benchmark import BenchmarkResult, BenchmarkRunner
from ..config import BenchmarkConfig
from ..llm import OpenAICompatibleClient
from .messages import (
    BenchFinishedMsg,
    LogMsg,
    SolveDoneMsg,
    SolveStartedMsg,
    StateMsg,
    TurnMsg,
)
from .widgets import CubeNet

_TIME_FMT = "%H:%M:%S"


class RunScreen(Screen):
    CSS = """
    #run-top { height: 3; padding: 0 1; }
    #run-status { width: 1fr; content-align: left middle; }
    #run-progress { width: 30; margin: 0 1; }
    #abort-btn { width: 14; }
    #run-body { height: 1fr; padding: 0 1; }
    #left-col { width: 42; }
    .card { border: round $panel; padding: 0 1 1 1; margin: 0 0 1 0; }
    .card-title { text-style: bold; }
    #stats-grid { height: auto; grid-size: 2; grid-columns: 1fr 1fr; grid-rows: 1fr 1fr 1fr; }
    #cube-net { width: auto; height: auto; }
    #history { height: 8; }
    #log { height: 1fr; border: round $panel; }
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel = threading.Event()
        self.completed = 0
        self.last_solved_at = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="run-top"):
            yield Static("", id="run-status")
            yield ProgressBar(id="run-progress", total=self.config.num_solves, show_eta=False)
            yield Button("Abort", id="abort-btn", variant="error")
        with Horizontal(id="run-body"):
            with Vertical(id="left-col"):
                with Vertical(classes="card"):
                    yield Label("Cube", classes="card-title")
                    yield CubeNet("", id="cube-net")
                    yield Label("Moves: —", id="history")
                with Vertical(classes="card"):
                    yield Label("Stats", classes="card-title")
                    with Grid(id="stats-grid"):
                        yield Label("Turns")
                        yield Label("—", id="st-turns")
                        yield Label("Tool calls")
                        yield Label("—", id="st-tools")
                        yield Label("Moves")
                        yield Label("—", id="st-moves")
                        yield Label("Elapsed")
                        yield Label("—", id="st-time")
                        yield Label("Score")
                        yield Label("—", id="st-score")
            yield RichLog(id="log", highlight=True, markup=True, wrap=True)

    # ------------------------------------------------------------------ worker
    def on_mount(self) -> None:
        self.run_worker(self._drive, thread=True, exclusive=True, group="bench")

    def _drive(self) -> None:
        body = self.config.effective_extra_body()
        self.post_message(
            LogMsg(f"[cyan]Starting {self.config.num_solves} solve(s) / {self.config.max_turns} max turns "
                   f"-> {self.config.model} @ {self.config.base_url}[/cyan]")
        )
        if body:
            self.post_message(LogMsg(f"[dim]extra body params: {body}[/dim]"))
        client = OpenAICompatibleClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            stream=self.config.stream,
            temperature=self.config.temperature,
            extra_body=body,
            tool_choice=self.config.tool_choice,
        )
        try:
            runner = BenchmarkRunner(self.config, client, emitter=self._emit)
            result = runner.run(cancel_event=self.cancel)
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self.post_message(LogMsg(f"[red]Benchmark crashed: {exc}[/red]", level="error"))
            result = BenchmarkResult(
                config=self.config, solves=[], started_at_iso=datetime.now(timezone.utc).isoformat(), duration=0.0
            )
        self.post_message(BenchFinishedMsg(result))
    def _emit(self, kind: str, payload: dict) -> None:
        # post_message is thread-safe; safe to call from the worker thread.
        self._on_engine_event(kind, payload)


    def _on_engine_event(self, kind: str, payload: dict) -> None:
        if kind == "solve_started":
            self.post_message(SolveStartedMsg(payload["index"], self.config.num_solves, payload["scramble"]))
        elif kind == "turn":
            self.post_message(TurnMsg(payload["turn"], payload["content"], payload["tool_call_names"], payload["latency"]))
        elif kind == "state":
            p = payload
            self.post_message(StateMsg(p["facelets"], p["history"], p["total_moves"], p["turns"], p["tool_calls"], p["solved"]))
        elif kind == "log":
            self.post_message(LogMsg(payload["message"], payload.get("level", "info")))
        elif kind == "solve_done":
            self.post_message(SolveDoneMsg(payload["result"].index, self.config.num_solves, payload["result"]))

    # ---------------------------------------------------------------- handlers
    @on(SolveStartedMsg)
    def _on_solve_started(self, msg: SolveStartedMsg) -> None:

        self._solve_started = time.monotonic()
        self.query_one("#run-status", Static).update(
            f"Solve {msg.index + 1}/{msg.total}  |  scramble: {msg.scramble}"
        )
        log = self.query_one("#log", RichLog)
        log.write(f"\n[bold cyan]--- solve {msg.index + 1}/{msg.total} ---[/bold cyan]")
        log.write(f"[dim]scramble: {msg.scramble}[/dim]")

    @on(TurnMsg)
    def _on_turn(self, msg: TurnMsg) -> None:
        log = self.query_one("#log", RichLog)
        what = ", ".join(msg.tool_call_names) if msg.tool_call_names else "no tool call"
        content = (msg.content or "").strip().replace("\n", " ")[:80]
        line = f"[yellow]turn {msg.turn}[/yellow] ({msg.latency:.1f}s) [{what}]"
        if content:
            line += f"  {content}"
        log.write(line)

    @on(StateMsg)
    def _on_state(self, msg: StateMsg) -> None:
        self.query_one("#cube-net", CubeNet).set_state(msg.facelets)
        history = " ".join(msg.history[-40:])
        self.query_one("#history", Label).update(f"Moves: {msg.total_moves}  |  {history or '(none)'}")
        self.query_one("#st-turns", Label).update(str(msg.turns))
        self.query_one("#st-tools", Label).update(str(msg.tool_calls))
        self.query_one("#st-moves", Label).update(str(msg.total_moves))

        elapsed = time.monotonic() - getattr(self, "_solve_started", time.monotonic())
        self.query_one("#st-time", Label).update(f"{elapsed:.1f}s")
        if msg.solved:
            self.query_one("#st-score", Label).update("solved ✓")

    @on(SolveDoneMsg)
    def _on_solve_done(self, msg: SolveDoneMsg) -> None:
        self.completed += 1
        self.query_one("#run-progress", ProgressBar).advance(1)
        s = msg.solve
        status = "SOLVED" if s.solved else ("FAILED" if s.error else "UNSOLVED")
        color = "green" if s.solved else "red"
        log = self.query_one("#log", RichLog)
        log.write(
            f"[{color}]{status}[/{color}] moves={s.total_moves} turns={s.turns} tools={s.tool_calls} "
            f"par={s.par} score={s.score} time={s.elapsed:.1f}s"
            + (f"  [dim]({s.error})[/dim]" if s.error else "")
        )
        self.query_one("#st-score", Label).update(str(s.score))

    @on(BenchFinishedMsg)
    def _on_bench_finished(self, msg: BenchFinishedMsg) -> None:
        self.app.show_results(msg.result)  # type: ignore[attr-defined]

    @on(Button.Pressed, "#abort-btn")
    def _on_abort(self) -> None:
        self.cancel.set()
        self.query_one("#run-status", Static).update("Aborting after current step...")
        self.query_one("#abort-btn", Button).disabled = True
