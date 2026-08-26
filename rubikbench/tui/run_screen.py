"""Live run screen: one tall model pane, realtime stats, colored cube.

The model's reply streams chunk-by-chunk into a single scrollable pane:
reasoning in grey (toggleable with Ctrl+T), the actual response in white,
tool calls as they fire. Token counters, elapsed time, and token rate update
in real time. The benchmark runs in a worker thread and hands chunks to this
screen via messages (thread-safe).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import ClassVar

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Label, ProgressBar, Static

from ..benchmark import BenchmarkResult, BenchmarkRunner
from ..config import BenchmarkConfig, api_key_from_env
from ..llm import OpenAICompatibleClient
from .messages import (
    BenchFinishedMsg,
    LogMsg,
    SolveDoneMsg,
    SolveStartedMsg,
    StateMsg,
    StreamMsg,
    ToolCallMsg,
    ToolResultMsg,
    TurnMsg,
    TurnStartedMsg,
)
from .widgets import CubeNet


class RunScreen(Screen):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+t", "toggle_reasoning", "Toggle reasoning"),
    ]

    CSS = """
    #run-top { height: 3; padding: 0 1; }
    #run-status { width: 1fr; content-align: left middle; }
    #run-progress { width: 28; margin: 0 1; }
    #run-body { height: 1fr; padding: 0 1; }
    #left-col { width: 44; }
    .card { border: round $panel; padding: 0 1 1 1; margin: 0 0 1 0; }
    .card-title { text-style: bold; }
    #cube-net { width: auto; height: auto; }
    #history { height: 8; }
    #stats-grid { height: auto; grid-size: 2; grid-columns: 1fr 1fr; }
    #model-scroll { height: 1fr; border: round $panel; padding: 0 1; }
    #model-text { width: 1fr; }
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel = threading.Event()
        self.completed = 0

        # Transcript: completed entries as (kind, text) pairs. Kinds:
        # divider / reasoning / content / tool / result / log.
        self._history: list[tuple[str, str]] = []
        self._show_reasoning = True

        # In-flight streaming state.
        self._live_turn: int | None = None
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots: dict[int, dict[str, str]] = {}
        self._live_ttft: float | None = None

        # Request-in-flight state (shown while waiting for the first chunk).
        self._waiting_turn: int | None = None
        self._wait_started = 0.0
        self._wait_attempt = 1
        self._spinner = 0

        # Run-wide counters (from stream usage + turn results).
        self._tok_in = 0
        self._tok_out = 0
        self._tok_cached = 0
        self._tok_total = 0
        self._usage_turn: int | None = None

        self._bench_started = time.monotonic()
        self._solve_started = time.monotonic()
        self._stats_timer = None

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        with Horizontal(id="run-top"):
            yield Static("", id="run-status")
            yield ProgressBar(id="run-progress", total=self.config.num_solves, show_eta=False)
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
                        yield Label("Tokens in")
                        yield Label("0", id="st-tok-in")
                        yield Label("Tokens out")
                        yield Label("0", id="st-tok-out")
                        yield Label("Cached")
                        yield Label("0", id="st-tok-cached")
                        yield Label("Elapsed")
                        yield Label("0.0s", id="st-time")
                        yield Label("Speed")
                        yield Label("—", id="st-speed")
                        yield Label("Score")
                        yield Label("—", id="st-score")
            # One tall pane: the whole model conversation, live.
            with VerticalScroll(id="model-scroll"):
                yield Static("", id="model-text")
        yield Footer()

    # ------------------------------------------------------------------ worker
    def on_mount(self) -> None:
        self.run_worker(self._drive, thread=True, exclusive=True, group="bench")
        self._stats_timer = self.set_interval(0.5, self._tick_stats)

    def on_unmount(self) -> None:
        if self._stats_timer is not None:
            self._stats_timer.stop()

    def _drive(self) -> None:
        body = self.config.effective_extra_body()
        self._append_log(
            f"[cyan]Starting {self.config.num_solves} solve(s) / {self.config.max_turns} max turns "
            f"-> {escape(self.config.model)} @ {escape(self.config.base_url)}[/cyan]"
        )
        if body:
            self._append_log(f"[dim]extra body params: {body}[/dim]")
        # The TUI always streams: the whole point is watching the model work.
        client = OpenAICompatibleClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or api_key_from_env(self.config.base_url)[0],
            model=self.config.model,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            stream=True,
            temperature=self.config.temperature,
            extra_body=body,
            tool_choice=self.config.tool_choice,
        )
        try:
            runner = BenchmarkRunner(self.config, client, emitter=self._emit)
            result = runner.run(cancel_event=self.cancel)
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self._append_log(f"[red]Benchmark crashed: {exc}[/red]")
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
        elif kind == "turn_started":
            self.post_message(TurnStartedMsg(payload["turn"], payload.get("attempt", 1)))
        elif kind == "stream":
            p = payload
            self.post_message(StreamMsg(
                p["turn"], p.get("content"), p.get("tool_calls"), p.get("usage"),
                p.get("finish_reason"), p.get("ttft"), p.get("reasoning"),
            ))
        elif kind == "turn":
            self.post_message(TurnMsg(
                payload["turn"], payload["content"], payload["tool_call_names"],
                payload["latency"], payload.get("reasoning"),
            ))
        elif kind == "tool_call":
            self.post_message(ToolCallMsg(payload["turn"], payload["name"], payload["arguments"], payload["action"]))
        elif kind == "tool_result":
            self.post_message(ToolResultMsg(payload["turn"], payload["name"], payload["content"]))
        elif kind == "state":
            p = payload
            self.post_message(StateMsg(p["facelets"], p["history"], p["total_moves"], p["turns"], p["tool_calls"], p["solved"]))
        elif kind == "log":
            self.post_message(LogMsg(payload["message"], payload.get("level", "info")))
        elif kind == "solve_done":
            self.post_message(SolveDoneMsg(payload["result"].index, self.config.num_solves, payload["result"]))

    # --------------------------------------------------------------- model pane
    def _append(self, kind: str, text: str) -> None:
        self._history.append((kind, text))
        self._update_model()

    def _append_log(self, text: str) -> None:
        self._append("log", text)

    def _render_entry(self, kind: str, text: str) -> str:
        if kind == "divider":
            return f"[dim]── {text} ──[/dim]"
        if kind == "reasoning":
            return f"[grey62 italic]{escape(text)}[/]" if self._show_reasoning else ""
        if kind == "content":
            return escape(text)
        if kind == "result":
            return f"[dim]{escape(text)}[/dim]"
        return text  # tool / log entries already carry markup

    def _render_model(self) -> str:
        parts: list[str] = []
        for kind, text in self._history:
            rendered = self._render_entry(kind, text)
            if rendered:
                parts.append(rendered)
        if self._waiting_turn is not None:
            elapsed = time.monotonic() - self._wait_started
            retry = f" (retry {self._wait_attempt})" if self._wait_attempt > 1 else ""
            parts.append(f"[dim]waiting for model{retry} {self._spinner_char()} {elapsed:.0f}s[/dim]")
        if self._live_reasoning and self._show_reasoning:
            parts.append(f"[grey62 italic]{escape(self._live_reasoning)}[/]")
        if self._live_content:
            parts.append(escape(self._live_content))
        for idx in sorted(self._live_tool_slots):
            slot = self._live_tool_slots[idx]
            name = slot["name"] or "…"
            args = slot["args"] or "…"
            parts.append(f"[bright_black]→ tool {idx + 1}:[/bright_black] [bold cyan]{escape(name)}[/bold cyan]({escape(args)})")
        return "\n".join(parts)

    def _update_model(self) -> None:
        self.query_one("#model-text", Static).update(self._render_model())
        self.query_one("#model-scroll", VerticalScroll).scroll_end(animate=False)

    def _spinner_char(self) -> str:
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        char = frames[self._spinner % len(frames)]
        self._spinner += 1
        return char

    def _start_live_turn(self, turn: int) -> None:
        self._waiting_turn = None  # first chunk arrived; we are streaming now
        self._live_turn = turn
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots = {}
        self._live_ttft = None

    def action_toggle_reasoning(self) -> None:
        self._show_reasoning = not self._show_reasoning
        self._update_model()

    # ---------------------------------------------------------------- handlers
    @on(SolveStartedMsg)
    def _on_solve_started(self, msg: SolveStartedMsg) -> None:
        self._solve_started = time.monotonic()
        self._live_turn = None
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots = {}
        self._waiting_turn = None
        self.query_one("#run-status", Static).update(
            f"Solve {msg.index + 1}/{msg.total}  |  scramble: {msg.scramble}"
        )
        self._append("divider", f"solve {msg.index + 1}/{msg.total} · scramble: {msg.scramble}")

    @on(TurnStartedMsg)
    def _on_turn_started(self, msg: TurnStartedMsg) -> None:
        self._waiting_turn = msg.turn
        self._wait_started = time.monotonic()
        self._wait_attempt = msg.attempt
        self._tick_stats()  # paint the indicator right away, then every 0.5s

    @on(StreamMsg)
    def _on_stream(self, msg: StreamMsg) -> None:
        if msg.turn != self._live_turn:
            self._start_live_turn(msg.turn)
        if msg.reasoning:
            self._live_reasoning += msg.reasoning
        if msg.content:
            self._live_content += msg.content
        for tc in msg.tool_calls or []:
            slot = self._live_tool_slots.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
            if tc.get("id"):
                slot["id"] += tc["id"]
            if tc.get("name"):
                slot["name"] += tc["name"]
            if tc.get("arguments"):
                slot["args"] += tc["arguments"]
        if msg.ttft and self._live_ttft is None:
            self._live_ttft = msg.ttft
            self.query_one("#run-status", Static).update(f"streaming… · first token {msg.ttft:.1f}s")
        if msg.usage and msg.turn != self._usage_turn:
            # OpenAI sends usage once per request (on the final chunk).
            self._usage_turn = msg.turn
            self._tok_in += msg.usage.get("prompt", 0)
            self._tok_out += msg.usage.get("completion", 0)
            self._tok_cached += msg.usage.get("cached", 0)
            self._tok_total += msg.usage.get("total", 0)
            self.query_one("#st-tok-in", Label).update(f"{self._tok_in:,}")
            self.query_one("#st-tok-out", Label).update(f"{self._tok_out:,}")
            self.query_one("#st-tok-cached", Label).update(f"{self._tok_cached:,}")
        self._update_model()

    @on(TurnMsg)
    def _on_turn(self, msg: TurnMsg) -> None:
        self._waiting_turn = None
        if msg.reasoning or (msg.content and msg.content.strip()):
            self._append("divider", f"turn {msg.turn} · {msg.latency:.1f}s")
        if msg.reasoning and msg.reasoning.strip():
            self._append("reasoning", msg.reasoning.strip())
        if msg.content and msg.content.strip():
            self._append("content", msg.content.strip())
        self._live_turn = None
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots = {}
        self._live_ttft = None

    @on(ToolCallMsg)
    def _on_tool_call(self, msg: ToolCallMsg) -> None:
        args = escape(json.dumps(msg.arguments, ensure_ascii=False) if msg.arguments else "{}")
        icon = {"apply": "🔧", "reset": "↺"}.get(msg.action, "→")
        self._append("tool", f"{icon} [bold cyan]{escape(msg.name)}[/bold cyan]({args})")

    @on(ToolResultMsg)
    def _on_tool_result(self, msg: ToolResultMsg) -> None:
        first = (msg.content.strip().splitlines() or [""])[0]
        if len(first) > 200:
            first = first[:200] + "…"
        self._append("result", first)

    @on(StateMsg)
    def _on_state(self, msg: StateMsg) -> None:
        self.query_one("#cube-net", CubeNet).set_state(msg.facelets)
        history = " ".join(msg.history[-40:])
        self.query_one("#history", Label).update(f"Moves: {msg.total_moves}  |  {history or '(none)'}")
        self.query_one("#st-turns", Label).update(f"{msg.turns}/{self.config.max_turns}")
        self.query_one("#st-tools", Label).update(str(msg.tool_calls))
        self.query_one("#st-moves", Label).update(str(msg.total_moves))
        if msg.solved:
            self.query_one("#st-score", Label).update("solved ✓")

    @on(SolveDoneMsg)
    def _on_solve_done(self, msg: SolveDoneMsg) -> None:
        self.completed += 1
        self.query_one("#run-progress", ProgressBar).advance(1)
        s = msg.solve
        status = "SOLVED" if s.solved else ("FAILED" if s.error else "UNSOLVED")
        color = "green" if s.solved else "red"
        line = f"[{color}]{status}[/{color}] moves={s.total_moves} turns={s.turns} tools={s.tool_calls} "
        line += f"par={s.par} score={s.score} time={s.elapsed:.1f}s"
        if s.error:
            line += f"  [dim]({s.error})[/dim]"
        self._append("log", line)
        self.query_one("#st-score", Label).update(str(s.score))
        self.query_one("#run-status", Static).update(
            f"Solve {msg.index + 1}/{msg.total} {status}"
        )

    @on(BenchFinishedMsg)
    def _on_bench_finished(self, msg: BenchFinishedMsg) -> None:
        self.app.show_results(msg.result)  # type: ignore[attr-defined]

    @on(LogMsg)
    def _on_log(self, msg: LogMsg) -> None:
        self._append_log(msg.text)

    # ------------------------------------------------------------------ ticker
    def _tick_stats(self) -> None:
        """Periodic refresh of elapsed time, waiting spinner, and token rate."""
        if self._waiting_turn is not None:
            elapsed = time.monotonic() - self._wait_started
            retry = f" (retry {self._wait_attempt})" if self._wait_attempt > 1 else ""
            self.query_one("#run-status", Static).update(
                f"waiting for model{retry} {self._spinner_char()} {elapsed:.0f}s"
            )
            self._update_model()
        elif self._live_turn is not None:
            self._update_model()
        elapsed = time.monotonic() - self._solve_started
        self.query_one("#st-time", Label).update(f"{elapsed:.1f}s")
        if self._tok_out > 0 and elapsed > 0:
            self.query_one("#st-speed", Label).update(f"{self._tok_out / elapsed:.0f} tok/s")
