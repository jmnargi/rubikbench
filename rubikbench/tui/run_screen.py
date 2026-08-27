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
from typing import Any, ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Label, Static

from ..benchmark import BenchmarkResult, BenchmarkRunner
from ..config import BenchmarkConfig, api_key_from_env
from ..llm import OpenAICompatibleClient
from .messages import (
    BenchFinishedMsg,
    ContextMsg,
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
    #run-body { height: 1fr; padding: 0 1; }
    #left-col { width: 54; }
    .card { border: round $panel; padding: 0 1 1 1; margin: 0 0 1 0; }
    .card-title { text-style: bold; }
    #cube-net { width: auto; height: auto; }
    #history { height: 8; }
    #info-grid { height: auto; grid-size: 2; grid-columns: 1fr 2fr; }
    #info-grid Label { width: 1fr; }
    .info-label { text-style: dim; }
    #stats-grid { height: auto; grid-size: 2; grid-columns: 1fr 1fr; }
    #run-status { height: 1; content-align: left middle; }
    #model-scroll { height: 1fr; border: round $panel; padding: 0 1; }
    #history-text { width: 1fr; }
    #live-text { width: 1fr; }
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel = threading.Event()
        self.completed = 0

        # Completed LLM-output history as (kind, text). Kinds:
        # divider / reasoning / content / tool / result / log.
        self._history: list[tuple[str, str | Text]] = []
        self._show_reasoning = True

        # In-flight streaming state.
        self._live_turn: int | None = None
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots: dict[int, dict[str, str]] = {}
        self._live_ttft: float | None = None
        # Realtime rate bookkeeping: when the current stream began and how many
        # output characters have arrived (the API only reports usage at the end).
        self._live_stream_started = 0.0
        self._live_chars = 0

        # Request-in-flight state (shown while waiting for the first chunk).
        self._waiting_turn: int | None = None
        self._wait_started = 0.0
        self._wait_attempt = 1
        self._spinner = 0

        # Run-wide counters (from API usage, using delta math so cumulative
        # usage reported by some proxies does not double-count).
        self._tok_in = 0
        self._tok_out = 0
        self._tok_reasoning = 0
        self._tok_cached = 0
        self._tok_total = 0
        self._usage_by_turn: dict[int, dict[str, int]] = {}
        # Latest provider-reported completion tokens, shown against the output cap.
        self._last_output_use: int | None = None

        self._solve_started = time.monotonic()
        self._stats_timer = None

        # Live pane render throttling: coalesce high-frequency chunks so a
        # fast stream never stalls the UI thread re-rendering a growing string.
        self._live_dirty = False
        self._last_live_render = 0.0

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        with Horizontal(id="run-body"):
            with Vertical(id="left-col"):
                with Vertical(classes="card"):
                    yield Label("Cube", classes="card-title")
                    yield CubeNet("", id="cube-net")
                    yield Label("Moves: —", id="history")
                with Vertical(classes="card"):
                    yield Label("Run", classes="card-title")
                    with Grid(id="info-grid"):
                        yield Label("model:", classes="info-label")
                        yield Label(self.config.model, id="info-model")
                        yield Label("endpoint:", classes="info-label")
                        yield Label(self.config.base_url, id="info-endpoint")
                        yield Label("solves:", classes="info-label")
                        yield Label(f"{self.config.num_solves}", id="info-solves")
                        yield Label("max turns:", classes="info-label")
                        yield Label(f"{self.config.max_turns}", id="info-turns")
                        yield Label("output cap:", classes="info-label")
                        yield Label(f"{self.config.max_output_tokens:,}" if self.config.max_output_tokens else "—", id="info-tokens")
                        yield Label("protocol / view:", classes="info-label")
                        yield Label(f"{self.config.protocol_mode} / {self.config.presentation_mode}", id="info-protocol")
                        yield Label("context / trim:", classes="info-label")
                        yield Label("—", id="info-context")
                        yield Label("temperature:", classes="info-label")
                        yield Label(str(self.config.temperature if self.config.temperature is not None else "—"), id="info-temp")
                with Vertical(classes="card"):
                    yield Label("Stats", classes="card-title")
                    with Grid(id="stats-grid"):
                        yield Label("Turns")
                        yield Label("—", id="st-turns")
                        yield Label("Tools")
                        yield Label("—", id="st-tools")
                        yield Label("Text actions")
                        yield Label("—", id="st-text")
                        yield Label("Moves")
                        yield Label("—", id="st-moves")
                        yield Label("Tokens in")
                        yield Label("0", id="st-tok-in")
                        yield Label("Tokens out")
                        yield Label("0", id="st-tok-out")
                        yield Label("Output use")
                        yield Label("—", id="st-out-cap")
                        yield Label("Reasoning (reported)")
                        yield Label("0", id="st-reasoning")
                        yield Label("Cached (reported)")
                        yield Label("0", id="st-tok-cached")
                        yield Label("Elapsed")
                        yield Label("0.0s", id="st-time")
                        yield Label("Speed")
                        yield Label("—", id="st-speed")
                        yield Label("Score")
                        yield Label("—", id="st-score")
                yield Static("", id="run-status")
            # One tall pane: the whole model conversation, live.
            with VerticalScroll(id="model-scroll"):
                yield Static("", id="history-text")
                yield Static("", id="live-text")
        yield Footer()

    # ------------------------------------------------------------------ worker
    def on_mount(self) -> None:
        self.run_worker(self._drive, thread=True, exclusive=True, group="bench")
        self._stats_timer = self.set_interval(0.1, self._tick_stats)

    def on_unmount(self) -> None:
        if self._stats_timer is not None:
            self._stats_timer.stop()

    def _drive(self) -> None:
        body = self.config.effective_extra_body()
        # The TUI always streams: the whole point is watching the model work.
        client = OpenAICompatibleClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key or api_key_from_env(self.config.base_url)[0],
            model=self.config.model,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            stream=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_output_tokens,
            stream_idle_timeout=self.config.stream_idle_timeout,
            loop_detection=self.config.loop_detection,
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
                payload.get("prompt_tokens", 0),
                payload.get("completion_tokens", 0),
                payload.get("reasoning_tokens", 0),
                payload.get("cached_tokens", 0),
                payload.get("total_tokens", 0),
                payload.get("finish_reason"),
            ))
        elif kind == "tool_call":
            self.post_message(ToolCallMsg(payload["turn"], payload["name"], payload["arguments"], payload["action"]))
        elif kind == "tool_result":
            self.post_message(ToolResultMsg(payload["turn"], payload["name"], payload["content"]))
        elif kind == "state":
            p = payload
            self.post_message(StateMsg(
                p["facelets"], p["history"], p["total_moves"], p["turns"],
                p["tool_calls"], p["solved"], p.get("text_actions", 0),
            ))
        elif kind == "context":
            self.post_message(ContextMsg(
                payload["current"], payload["peak"], payload.get("input_budget"),
                payload["trim_count"], payload.get("output_cap"),
            ))
        elif kind == "log":
            self.post_message(LogMsg(payload["message"], payload.get("level", "info")))
        elif kind == "solve_done":
            self.post_message(SolveDoneMsg(payload["result"].index, self.config.num_solves, payload["result"]))

    # --------------------------------------------------------------- model pane
    def _append_history(self, kind: str, text: str | Text) -> None:
        self._history.append((kind, text))
        self._refresh_history()
        self._scroll_model()

    def _append_log(self, text: str) -> None:
        self._append_history("log", text)

    def _render_entry(self, kind: str, value: str | Text) -> Text | None:
        if kind == "divider":
            return Text(f"── {value} ──", style="dim")
        if kind == "reasoning":
            return Text(value, style="#888888 italic") if self._show_reasoning else None
        if kind == "content":
            return Text(value, style="white")
        if kind == "result":
            return Text(value, style="dim")
        # tool entries are pre-rendered Text; log entries are trusted markup.
        return value if isinstance(value, Text) else Text.from_markup(value)

    def _refresh_history(self) -> None:
        rendered = Text()
        first = True
        for kind, text in self._history:
            entry = self._render_entry(kind, text)
            if entry is None:
                continue
            if not first:
                rendered.append("\n")
            rendered.append_text(entry)
            first = False
        self.query_one("#history-text", Static).update(rendered)

    def _render_tool_call(self, name: str, arguments: dict[str, Any] | None, idx: int | None = None) -> Text:
        if arguments:
            # Show each kwarg as name="value" on one clean line.
            rendered = ", ".join(
                f"{k}={json.dumps(v, ensure_ascii=False)}"
                for k, v in arguments.items()
            )
        else:
            rendered = ""
        result = Text()
        if idx is not None:
            result.append(f"tool {idx + 1}: ", style="bright_black")
        result.append(name, style="bold cyan")
        result.append(f"({rendered})")
        return result

    def _render_live(self) -> Text:
        parts: list[Text] = []
        if self._live_reasoning and self._show_reasoning:
            parts.append(Text(self._live_reasoning, style="#888888 italic"))
        if self._live_content:
            parts.append(Text(self._live_content, style="white"))
        for idx in sorted(self._live_tool_slots):
            slot = self._live_tool_slots[idx]
            try:
                args = json.loads(slot["args"]) if slot["args"] else None
            except json.JSONDecodeError:
                args = {"_unparsed": slot["args"]}
            parts.append(self._render_tool_call(slot["name"] or "…", args, idx))
        return self._join_texts(parts)

    @staticmethod
    def _join_texts(parts: list[Text]) -> Text:
        """Join rendered segments with newlines into one styled Text."""
        result = Text()
        for i, part in enumerate(parts):
            if i:
                result.append("\n")
            result.append_text(part)
        return result

    def _update_live(self) -> None:
        """Render the live pane now (toggles and final flushes use this)."""
        self._last_live_render = time.monotonic()
        self._live_dirty = False
        self.query_one("#live-text", Static).update(self._render_live())
        self._scroll_model()

    def _maybe_flush_live(self) -> None:
        """Coalesced render: re-render at most every ~30ms while streaming."""
        self._live_dirty = True
        if time.monotonic() - self._last_live_render >= 0.03:
            self._update_live()

    def _add_usage(self, turn: int, usage: dict[str, int] | None) -> None:
        """Add usage with delta math; some proxies report cumulative per-chunk totals."""
        if not usage:
            return
        prompt = usage.get("prompt", 0)
        completion = usage.get("completion", 0)
        reasoning = usage.get("reasoning", 0)
        cached = usage.get("cached", 0)
        total = usage.get("total", 0)
        if prompt == 0 and completion == 0 and cached == 0 and total == 0 and reasoning == 0:
            return

        prev = self._usage_by_turn.get(turn, {})
        prev_prompt = prev.get("prompt", 0)
        prev_completion = prev.get("completion", 0)
        prev_reasoning = prev.get("reasoning", 0)
        prev_cached = prev.get("cached", 0)
        prev_total = prev.get("total", 0)

        delta_prompt = max(0, prompt - prev_prompt)
        delta_completion = max(0, completion - prev_completion)
        delta_reasoning = max(0, reasoning - prev_reasoning)
        delta_cached = max(0, cached - prev_cached)
        delta_total = max(0, total - prev_total)

        # response tokens = completion minus reasoning (when the API separates them)
        delta_response = max(0, delta_completion - delta_reasoning)

        self._usage_by_turn[turn] = {
            "prompt": prompt,
            "completion": completion,
            "reasoning": reasoning,
            "cached": cached,
            "total": total,
        }

        self._tok_in += delta_prompt
        self._tok_out += delta_response
        self._tok_reasoning += delta_reasoning
        self._tok_cached += delta_cached
        self._tok_total += delta_total
        self.query_one("#st-tok-in", Label).update(f"{self._tok_in:,}")
        self.query_one("#st-tok-out", Label).update(f"{self._tok_out:,}")
        self.query_one("#st-reasoning", Label).update(f"{self._tok_reasoning:,}")
        self.query_one("#st-tok-cached", Label).update(f"{self._tok_cached:,}")

    def _scroll_model(self) -> None:
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
        self._live_stream_started = time.monotonic()
        self._live_chars = 0

    def action_toggle_reasoning(self) -> None:
        self._show_reasoning = not self._show_reasoning
        self._refresh_history()
        self._update_live()

    # ---------------------------------------------------------------- handlers
    @on(ContextMsg)
    def _on_context(self, msg: ContextMsg) -> None:
        budget = f"{msg.budget:,}" if msg.budget is not None else "unlimited"
        self.query_one("#info-context", Label).update(
            f"{msg.current:,} current / {msg.peak:,} peak of {budget}; trims {msg.trim_count}"
        )

    @on(SolveStartedMsg)
    def _on_solve_started(self, msg: SolveStartedMsg) -> None:
        self._solve_started = time.monotonic()
        self._live_turn = None
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots = {}
        self._waiting_turn = None
        self._last_output_use = None
        self.query_one("#st-text", Label).update("0")
        self.query_one("#st-out-cap", Label).update("—")
        self.query_one("#run-status", Static).update(
            f"Solve {msg.index + 1}/{msg.total} · {msg.scramble}"
        )
        self._append_history("divider", f"solve {msg.index + 1}/{msg.total} · {msg.scramble}")

    @on(TurnStartedMsg)
    def _on_turn_started(self, msg: TurnStartedMsg) -> None:
        self._waiting_turn = msg.turn
        self._wait_started = time.monotonic()
        self._wait_attempt = msg.attempt
        self._tick_stats()  # paint the indicator right away, then every 0.1s

    @on(StreamMsg)
    def _on_stream(self, msg: StreamMsg) -> None:
        if msg.turn != self._live_turn:
            self._start_live_turn(msg.turn)
        if msg.reasoning:
            self._live_reasoning += msg.reasoning
            self._live_chars += len(msg.reasoning)
        if msg.content:
            self._live_content += msg.content
            self._live_chars += len(msg.content)
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
        self._add_usage(msg.turn, msg.usage)
        self._maybe_flush_live()

    @on(TurnMsg)
    def _on_turn(self, msg: TurnMsg) -> None:
        self._waiting_turn = None
        has_output = bool(msg.reasoning and msg.reasoning.strip()) or bool(msg.content and msg.content.strip())
        if has_output:
            divider = f"turn {msg.turn} · {msg.latency:.1f}s"
            if msg.finish_reason:
                divider += f" · {msg.finish_reason}"
            self._append_history("divider", divider)
        if msg.reasoning and msg.reasoning.strip():
            self._append_history("reasoning", msg.reasoning.strip())
        if msg.content and msg.content.strip():
            self._append_history("content", msg.content.strip())
        self._add_usage(
            msg.turn,
            {
                "prompt": msg.prompt_tokens,
                "completion": msg.completion_tokens,
                "reasoning": msg.reasoning_tokens,
                "cached": msg.cached_tokens,
                "total": msg.total_tokens,
            },
        )
        self._last_output_use = msg.completion_tokens
        self._update_output_use()
        self._live_turn = None
        self._live_content = ""
        self._live_reasoning = ""
        self._live_tool_slots = {}
        self._live_ttft = None
        self.query_one("#live-text", Static).update("")

    def _update_output_use(self) -> None:
        """Latest completion tokens versus the configured output cap."""
        cap = self.config.max_output_tokens
        if self._last_output_use is None:
            self.query_one("#st-out-cap", Label).update("—")
        elif cap:
            self.query_one("#st-out-cap", Label).update(f"{self._last_output_use:,}/{cap:,}")
        else:
            self.query_one("#st-out-cap", Label).update(f"{self._last_output_use:,}")


    @on(ToolCallMsg)
    def _on_tool_call(self, msg: ToolCallMsg) -> None:
        self._append_history("tool", self._render_tool_call(msg.name, msg.arguments))

    @on(ToolResultMsg)
    def _on_tool_result(self, msg: ToolResultMsg) -> None:
        first = (msg.content.strip().splitlines() or [""])[0]
        if len(first) > 200:
            first = first[:200] + "…"
        self._append_history("result", first)

    @on(StateMsg)
    def _on_state(self, msg: StateMsg) -> None:
        self.query_one("#cube-net", CubeNet).set_state(msg.facelets)
        history = " ".join(msg.history[-40:])
        self.query_one("#history", Label).update(f"Moves: {msg.total_moves}  |  {history or '(none)'}")
        self.query_one("#st-turns", Label).update(f"{msg.turns}/{self.config.max_turns}")
        self.query_one("#st-tools", Label).update(str(msg.tool_calls))
        self.query_one("#st-text", Label).update(str(msg.text_actions))
        self.query_one("#st-moves", Label).update(str(msg.total_moves))
        if msg.solved:
            self.query_one("#st-score", Label).update("solved ✓")

    @on(SolveDoneMsg)
    def _on_solve_done(self, msg: SolveDoneMsg) -> None:
        self.completed += 1
        s = msg.solve
        status = "SOLVED" if s.solved else ("FAILED" if s.error else "UNSOLVED")
        color = "green" if s.solved else "red"
        line = (f"[{color}]{status}[/{color}] moves={s.total_moves} turns={s.turns} "
                f"tools={s.tool_calls} text={s.text_actions} "
                f"par={s.par} score={s.score} time={s.elapsed:.1f}s")
        if s.finish_reasons:
            line += f" finish={','.join(s.finish_reasons)}"
            if s.truncated:
                line += " (truncated)"
        line += (f" est: reasoning={s.estimated_reasoning_tokens} "
                 f"cacheable={s.estimated_cacheable_tokens}")
        if s.error:
            line += f"  [dim]({s.error})[/dim]"
        self._append_history("log", line)
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
        elapsed = time.monotonic() - self._solve_started
        self.query_one("#st-time", Label).update(f"{elapsed:.1f}s")
        if self._live_turn is not None:
            # Stream usage generally arrives only in the final SSE chunk, so
            # estimate the active rate from streamed reasoning and content.
            stream_elapsed = time.monotonic() - self._live_stream_started
            if self._live_chars > 0 and stream_elapsed > 0:
                estimated_tokens = (self._live_chars + 3) // 4
                self.query_one("#st-speed", Label).update(
                    f"{estimated_tokens / stream_elapsed:.0f} tok/s"
                )
        elif self._tok_out > 0 and elapsed > 0:
            self.query_one("#st-speed", Label).update(f"{self._tok_out / elapsed:.0f} tok/s")
