"""Replay screen: step through a completed run's state timeline."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, OptionList, ProgressBar, Select, Static

from ..benchmark import BenchmarkResult
from .widgets import CubeNet


class ReplayScreen(Screen):
    """Walk the per-solve timeline (scramble start -> each move batch)."""

    BINDINGS = [  # noqa: RUF012
        ("space", "toggle_play", "Play/pause"),
        ("left", "prev", "Previous step"),
        ("right", "next", "Next step"),
        ("+", "faster", "Faster"),
        ("-", "slower", "Slower"),
        ("escape", "back", "Back to results"),
    ]

    CSS = """
    #replay-row { height: 1fr; }
    #replay-left { width: 58%; height: 100%; padding: 1; }
    #replay-right { width: 42%; height: 100%; padding: 1 2; border-left: solid $primary; }
    #cube { height: 16; }
    #controls { height: 3; content-align: center middle; }
    #controls Button { margin: 0 1; }
    #step-info { height: 5; border: round $primary; padding: 1; }
    #timeline { height: 1fr; }
    #speed-label { width: 10; content-align: left middle; }
    """

    def __init__(self, result: BenchmarkResult, initial_solve: int = 0) -> None:
        super().__init__()
        self.result = result
        self.index = min(initial_solve, max(0, len(result.solves) - 1))
        self.step = 0
        self.playing = False
        self.speed = 1.0
        self._timer: Any | None = None

    # ------------------------------------------------------------------ data
    def _timeline(self) -> list[dict[str, Any]]:
        solve = self.result.solves[self.index]
        return getattr(solve, "timeline", None) or []

    def _push_info_solves(self) -> None:
        solves = self.result.solves
        self.query_one("#solve-select", Select).set_options([
            ((f"#{s.index} {'solved' if s.solved else 'unsolved'} "
              f"moves={s.total_moves} turns={s.turns} score={s.score}"), s.index)
            for s in solves
        ])
        if solves:
            self.query_one("#solve-select", Select).value = solves[self.index].index

    def compose(self) -> ComposeResult:
        with Horizontal(id="replay-row"):
            with Vertical(id="replay-left"):
                yield CubeNet("", id="cube")
                with Horizontal(id="controls"):
                    yield Button("Play", id="btn-play", variant="primary")
                    yield Button("◀", id="btn-prev")
                    yield Button("▶", id="btn-next")
                    yield ProgressBar(total=1, show_eta=False, show_percentage=False, id="slider")
                    yield Label("1x", id="speed-label")
                yield Static("", id="step-info")
            with Vertical(id="replay-right"):
                yield Select([("", "")], id="solve-select", prompt="Solve")
                yield Label("Timeline", classes="section-title")
                yield OptionList(id="timeline")

    def on_mount(self) -> None:
        self._push_info_solves()
        self._rebuild_timeline()
        self._goto(0, render=True)

    # ------------------------------------------------------------------ core
    def _rebuild_timeline(self) -> None:
        tl = self._timeline()
        op = self.query_one("#timeline", OptionList)
        op.clear_options()
        if not tl:
            op.add_option("No replay data for this solve.")
            return
        op.add_options([
            f"{i:>3} {t.get('t', 0):6.1f}s {t.get('action', '')!s:8} "
            f"{' '.join(t.get('moves', []))[:48]}"
            for i, t in enumerate(tl)
        ])
        slider = self.query_one("#slider", ProgressBar)
        slider.total = max(1, len(tl) - 1)
        slider.progress = 0

    def _goto(self, step: int, render: bool = False) -> None:
        tl = self._timeline()
        if not tl:
            return
        self.step = max(0, min(len(tl) - 1, step))
        entry = tl[self.step]
        if render:
            self.query_one("#cube", CubeNet).set_state(entry.get("facelets", ""))
        self.query_one("#slider", ProgressBar).progress = self.step
        self.query_one("#step-info", Static).update(
            f"step {self.step}/{len(tl) - 1}   turn {entry.get('turn', '-')}   "
            f"{entry.get('action', '')}\n"
            f"moves: {' '.join(entry.get('moves', [])) or '—'}\n"
            f"t={entry.get('t', 0):.1f}s   "
            f"{'SOLVED' if entry.get('solved') else 'not solved'}"
        )
        timeline = self.query_one("#timeline", OptionList)
        if self.step < timeline.option_count:
            timeline.highlighted = self.step

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.step >= len(self._timeline()) - 1:
            self.playing = False
            self._sync_play_ui()
            return
        self._goto(self.step + 1, render=True)

    def _sync_play_ui(self) -> None:
        self.query_one("#btn-play", Button).label = "Pause" if self.playing else "Play"

    # ------------------------------------------------------------------ input
    def action_toggle_play(self) -> None:
        if not self._timeline():
            return
        self.playing = not self.playing
        self._sync_play_ui()
        if self.playing:
            if self._timer is None:
                self._timer = self.set_interval(1 / 8, self._tick)
            if self.step >= len(self._timeline()) - 1:
                self._goto(0, render=True)
        elif self._timer is not None:
            self._timer.pause()
            self._timer = None

    def action_prev(self) -> None:
        self.playing = False
        self._sync_play_ui()
        self._goto(self.step - 1, render=True)

    def action_next(self) -> None:
        self.playing = False
        self._sync_play_ui()
        self._goto(self.step + 1, render=True)

    def action_faster(self) -> None:
        self.speed = min(16.0, self.speed * 2)
        self._apply_speed()

    def action_slower(self) -> None:
        self.speed = max(0.5, self.speed / 2)
        self._apply_speed()

    def _apply_speed(self) -> None:
        self.query_one("#speed-label", Label).update(f"{self.speed:g}x")
        if self._timer is not None:
            self._timer.stop()
        if self.playing:
            self._timer = self.set_interval(1 / (8 * self.speed), self._tick)
        else:
            self._timer = None

    def action_back(self) -> None:
        self.app.show_results(self.result)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ handlers
    @on(Button.Pressed, "#btn-play")
    def _on_play(self) -> None:
        self.action_toggle_play()

    @on(Button.Pressed, "#btn-prev")
    def _on_prev(self) -> None:
        self.action_prev()

    @on(Button.Pressed, "#btn-next")
    def _on_next(self) -> None:
        self.action_next()

    @on(OptionList.OptionSelected, "#timeline")
    def _on_timeline(self, event: OptionList.OptionSelected) -> None:
        self.playing = False
        self._sync_play_ui()
        self._goto(event.option_index, render=True)

    @on(Select.Changed, "#solve-select")
    def _on_solve(self, event: Select.Changed) -> None:
        value = event.value
        if isinstance(value, int) and 0 <= value < len(self.result.solves):
            self.playing = False
            self._sync_play_ui()
            self.index = value
            self._rebuild_timeline()
            self._goto(0, render=True)
