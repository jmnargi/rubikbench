"""RubikBench Textual application: a view-only live run monitor.

The TUI starts the benchmark immediately from the resolved configuration
(config file if present, otherwise ``.env``) and shows live streaming output.
There is no config menu and nothing to click through: press Ctrl+C (or ``q``)
to quit.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App
from textual.binding import Binding

from .. import __version__
from ..benchmark import BenchmarkResult
from ..config import (
    DEFAULT_CONFIG_PATH,
    apply_env_overrides,
    config_from_env,
    load_config,
)
from .replay_screen import ReplayScreen
from .results_screen import ResultsScreen
from .run_screen import RunScreen


class RubikBenchApp(App):
    TITLE = "RubikBench"
    SUB_TITLE = f"LLM Rubik's cube benchmark v{__version__}"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config_path: str | Path | None = None, *, auto_run: bool = True) -> None:
        super().__init__()
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        try:
            cfg = apply_env_overrides(load_config(path))
        except FileNotFoundError:
            cfg = config_from_env()
        cfg.validate()  # raises ValueError; the caller surfaces it with help
        self.config = cfg
        self.result: BenchmarkResult | None = None
        # auto_run=False lets tests/embedders mount their own screens without
        # the default run starting in on_mount.
        self.auto_run = auto_run

    def on_mount(self) -> None:
        # View-only: the run starts immediately, no menus or buttons.
        if self.auto_run:
            self.push_screen(RunScreen(self.config))

    # -- navigation ---------------------------------------------------------
    def start_run(self) -> None:
        self.switch_screen(RunScreen(self.config))

    def show_results(self, result: BenchmarkResult) -> None:
        self.result = result
        self.switch_screen(ResultsScreen(result))

    def show_replay(self, result: BenchmarkResult, initial_solve: int = 0) -> None:
        self.result = result
        self.push_screen(ReplayScreen(result, initial_solve))
