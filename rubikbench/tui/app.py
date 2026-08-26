"""RubikBench Textual application: screen wiring and shared state."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from .. import __version__
from ..benchmark import BenchmarkResult
from ..config import DEFAULT_CONFIG_PATH, BenchmarkConfig, load_config
from .config_screen import ConfigScreen
from .results_screen import ResultsScreen
from .run_screen import RunScreen


class RubikBenchApp(App):
    TITLE = "RubikBench"
    SUB_TITLE = f"LLM Rubik's cube benchmark v{__version__}"

    BINDINGS = [  # noqa: RUF012
        ("q", "quit", "Quit"),
        ("ctrl+r", "reset", "Reset config"),
    ]

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        try:
            self.config = load_config(self.config_path)
        except FileNotFoundError:
            self.config = BenchmarkConfig()
        self.result: BenchmarkResult | None = None

    def on_mount(self) -> None:
        self.push_screen(ConfigScreen())

    # -- navigation ---------------------------------------------------------
    def start_run(self) -> None:
        self.switch_screen(RunScreen(self.config))

    def show_results(self, result: BenchmarkResult) -> None:
        self.result = result
        self.switch_screen(ResultsScreen(result))

    def show_config(self) -> None:
        self.switch_screen(ConfigScreen())

    def action_reset(self) -> None:
        self.config = BenchmarkConfig()
        self.notify("Configuration reset to defaults.")
        self.show_config()
