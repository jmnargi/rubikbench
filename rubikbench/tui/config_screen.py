"""Configuration screen: endpoint, model, extra body params, benchmark knobs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Switch, TextArea

from ..config import PRESETS, BenchmarkConfig, load_config, save_config

_REASONING_OPTIONS = [("Default", ""), ("low", "low"), ("medium", "medium"), ("high", "high")]
_PAR_OPTIONS = [("auto (kociemba)", "auto"), ("God's number (20)", "god"), ("fixed value", "fixed")]
_TOOL_CHOICE_OPTIONS = [("auto", "auto"), ("none", "none"), ("required", "required")]

class ConfigScreen(Screen):
    CSS = """
    #config-scroll { padding: 1 2; }
    .card { border: round $panel; padding: 0 1 1 1; margin: 0 0 1 0; }
    .card-title { text-style: bold; }
    .row { height: auto; margin: 0 0 1 0; }
    .field-label { width: 22; content-align: right middle; }
    #buttons { height: auto; margin-top: 1; }
    .error { color: $error; }
    Button#start_btn { margin-right: 1; }
    """

    def compose(self) -> ComposeResult:
        cfg = self.app.config  # type: ignore[attr-defined]
        with VerticalScroll(id="config-scroll"):
            yield Label("RubikBench — benchmark config", id="title")
            with Vertical(classes="card"):
                yield Label("Endpoint", classes="card-title")
                with Horizontal(classes="row"):
                    yield Label("Preset", classes="field-label")
                    yield Select(
                        [(name, name) for name in PRESETS],
                        id="preset_select",
                        allow_blank=True,
                        prompt="Choose a provider preset...",
                    )
                    yield Button("Apply preset", id="apply_preset_btn")
                for label, field in (
                    ("Base URL (OpenAI-compatible /v1)", Input(cfg.base_url, id="base_url", placeholder="https://api.openai.com/v1")),
                    ("API key", Input(cfg.api_key, id="api_key", password=True, placeholder="sk-... (blank for local servers)")),
                    ("Model", Input(cfg.model, id="model", placeholder="gpt-4o")),
                ):
                    with Horizontal(classes="row"):
                        yield Label(label, classes="field-label")
                        yield field
            with Vertical(classes="card"):
                yield Label("Request", classes="card-title")
                rows = [
                    ("Reasoning effort", Select(_REASONING_OPTIONS, value=cfg.reasoning_effort or "", id="reasoning_select")),
                    ("Temperature (blank = default)", Input(
                        "" if cfg.temperature is None else str(cfg.temperature), id="temperature", placeholder="e.g. 0.2")),
                    ("Request timeout (s)", Input(str(cfg.timeout), id="timeout")),
                    ("Retries per request", Input(str(cfg.max_retries), id="max_retries")),
                    ("Tool choice", Select(_TOOL_CHOICE_OPTIONS, value=cfg.tool_choice, id="tool_choice")),
                    ("Stream responses", Switch(value=cfg.stream, id="stream_switch")),
                    ("Extra body params (JSON)", TextArea(json.dumps(cfg.extra_body or {}, indent=1), id="extra_body", language="json")),
                ]
                for label, field in rows:
                    with Horizontal(classes="row"):
                        yield Label(label, classes="field-label")
                        yield field
            with Vertical(classes="card"):
                yield Label("Benchmark", classes="card-title")
                rows = [
                    ("Number of solves", Input(str(cfg.num_solves), id="num_solves")),
                    ("Max turns per solve", Input(str(cfg.max_turns), id="max_turns")),
                    ("Scramble length", Input(str(cfg.scramble_len), id="scramble_len")),
                    ("Seed (blank = random)", Input("" if cfg.seed is None else str(cfg.seed), id="seed")),
                    ("Allow text moves", Switch(value=cfg.allow_text_moves, id="text_moves_switch")),
                    ("Custom scrambles file (optional)", Input(
                        "", id="scrambles_file", placeholder="path to file, one scramble per line")),
                ]
                for label, field in rows:
                    with Horizontal(classes="row"):
                        yield Label(label, classes="field-label")
                        yield field
            with Vertical(classes="card"):
                yield Label("Scoring", classes="card-title")
                rows = [
                    ("Weight: moves", Input(str(cfg.weight_moves), id="w_moves")),
                    ("Weight: turns", Input(str(cfg.weight_turns), id="w_turns")),
                    ("Weight: tool calls", Input(str(cfg.weight_tools), id="w_tools")),
                    ("Par strategy", Select(_PAR_OPTIONS, value=cfg.par_strategy, id="par_select")),
                    ("Fixed par (moves)", Input(str(cfg.par_fixed), id="par_fixed")),
                ]
                for label, field in rows:
                    with Horizontal(classes="row"):
                        yield Label(label, classes="field-label")
                        yield field
            with Horizontal(id="buttons"):
                yield Button("Start benchmark", id="start_btn", variant="primary")
                yield Button("Save config", id="save_btn")
                yield Button("Load config", id="load_btn")
                yield Button("Reset", id="reset_btn")
            yield Label("", id="status")

    # ------------------------------------------------------------------ helpers
    def _set_status(self, text: str, error: bool = False) -> None:
        label = self.query_one("#status", Label)
        label.update(text)
        label.set_class(error, "error")

    def _val(self, id_: str) -> str:
        return self.query_one(f"#{id_}", Input).value.strip()

    def _int(self, id_: str, default: int) -> int:
        text = self._val(id_)
        return int(text) if text else default

    def _float(self, id_: str, default: float) -> float:
        text = self._val(id_)
        return float(text) if text else default

    def _opt_int(self, id_: str) -> int | None:
        text = self._val(id_)
        return int(text) if text else None

    def _opt_float(self, id_: str) -> float | None:
        text = self._val(id_)
        return float(text) if text else None

    def _fill(self, id_: str, value: object) -> None:
        self.query_one(f"#{id_}", Input).value = "" if value is None else str(value)

    def _collect(self) -> BenchmarkConfig:
        cfg = BenchmarkConfig(
            base_url=self._val("base_url"),
            api_key=self._val("api_key"),
            model=self._val("model"),
            temperature=self._opt_float("temperature"),
            timeout=self._float("timeout", 120.0),
            max_retries=self._int("max_retries", 2),
            stream=self.query_one("#stream_switch", Switch).value,
            tool_choice=self.query_one("#tool_choice", Select).value or "auto",
            num_solves=self._int("num_solves", 5),
            max_turns=self._int("max_turns", 40),
            scramble_len=self._int("scramble_len", 22),
            seed=self._opt_int("seed"),
            allow_text_moves=self.query_one("#text_moves_switch", Switch).value,
            weight_moves=self._float("w_moves", 0.5),
            weight_turns=self._float("w_turns", 0.3),
            weight_tools=self._float("w_tools", 0.2),
            par_strategy=self.query_one("#par_select", Select).value or "auto",
            par_fixed=self._int("par_fixed", 20),
            reasoning_effort=self._reasoning_value(),
        )
        body = self.query_one("#extra_body", TextArea).text.strip()
        cfg.extra_body = json.loads(body) if body else {}
        scrambles_path = self._val("scrambles_file")
        if scrambles_path:
            lines = [ln.strip() for ln in Path(scrambles_path).read_text().splitlines() if ln.strip()]
            if not lines:
                raise ValueError(f"scrambles file is empty: {scrambles_path}")
            cfg.scrambles = lines
        cfg.validate()
        return cfg

    def _reasoning_value(self) -> str | None:
        value = self.query_one("#reasoning_select", Select).value
        return None if value in ("", None) else str(value)

    # ------------------------------------------------------------------ actions
    @on(Button.Pressed, "#start_btn")
    def _start(self) -> None:
        try:
            cfg = self._collect()
        except Exception as exc:  # noqa: BLE001 - user input/UI guard
            self._set_status(f"Configuration error: {exc}", error=True)
            return
        self.app.config = cfg  # type: ignore[attr-defined]
        try:
            save_config(cfg, self.app.config_path)  # type: ignore[attr-defined]
        except OSError as exc:
            self._set_status(f"Could not save config: {exc}", error=True)
            return
        self.app.start_run()  # type: ignore[attr-defined]

    @on(Button.Pressed, "#apply_preset_btn")
    def _apply_preset(self) -> None:
        select = self.query_one("#preset_select", Select)
        name = select.value
        if not name or name not in PRESETS:
            self._set_status("Pick a preset first.", error=True)
            return
        p = PRESETS[name]
        self._fill("base_url", p["base_url"])
        self._fill("model", p["model"])
        if p.get("env"):
            env_key = os.environ.get(p["env"], "")
            if env_key:
                self._fill("api_key", env_key)
        self._set_status(f"Preset '{name}' applied.")

    @on(Button.Pressed, "#save_btn")
    def _save(self) -> None:
        try:
            cfg = self._collect()
            save_config(cfg, self.app.config_path)  # type: ignore[attr-defined]
            self._set_status(f"Config saved to {self.app.config_path}.")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - user input/UI guard
            self._set_status(f"Could not save config: {exc}", error=True)

    @on(Button.Pressed, "#load_btn")
    def _load(self) -> None:
        try:
            cfg = load_config(self.app.config_path)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - user input/UI guard
            self._set_status(f"Could not load config: {exc}", error=True)
            return
        self._fill("base_url", cfg.base_url)
        self._fill("api_key", cfg.api_key)
        self._fill("model", cfg.model)
        self.query_one("#reasoning_select", Select).value = cfg.reasoning_effort or ""
        self._fill("temperature", cfg.temperature)
        self._fill("timeout", cfg.timeout)
        self._fill("max_retries", cfg.max_retries)
        self._fill("num_solves", cfg.num_solves)
        self._fill("max_turns", cfg.max_turns)
        self._fill("scramble_len", cfg.scramble_len)
        self._fill("seed", cfg.seed)
        self.query_one("#stream_switch", Switch).value = cfg.stream
        self.query_one("#text_moves_switch", Switch).value = cfg.allow_text_moves
        self._fill("w_moves", cfg.weight_moves)
        self._fill("w_turns", cfg.weight_turns)
        self._fill("w_tools", cfg.weight_tools)
        self._fill("par_fixed", cfg.par_fixed)
        self.query_one("#extra_body", TextArea).text = json.dumps(cfg.extra_body or {}, indent=1)
        self.app.config = cfg  # type: ignore[attr-defined]
        self._set_status("Config loaded.")

    @on(Button.Pressed, "#reset_btn")
    def _reset(self) -> None:
        self.app.config = BenchmarkConfig()  # type: ignore[attr-defined]
        self.app.show_config()  # type: ignore[attr-defined]
