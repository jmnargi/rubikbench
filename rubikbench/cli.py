"""Command-line entry point.

``rubikbench``            - launch the TUI (default)
``rubikbench run``        - headless benchmark run (for scripts/CI)
``rubikbench presets``    - list provider presets
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from . import __version__
from .benchmark import BenchmarkRunner, export_jsonl
from .config import (
    DEFAULT_CONFIG_PATH,
    PRESETS,
    api_key_from_env,
    api_key_source,
    apply_env_overrides,
    config_from_env,
    load_config,
    load_env,
)
from .llm import OpenAICompatibleClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rubikbench",
        description="Benchmark LLMs solving a Rubik's cube via OpenAI-compatible tool calls.",
    )
    parser.add_argument("--version", action="version", version=f"rubikbench {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a benchmark headlessly (no TUI)")
    run.add_argument("--config", type=Path, default=None, help="config JSON file (default: rubikbench_config.json, or .env settings when absent)")
    run.add_argument("-o", "--out", type=Path, default=None, help="JSONL output file")
    run.add_argument("--no-color", action="store_true", help="disable ANSI colors in progress output")

    sub.add_parser("presets", help="list available provider presets")

    agg = sub.add_parser(
        "aggregate",
        help="merge run JSONL files into one dataset JSON file",
    )
    agg.add_argument("files", nargs="+", type=Path, help="run JSONL files to merge")
    agg.add_argument("-o", "--out", type=Path, default=Path("dataset.json"), help="output dataset JSON file")

    view = sub.add_parser(
        "view",
        help="open a 3D web replay of a run file in the browser",
    )
    view.add_argument("file", type=Path, help="run JSONL file")
    view.add_argument("--port", type=int, default=None, help="HTTP port (default 8321)")
    view.add_argument("--no-open", action="store_true", help="do not open the browser")

    validate = sub.add_parser("validate", help="validate a config file (or .env settings)")
    validate.add_argument("--config", type=Path, default=None)
    return parser


def _print_progress_line(text: str, color: bool) -> None:
    if color and sys.stderr.isatty():
        text = f"\x1b[36m{text}\x1b[0m"
    print(text, file=sys.stderr, flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    # An explicitly requested config file that is missing is an error; running
    # without --config falls back to .env settings (no TUI / config file needed).
    explicit = args.config is not None
    path = args.config or DEFAULT_CONFIG_PATH
    try:
        if path.is_file():
            cfg = apply_env_overrides(load_config(path))
        elif explicit:
            raise FileNotFoundError(path)
        else:
            cfg = config_from_env()
        cfg.validate()
    except FileNotFoundError:
        print(
            f"error: config file not found: {path} (run the TUI first, or run without --config to use .env)",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:  # surface config errors
        print(f"error: invalid config: {exc}", file=sys.stderr)
        return 2

    if not cfg.api_key and api_key_source(cfg.base_url):
        print(
            f"error: no API key set; add {api_key_source(cfg.base_url)}=... (or RUBIKBENCH_API_KEY=...) to .env",
            file=sys.stderr,
        )
        return 2

    cancel = threading.Event()
    client = OpenAICompatibleClient(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        # Always stream: the run prints the model's text and tool calls live
        # to stderr, so a slow model never looks hung.
        stream=True,
        temperature=cfg.temperature,
        extra_body=cfg.effective_extra_body(),
        tool_choice=cfg.tool_choice if cfg.tool_choice != "auto" else "auto",
    )

    # Tracks whether we are mid-stream so line-based messages start on a fresh
    # line instead of gluing onto the model's text.
    stream_state = {"active": False}

    def end_stream_line() -> None:
        if stream_state["active"]:
            sys.stderr.write("\n")
            stream_state["active"] = False

    def emit(kind: str, payload: dict) -> None:
        if kind == "solve_started":
            _print_progress_line(f"[solve {payload['index'] + 1}/{cfg.num_solves}] scramble: {payload['scramble']}", not args.no_color)
        elif kind == "turn_started":
            attempt = f" (retry {payload['attempt']})" if payload["attempt"] > 1 else ""
            _print_progress_line(f"  turn {payload['turn']}: waiting for model{attempt}...", not args.no_color)
        elif kind == "stream":
            content = payload.get("content")
            if content:
                if not stream_state["active"]:
                    sys.stderr.write("  ")
                    stream_state["active"] = True
                sys.stderr.write(content)
                sys.stderr.flush()
        elif kind == "tool_call":
            end_stream_line()
            _print_progress_line(
                f"  → {payload['name']}({json.dumps(payload['arguments'], ensure_ascii=False)})", not args.no_color
            )
        elif kind == "tool_result":
            first = (payload["content"].strip().splitlines() or [""])[0]
            if len(first) > 140:
                first = first[:140] + "…"
            _print_progress_line(f"    {first}", not args.no_color)
        elif kind == "turn":
            end_stream_line()
            what = f" · tools: {', '.join(payload['tool_call_names'])}" if payload["tool_call_names"] else ""
            _print_progress_line(f"  turn {payload['turn']} · {payload['latency']:.1f}s{what}", not args.no_color)
        elif kind == "solve_done":
            end_stream_line()
            s = payload["result"]
            status = "SOLVED" if s.solved else ("FAILED" if s.error else "UNSOLVED")
            extra = f"moves={s.total_moves} turns={s.turns} tools={s.tool_calls} score={s.score}"
            _print_progress_line(f"[solve {s.index + 1}/{cfg.num_solves}] {status} ({extra})", not args.no_color)
        elif kind == "log":
            end_stream_line()
            _print_progress_line(f"  {payload['message']}", not args.no_color)

    runner = BenchmarkRunner(cfg, client, emitter=emit)
    try:
        result = runner.run(cancel_event=cancel)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    out = args.out or Path("rubikbench_results") / f"bench_{result.started_at_iso.replace(':', '-').replace('+', '_')}.jsonl"
    export_jsonl(result, out)

    agg = result.aggregates()
    print(json.dumps(agg, indent=2))
    print(f"results written to {out}")
    return 0


def cmd_presets() -> int:
    for name, p in PRESETS.items():
        env = f" (api key from ${p['env']})" if p.get("env") else " (no key needed)"
        print(f"{name:20} {p['base_url']:<40} model: {p['model'] or '<set in TUI>'}{env}")
    print("\nTip: put your provider key (OPENAI_API_KEY, OPENROUTER_API_KEY, ...) or RUBIKBENCH_API_KEY in .env and run `rubikbench run` without a config file.")
    return 0

def cmd_aggregate(args: argparse.Namespace) -> int:
    from .aggregate import aggregate_files, write_dataset

    try:
        dataset = aggregate_files(args.files)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = write_dataset(dataset, args.out)
    t = dataset["totals"]
    print(f"dataset: {t['solves']} solves across {dataset['runs']} run(s)")
    print(f"  solved {t['solved']}/{t['solves']} (rate {t['solve_rate']})")
    print(f"  tokens: {t['prompt_tokens']} in / {t['completion_tokens']} out / {t['cached_tokens']} cached")
    print(f"  turns {t['turns']} tools {t['tool_calls']} moves {t['moves']} retries {t['retries']} truncated {t['truncated']}")
    print(f"written to {out}")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    from .webui.server import DEFAULT_PORT, serve_run

    try:
        serve_run(args.file, port=args.port or DEFAULT_PORT, open_browser=not args.no_open)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    explicit = args.config is not None
    path = args.config or DEFAULT_CONFIG_PATH
    try:
        if path.is_file():
            cfg = apply_env_overrides(load_config(path))
        elif explicit:
            raise FileNotFoundError(path)
        else:
            cfg = config_from_env()
        cfg.validate()
    except FileNotFoundError:
        print(f"invalid: config file not found: {path}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface config errors
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    env_key, env_source = api_key_from_env(cfg.base_url)
    if env_key:
        key_state = f"set (from {env_source})"
    elif cfg.api_key:
        key_state = "set (from config file)"
    elif api_key_source(cfg.base_url):
        key_state = f"not set (add {api_key_source(cfg.base_url)}=... or RUBIKBENCH_API_KEY=... to .env)"
    else:
        key_state = "not set (local server needs no key)"
    print(
        f"config OK: {cfg.model} @ {cfg.base_url} "
        f"({cfg.num_solves} solves, {cfg.max_turns} max turns); api key: {key_state}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Credentials and run settings can come from a .env file next to the
    # project (gitignored); load it before any command reads the environment.
    load_env()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "presets":
        return cmd_presets()
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    if args.command == "view":
        return cmd_view(args)
    # default: TUI
    from .tui.app import RubikBenchApp

    app = RubikBenchApp()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
