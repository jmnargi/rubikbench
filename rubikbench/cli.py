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
from dataclasses import replace
from pathlib import Path

from . import __version__
from .benchmark import BenchmarkRunner, export_jsonl
from .config import (
    DEFAULT_CONFIG_PATH,
    PRESETS,
    api_key_from_env,
    api_key_source,
    apply_env_overrides,
    apply_profile,
    config_from_env,
    load_config,
    load_env,
)
from .llm import OpenAICompatibleClient


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Endpoint, request, and benchmark flags shared by the top-level command
    (bare ``rubikbench`` runs a benchmark) and the ``run`` subcommand.
    """
    parser.add_argument("--config", type=Path, default=None, help="config JSON file (default: rubikbench_config.json, or .env settings when absent)")
    parser.add_argument("-o", "--out", type=Path, default=None, help="JSONL output file")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors in progress output")
    # Endpoint and request knobs; each overrides .env / the config file.
    parser.add_argument("--base-url", type=str, default=None, help="endpoint URL (overrides RUBIKBENCH_BASE_URL)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides .env)")
    parser.add_argument("--model", type=str, default=None, help="model name (overrides RUBIKBENCH_MODEL)")
    parser.add_argument("--profile", choices=("smoke", "diagnostic", "full", "research"), default=None,
                        help="apply named run defaults before explicit CLI flags")
    parser.add_argument("--context-window-tokens", type=int, default=None,
                        help="total context window; input is budgeted after output reserve")
    parser.add_argument("--output-token-reserve", type=int, default=None,
                        help="tokens reserved for output while budgeting input")
    parser.add_argument("--protocol-mode", choices=("tool_only", "text_compat"), default=None,
                        help="move protocol (tool_only is the benchmark default)")
    parser.add_argument("--presentation-mode", choices=("stickers-v1", "cubie-v1"), default=None,
                        help="cube-state presentation schema")
    parser.add_argument("--scramble-preset", type=str, default=None,
                        help="premade benchmark set, including versioned ladder fixtures")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="context cap (overrides RUBIKBENCH_MAX_INPUT_TOKENS)")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="sent as max_tokens (overrides RUBIKBENCH_MAX_OUTPUT_TOKENS)")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (overrides RUBIKBENCH_TEMPERATURE)")
    parser.add_argument("--top-p", type=float, default=None, help="nucleus sampling top_p (overrides RUBIKBENCH_TOP_P)")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="repetition penalty (overrides RUBIKBENCH_REPETITION_PENALTY)")
    parser.add_argument("--top-k", type=int, default=None, help="top-k sampling (overrides RUBIKBENCH_TOP_K)")
    parser.add_argument("--stream-idle-timeout", type=float, default=None, help="abort+retry if no chunk for this many seconds (overrides RUBIKBENCH_STREAM_IDLE_TIMEOUT)")
    parser.add_argument(
        "--loop-detection",
        dest="loop_detection",
        action="store_true",
        default=None,
        help="abort+retry when model output loops (default: on)",
    )
    parser.add_argument(
        "--no-loop-detection",
        dest="loop_detection",
        action="store_false",
        default=None,
        help="disable loop detection",
    )
    parser.add_argument("--timeout", type=float, default=None, help="request timeout in seconds (overrides RUBIKBENCH_TIMEOUT)")
    parser.add_argument("--max-retries", type=int, default=None, help="retries per request (overrides RUBIKBENCH_MAX_RETRIES)")
    # Benchmark knobs.
    parser.add_argument("-n", "--solves", type=int, default=None, help="number of solves (overrides RUBIKBENCH_SOLVES)")
    parser.add_argument("--max-turns", type=int, default=None, help="turn budget per solve (overrides RUBIKBENCH_MAX_TURNS)")
    parser.add_argument("--scramble-len", type=int, default=None, help="scramble length (overrides RUBIKBENCH_SCRAMBLE_LEN)")
    parser.add_argument("--seed", type=int, default=None, help="fixed scramble seed (overrides RUBIKBENCH_SEED)")
    parser.add_argument("--cube-size", type=int, choices=(2, 3), default=None,
                        help="cube size: 3 = 3x3x3 (default), 2 = 2x2x2 (overrides RUBIKBENCH_CUBE_SIZE)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rubikbench",
        description="Benchmark LLMs solving a Rubik's cube via OpenAI-compatible tool calls.",
    )
    parser.add_argument("--version", action="version", version=f"rubikbench {__version__}")
    # Bare `rubikbench` = run a benchmark (reads .env when no flags are given).
    _add_run_args(parser)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a benchmark headlessly (no TUI)")
    _add_run_args(run)

    sub.add_parser("tui", help="launch the Textual TUI")

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
            f"error: config file not found: {path} (run without --config to use .env)",
            file=sys.stderr,
        )
        print("run `rubikbench --help` for all options", file=sys.stderr)
        return 2
    except ValueError as exc:  # surface config errors
        print(f"error: invalid config: {exc}", file=sys.stderr)
        return 2

    if args.profile is not None:
        cfg = apply_profile(cfg, args.profile)

    # CLI flags win over everything; only explicitly given flags are applied.
    cli_overrides = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "context_window_tokens": args.context_window_tokens,
        "output_token_reserve": args.output_token_reserve,
        "protocol_mode": args.protocol_mode,
        "presentation_mode": args.presentation_mode,
        "cube_size": args.cube_size,
        "scramble_preset": args.scramble_preset,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "top_k": args.top_k,
        "stream_idle_timeout": args.stream_idle_timeout,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "num_solves": args.solves,
        "max_turns": args.max_turns,
        "scramble_len": args.scramble_len,
        "seed": args.seed,
    }
    overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    if args.loop_detection is not None:
        overrides["loop_detection"] = args.loop_detection
    if overrides:
        cfg = replace(cfg, **overrides)
        try:
            cfg.validate()
        except ValueError as exc:  # surface config errors
            print(f"error: invalid config: {exc}", file=sys.stderr)
            return 2

    if not cfg.api_key and api_key_source(cfg.base_url):
        print(
            f"error: no API key set; add {api_key_source(cfg.base_url)}=... (or RUBIKBENCH_API_KEY=...) to .env, or pass --api-key",
            file=sys.stderr,
        )
        print("run `rubikbench --help` for all options", file=sys.stderr)
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
        top_p=cfg.top_p,
        max_output_tokens=cfg.max_output_tokens,
        stream_idle_timeout=cfg.stream_idle_timeout,
        loop_detection=cfg.loop_detection,
        extra_body=cfg.effective_extra_body(),
        tool_choice=cfg.tool_choice if cfg.tool_choice != "auto" else "auto",
    )

    # Tracks whether we are mid-stream so line-based messages start on a fresh
    # line instead of gluing onto the model's text.
    stream_state = {"active": False}

    # Cumulative token counters (delta-math, same as the TUI).
    tok_in = 0
    tok_out = 0
    tok_reasoning = 0
    tok_cached = 0
    tok_total = 0
    usage_by_turn: dict[int, dict[str, int]] = {}

    def add_usage(turn: int, usage: dict[str, int] | None) -> None:
        nonlocal tok_in, tok_out, tok_reasoning, tok_cached, tok_total
        if not usage:
            return
        prompt = usage.get("prompt", 0)
        completion = usage.get("completion", 0)
        reasoning = usage.get("reasoning", 0)
        cached = usage.get("cached", 0)
        total = usage.get("total", 0)
        if prompt == completion == reasoning == cached == total == 0:
            return
        prev = usage_by_turn.get(turn, {})
        delta_prompt = max(0, prompt - prev.get("prompt", 0))
        delta_completion = max(0, completion - prev.get("completion", 0))
        delta_reasoning = max(0, reasoning - prev.get("reasoning", 0))
        delta_cached = max(0, cached - prev.get("cached", 0))
        delta_total = max(0, total - prev.get("total", 0))
        delta_response = max(0, delta_completion - delta_reasoning)
        usage_by_turn[turn] = {"prompt": prompt, "completion": completion, "reasoning": reasoning, "cached": cached, "total": total}
        tok_in += delta_prompt
        tok_out += delta_response
        tok_reasoning += delta_reasoning
        tok_cached += delta_cached
        tok_total += delta_total

    def fmt(n: int) -> str:
        return f"{n:,}"

    def token_summary() -> str:
        parts = [f"ctx={fmt(tok_in)}", f"out={fmt(tok_out)}"]
        if tok_reasoning:
            parts.append(f"reason={fmt(tok_reasoning)}")
        if tok_cached:
            parts.append(f"cached={fmt(tok_cached)}")
        return " · ".join(parts)

    def end_stream_line() -> None:
        if stream_state["active"]:
            sys.stderr.write("\n")
            stream_state["active"] = False

    def emit(kind: str, payload: dict) -> None:
        nonlocal tok_in, tok_out, tok_reasoning, tok_cached, tok_total
        if kind == "solve_started":
            _print_progress_line(f"[solve {payload['index'] + 1}/{cfg.num_solves}] scramble: {payload['scramble']}", not args.no_color)
        elif kind == "turn_started":
            attempt = f" (retry {payload['attempt']})" if payload["attempt"] > 1 else ""
            _print_progress_line(f"  turn {payload['turn']}: waiting for model{attempt}...", not args.no_color)
        elif kind == "stream":
            add_usage(payload["turn"], payload.get("usage"))
            for text in (payload.get("reasoning"), payload.get("content")):
                if not text:
                    continue
                if not stream_state["active"]:
                    sys.stderr.write("  ")
                    stream_state["active"] = True
                sys.stderr.write(text)
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
            add_usage(
                payload["turn"],
                {
                    "prompt": payload.get("prompt_tokens", 0),
                    "completion": payload.get("completion_tokens", 0),
                    "reasoning": payload.get("reasoning_tokens", 0),
                    "cached": payload.get("cached_tokens", 0),
                    "total": payload.get("total_tokens", 0),
                },
            )
            what = f" · tools: {', '.join(payload['tool_call_names'])}" if payload["tool_call_names"] else ""
            _print_progress_line(
                f"  turn {payload['turn']} · {payload['latency']:.1f}s{what} · {token_summary()}", not args.no_color
            )
        elif kind == "solve_done":
            end_stream_line()
            s = payload["result"]
            status = "SOLVED" if s.solved else ("FAILED" if s.error else "UNSOLVED")
            extra = (
                f"moves={fmt(s.total_moves)} turns={s.turns} tools={s.tool_calls} "
                f"score={s.score} · {token_summary()}"
            )
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


def cmd_tui(args: argparse.Namespace) -> int:
    from .tui.app import RubikBenchApp

    try:
        app = RubikBenchApp()
    except Exception as exc:  # noqa: BLE001 - surface config errors before the UI starts
        print(f"error: {exc}", file=sys.stderr)
        print("run `rubikbench --help` for all options", file=sys.stderr)
        return 2
    return app.run()


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
    if args.command in (None, "run"):
        # Bare `rubikbench` and `rubikbench run` are the same: start the
        # benchmark immediately (from .env when no flags are given).
        return cmd_run(args)
    if args.command == "tui":
        return cmd_tui(args)
    if args.command == "presets":
        return cmd_presets()
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    if args.command == "view":
        return cmd_view(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
