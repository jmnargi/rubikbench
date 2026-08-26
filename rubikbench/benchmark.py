"""Benchmark orchestration: tool execution, the solve loop, runner, results.

The turn loop drives an LLM through tool calls against a cube. One "turn" is one
chat completion round trip; the loop stops when the cube is solved, the turn
budget is exhausted, an unrecoverable API error occurs, or cancellation is
requested. Wall-clock time and token usage are recorded per turn.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .cube import Cube
from .llm import LLMClient, LLMError
from .prompts import SYSTEM_PROMPT, TOOLS, initial_user_prompt
from .rendering import render_plain
from .scoring import Weights, compute_score
from .scramble import (
    assert_valid_scramble,
    random_scramble,
    scramble_from_string,
    scramble_to_string,
)
from .solver_ref import GODS_NUMBER, par_moves

#: callback(kind: str, payload: dict) used to stream progress to a UI.
Emitter = Callable[[str, dict[str, Any]], None]


def _emit(emitter: Emitter | None, kind: str, **payload: Any) -> None:
    if emitter is not None:
        try:
            emitter(kind, payload)
        except Exception:  # noqa: BLE001,S110 - UI sinks must never break the run
            pass


# --------------------------------------------------------------------------- tools

class SolveContext:
    """Mutable state shared by the tool calls of one solve."""

    def __init__(self, scramble: list[str]) -> None:
        self.scramble = list(scramble)
        self.cube = Cube.solved()
        self.cube.scramble = self.scramble
        self.cube.reset_to_scramble()
        self.total_moves = 0
        self.resets = 0

    def state_text(self) -> str:
        cube = self.cube
        history = " ".join(cube.history[-25:]) or "(none)"
        return (
            f"Facelets (U R F D L B): {cube.facelet_string()}\n"
            f"Net:\n{render_plain(cube.facelets)}\n"
            f"Move history ({len(cube.history)}): {history}\n"
            f"Solved: {cube.is_solved()}"
        )


def execute_tool(name: str, arguments: dict[str, Any], ctx: SolveContext) -> str:
    cube = ctx.cube
    if name == "get_cube_state":
        return ctx.state_text()
    if name == "apply_moves":
        moves_text = str(arguments.get("moves", "") or "").strip()
        if not moves_text:
            return "No moves provided. Pass a space-separated move string, e.g. \"R U R' U'\"."
        from .cube import parse_moves

        valid, invalid = parse_moves(moves_text)
        if invalid:
            note = f"Warning: ignored invalid tokens: {invalid!r}."
        else:
            note = ""
        cube.apply(valid)
        ctx.total_moves += len(valid)
        applied = " ".join(valid) if valid else "(none)"
        solved = cube.is_solved()
        return (
            f"Applied {len(valid)} move(s): {applied}. {note}\n"
            f"You have applied {ctx.total_moves} move(s) total.\n"
            f"{ctx.state_text()}"
            + ("\n*** The cube is SOLVED. ***" if solved else "")
        )
    if name == "reset_cube":
        cube.reset_to_scramble()
        ctx.resets += 1
        return (
            f"Cube reset to the original scramble. {ctx.resets} reset(s) so far.\n"
            f"{ctx.state_text()}"
        )
    return f"Unknown tool: {name}"


# --------------------------------------------------------------------------- results

@dataclass
class SolveResult:
    index: int
    scramble: list[str]
    solved: bool
    turns: int
    tool_calls: int
    total_moves: int
    resets: int
    elapsed: float
    prompt_tokens: int
    completion_tokens: int
    par: int
    score: float
    breakdown: dict[str, Any]
    transcript: list[dict[str, Any]]
    error: str | None = None


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    solves: list[SolveResult]
    started_at_iso: str
    duration: float

    def aggregates(self) -> dict[str, Any]:
        n = len(self.solves)
        if n == 0:
            return {}
        solved = [s for s in self.solves if s.solved]
        scored = [s.score for s in solved]
        moves = [s.total_moves for s in solved] or [0]
        turns = [s.turns for s in solved] or [0]
        tools = [s.tool_calls for s in solved] or [0]
        times = [s.elapsed for s in solved] or [0]
        return {
            "solves": n,
            "solved": len(solved),
            "solve_rate": round(len(solved) / n, 3),
            "avg_score": round(sum(scored) / len(scored), 1) if scored else 0.0,
            "median_score": round(sorted(scored)[len(scored) // 2], 1) if scored else 0.0,
            "best_score": round(max(scored), 1) if scored else 0.0,
            "avg_moves": round(sum(moves) / len(moves), 2) if solved else 0.0,
            "avg_turns": round(sum(turns) / len(turns), 2) if solved else 0.0,
            "avg_tool_calls": round(sum(tools) / len(tools), 2) if solved else 0.0,
            "avg_time": round(sum(times) / len(times), 2) if solved else 0.0,
            "total_prompt_tokens": sum(s.prompt_tokens for s in self.solves),
            "total_completion_tokens": sum(s.completion_tokens for s in self.solves),
        }

# --------------------------------------------------------------------------- context trimming

def estimate_tokens(text: str | None) -> int:
    """Rough token estimate: about 4 characters per token (common heuristic)."""
    return max(1, math.ceil(len(text or "") / 4))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimated tokens for one message, including a small role/name overhead."""
    total = 6
    content = message.get("content")
    if content:
        total += estimate_tokens(content)
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        total += estimate_tokens(fn.get("arguments") or "") + 8
    return total


def trim_messages(
    messages: list[dict[str, Any]], max_input_tokens: int | None
) -> tuple[list[dict[str, Any]], bool]:
    """Return a copy of `messages` that fits roughly within `max_input_tokens`.

    The system message and the initial user message (indexes 0 and 1) are never
    removed: the initial user message carries the scramble and the initial cube
    state. Older conversation turns are dropped as complete units (an assistant
    message together with any tool results that follow it), oldest first.

    Returns ``(request_messages, trimmed)`` where ``trimmed`` is True when at
    least one turn was removed. A ``None`` cap disables trimming.
    """
    if not max_input_tokens:
        return list(messages), False
    if sum(estimate_message_tokens(m) for m in messages) <= max_input_tokens:
        return list(messages), False

    kept = list(messages[:2])  # system + initial user survive trimming
    units: list[list[dict[str, Any]]] = []
    i = 2
    while i < len(messages):
        unit = [messages[i]]
        if messages[i].get("role") == "assistant":
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                unit.append(messages[i])
                i += 1
        else:
            i += 1
        units.append(unit)

    dropped = False
    while units:
        tokens = sum(estimate_message_tokens(m) for m in kept) + sum(
            estimate_message_tokens(m) for u in units for m in u
        )
        if tokens <= max_input_tokens:
            break
        units.pop(0)
        dropped = True
    for unit in units:
        kept.extend(unit)
    return kept, dropped



# --------------------------------------------------------------------------- solve loop

def run_solve(
    index: int,
    scramble: list[str],
    cfg: BenchmarkConfig,
    client: LLMClient,
    emitter: Emitter | None = None,
    cancel_event: threading.Event | None = None,
) -> SolveResult:
    ctx = SolveContext(scramble)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_prompt(ctx.cube)},
    ]
    transcript: list[dict[str, Any]] = []
    started = time.monotonic()
    turns = 0
    tool_calls_total = 0
    prompt_tokens = 0
    completion_tokens = 0
    solved = ctx.cube.is_solved()
    error: str | None = None

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    _emit(emitter, "solve_started", index=index, scramble=scramble_to_string(scramble), par=0)

    while turns < cfg.max_turns and not solved:
        if cancelled():
            error = "aborted by user"
            break

        attempt = 0
        while True:
            if cancelled():
                error = "aborted by user"
                break
            attempt += 1
            request_messages, trimmed = trim_messages(messages, cfg.max_input_tokens)
            if trimmed:
                _emit(emitter, "log", index=index, level="warn",
                      message=f"Turn {turns + 1}: trimmed conversation history to fit max input tokens ({cfg.max_input_tokens}).")
            try:
                turn = client.complete(request_messages, TOOLS, cfg.effective_extra_body())
                break
            except LLMError as exc:
                _emit(emitter, "log", index=index, level="error",
                      message=f"API error (attempt {attempt}): {exc}")
                if attempt > cfg.max_retries:
                    error = f"API error after {attempt} attempts: {exc}"
                    break
                time.sleep(min(2 ** attempt, 15))
        if cancelled():
            error = error or "aborted by user"
            break
        if error:
            break

        turns += 1
        prompt_tokens += turn.prompt_tokens
        completion_tokens += turn.completion_tokens
        transcript.append({
            "turn": turns, "role": "assistant", "content": turn.content,
            "tool_calls": [{"name": tc.name, "id": tc.id, "arguments": tc.arguments} for tc in turn.tool_calls],
            "latency": round(turn.latency, 3),
        })
        _emit(emitter, "turn", index=index, turn=turns, content=turn.content,
              tool_call_names=[tc.name for tc in turn.tool_calls], latency=turn.latency)

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if turn.content:
            assistant_msg["content"] = turn.content
        if turn.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in turn.tool_calls
            ]
        messages.append(assistant_msg)

        if turn.tool_calls:
            for tc in turn.tool_calls:
                if cancelled():
                    error = error or "aborted by user"
                    break
                tool_calls_total += 1
                result_text = execute_tool(tc.name, tc.arguments, ctx)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                transcript.append({
                    "turn": turns, "role": "tool", "name": tc.name,
                    "arguments": tc.arguments, "content": result_text,
                })
                _emit(emitter, "state", index=index, facelets=ctx.cube.facelets,
                      history=list(ctx.cube.history), total_moves=ctx.total_moves,
                      turns=turns, tool_calls=tool_calls_total, solved=ctx.cube.is_solved())
                if ctx.cube.is_solved():
                    solved = True
                    break
        else:
            text = (turn.content or "").strip()
            if cfg.allow_text_moves and text and not solved:
                valid, invalid = ctx.cube.apply_text(text)
                if valid:
                    ctx.total_moves += len(valid)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[system] Parsed {len(valid)} move(s) from your text and applied them: "
                            f"{' '.join(valid)}. Ignored: {invalid or 'none'}.\n{ctx.state_text()}"
                        ),
                    })
                    if ctx.cube.is_solved():
                        solved = True
                elif not any(k in text.lower() for k in ("error", "sorry", "fix", "retry", "invalid")):
                    _emit(emitter, "log", index=index, level="warn",
                          message=f"Turn {turns}: no moves or tool call; turn wasted.")
                _emit(emitter, "state", index=index, facelets=ctx.cube.facelets,
                      history=list(ctx.cube.history), total_moves=ctx.total_moves,
                      turns=turns, tool_calls=tool_calls_total, solved=ctx.cube.is_solved())
        if solved:
            break

    elapsed = time.monotonic() - started

    par = _par_for(scramble, cfg) if solved else 0
    breakdown = compute_score(
        solved=solved,
        par_moves=par if par else cfg.par_fixed,
        total_moves=ctx.total_moves,
        turns=turns,
        tool_calls=tool_calls_total,
        max_turns=cfg.max_turns,
        weights=Weights(cfg.weight_moves, cfg.weight_turns, cfg.weight_tools),
    )
    result = SolveResult(
        index=index,
        scramble=list(scramble),
        solved=solved,
        turns=turns,
        tool_calls=tool_calls_total,
        total_moves=ctx.total_moves,
        resets=ctx.resets,
        elapsed=round(elapsed, 3),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        par=par,
        score=breakdown.score,
        breakdown={
            "score": breakdown.score,
            "solved": breakdown.solved,
            "moves_eff": breakdown.moves_eff,
            "turns_eff": breakdown.turns_eff,
            "tools_eff": breakdown.tools_eff,
            "par_moves": breakdown.par_moves,
            "total_moves": breakdown.total_moves,
            "turns": breakdown.turns,
            "tool_calls": breakdown.tool_calls,
            "weights": {
                "moves": breakdown.weights.moves,
                "turns": breakdown.weights.turns,
                "tools": breakdown.weights.tools,
            },
            "reason": breakdown.reason,
        },
        transcript=transcript,
        error=error,
    )
    _emit(emitter, "solve_done", result=result)
    return result


def _par_for(scramble: list[str], cfg: BenchmarkConfig) -> int:
    """Reference move count for the scrambled state."""
    if cfg.par_strategy == "fixed":
        return cfg.par_fixed
    if cfg.par_strategy == "god":
        return GODS_NUMBER
    cube = Cube.solved()
    cube.scramble = scramble
    cube.reset_to_scramble()
    return par_moves(cube.facelet_string(), is_solved=False, default=GODS_NUMBER)


# --------------------------------------------------------------------------- runner

class BenchmarkRunner:
    """Runs a configurable number of solves, streaming progress events."""

    def __init__(self, cfg: BenchmarkConfig, client: LLMClient, emitter: Emitter | None = None) -> None:
        self.cfg = cfg
        self.client = client
        self.emitter = emitter

    def run(self, cancel_event: threading.Event | None = None) -> BenchmarkResult:
        started_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        rng = random.Random(self.cfg.seed)
        fixed = self._resolve_fixed_scrambles()
        solves: list[SolveResult] = []
        for i in range(self.cfg.num_solves):
            if cancel_event is not None and cancel_event.is_set():
                break
            if fixed is not None:
                scramble = fixed[i % len(fixed)]
            else:
                scramble = random_scramble(rng, self.cfg.scramble_len)
            result = run_solve(i, scramble, self.cfg, self.client, self.emitter, cancel_event)
            solves.append(result)
        duration = time.monotonic() - t0
        result = BenchmarkResult(config=self.cfg, solves=solves, started_at_iso=started_iso, duration=duration)
        _emit(self.emitter, "bench_done", result=result)
        return result

    def _resolve_fixed_scrambles(self) -> list[list[str]] | None:
        """Return the fixed scramble list (as parsed move lists), or None for random.

        Resolution: a premade set name wins; ``"file"`` (or a legacy non-empty
        ``scrambles`` list) uses the custom list. Every returned scramble is
        parsed and verified up front so bad custom files fail before any solve.
        """
        if self.cfg.scramble_preset == "file" or (
            self.cfg.scramble_preset == "" and self.cfg.scrambles
        ):
            parsed = [scramble_from_string(s) for s in self.cfg.scrambles]
        elif self.cfg.scramble_preset:
            from .scramble import PREMADE_SCRAMBLES

            parsed = [
                scramble_from_string(s) for s in PREMADE_SCRAMBLES[self.cfg.scramble_preset]
            ]
        else:
            return None
        for moves in parsed:
            assert_valid_scramble(moves)
        return parsed


# --------------------------------------------------------------------------- export

def export_jsonl(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        header = {
            "event": "benchmark",
            "started_at": result.started_at_iso,
            "duration": round(result.duration, 3),
            "config": result.config.to_dict(),
            "aggregates": result.aggregates(),
        }
        fh.write(json.dumps(header) + "\n")
        fh.writelines(json.dumps(solve_to_dict(solve)) + "\n" for solve in result.solves)
    return path


def solve_to_dict(solve: SolveResult) -> dict[str, Any]:
    return {
        "event": "solve",
        "index": solve.index,
        "scramble": scramble_to_string(solve.scramble),
        "solved": solve.solved,
        "turns": solve.turns,
        "tool_calls": solve.tool_calls,
        "total_moves": solve.total_moves,
        "resets": solve.resets,
        "elapsed": solve.elapsed,
        "prompt_tokens": solve.prompt_tokens,
        "completion_tokens": solve.completion_tokens,
        "par": solve.par,
        "score": solve.score,
        "breakdown": solve.breakdown,
        "error": solve.error,
        "transcript": solve.transcript,
    }
