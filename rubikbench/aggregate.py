"""Dataset aggregation across RubikBench run files.

A run file is JSONL: one header line (``event: benchmark``) plus one line per
solve (``event: solve``). ``aggregate_files`` merges any number of run files
into one dataset document, denormalizing each solve record with its run
metadata so the result is a flat, queryable array.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_run(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a run JSONL file into ``(header, solve_records)``.

    Raises ValueError when the file is not a valid RubikBench run file.
    """
    header: dict[str, Any] | None = None
    solves: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "benchmark":
                header = record
            elif record.get("event") == "solve":
                solves.append(record)
            else:
                raise ValueError(f"unknown record in run file: {record.get('event')!r}")
    if header is None:
        raise ValueError(f"not a RubikBench run file (no benchmark header): {path}")
    return header, solves


def capability_frontier(solves: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Per-difficulty solve rate and move efficiency for ladder result records."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for solve in solves:
        label = str(solve.get("difficulty") or solve.get("scramble_preset") or "random")
        groups.setdefault(label, []).append(solve)
    frontier: dict[str, dict[str, float | int]] = {}
    for label, group in groups.items():
        won = [s for s in group if s.get("solved")]
        frontier[label] = {
            "solves": len(group), "solve_rate": round(len(won) / len(group), 3),
            "efficiency": round(sum((s.get("par") or 0) / max(1, s.get("total_moves", 0)) for s in won)
                                / len(won), 3) if won else 0.0,
        }
    return frontier


def aggregate_files(paths: list[str | Path]) -> dict[str, Any]:
    """Merge run files into one dataset document."""
    runs: list[dict[str, Any]] = []
    solves: list[dict[str, Any]] = []
    for path in paths:
        header, run_solves = read_run(path)
        config = dict(header.get("config", {}))
        config.pop("api_key", None)
        runs.append({
            "file": str(path),
            "started_at": header.get("started_at"),
            "duration": header.get("duration"),
            "model": config.get("model"),
            "base_url": config.get("base_url"),
            "config": config,
            "aggregates": header.get("aggregates", {}),
        })
        for record in run_solves:
            record = dict(record)
            record["run"] = str(path)
            record["model"] = header.get("config", {}).get("model")
            record["base_url"] = header.get("config", {}).get("base_url")
            solves.append(record)

    def total(key: str) -> int:
        return sum(s.get(key, 0) for s in solves if isinstance(s.get(key), int))

    return {
        "event": "dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": len(runs),
        "total_solves": len(solves),
        "models": sorted({r.get("model") for r in runs if r.get("model")}),
        "totals": {
            "solves": len(solves),
            "solved": sum(1 for s in solves if s.get("solved")),
            "solve_rate": round(
                sum(1 for s in solves if s.get("solved")) / len(solves), 3
            ) if solves else 0.0,
            "avg_score": round(
                sum(s.get("score", 0.0) for s in solves if s.get("solved"))
                / max(1, sum(1 for s in solves if s.get("solved"))),
                2,
            ),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "cached_tokens": total("cached_tokens"),
            "total_tokens": total("total_tokens"),
            "turns": total("turns"),
            "tool_calls": total("tool_calls"),
            "moves": total("total_moves"),
            "retries": total("retries"),
            "truncated": sum(1 for s in solves if s.get("truncated")),
            "frontier": capability_frontier(solves),
        },
        "runs_detail": runs,
        "solves": solves,
    }


def write_dataset(dataset: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=1) + "\n", encoding="utf-8")
    return path
