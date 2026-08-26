"""Benchmark scoring.

A solve is scored 0-1000. Solving is a gate: unsolved runs score 0 regardless of
efficiency. For solved runs, three efficiency factors between 0 and 1 are blended
with configurable weights (defaults: moves 50%, turns 30%, tool calls 20%):

* ``moves_eff`` — how close the solution is to the reference ("par") length:
  ``min(1, par / moves)``. Par comes from kociemba when installed (near-optimal),
  otherwise God's number (20).
* ``turns_eff`` — how few LLM round trips were needed:
  ``min(1, (max_turns - turns + 1) / max_turns)`` (1 turn -> 1.0).
* ``tools_eff`` — tool-call discipline: ``min(1, turns / tool_calls)``
  (at most one tool call per turn is ideal).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Weights:
    moves: float = 0.5
    turns: float = 0.3
    tools: float = 0.2

    def validate(self) -> Weights:
        if any(w < 0 for w in (self.moves, self.turns, self.tools)):
            raise ValueError("score weights must be non-negative")
        total = self.moves + self.turns + self.tools
        if total <= 0:
            raise ValueError("score weights must sum to more than zero")
        if abs(total - 1.0) > 1e-6:
            # Normalize instead of rejecting, so users can mix weights freely.
            inv = 1.0 / total
            return Weights(self.moves * inv, self.turns * inv, self.tools * inv)
        return self


@dataclass
class ScoreBreakdown:
    solved: bool
    score: float
    moves_eff: float
    turns_eff: float
    tools_eff: float
    par_moves: int
    total_moves: int
    turns: int
    tool_calls: int
    weights: Weights
    reason: str


def compute_score(
    *,
    solved: bool,
    par_moves: int,
    total_moves: int,
    turns: int,
    tool_calls: int,
    max_turns: int,
    weights: Weights | None = None,
) -> ScoreBreakdown:
    w = (weights or Weights()).validate()

    if not solved:
        return ScoreBreakdown(
            solved=False, score=0.0, moves_eff=0.0, turns_eff=0.0, tools_eff=0.0,
            par_moves=par_moves, total_moves=total_moves, turns=turns,
            tool_calls=tool_calls, weights=w, reason="unsolved",
        )

    if total_moves <= 0:
        moves_eff = 1.0
    else:
        moves_eff = min(1.0, par_moves / total_moves)

    turns_eff = min(1.0, (max_turns - turns + 1) / max(1, max_turns))
    tools_eff = min(1.0, turns / max(1, tool_calls))

    score = 1000.0 * (w.moves * moves_eff + w.turns * turns_eff + w.tools * tools_eff)
    return ScoreBreakdown(
        solved=True, score=round(score, 2), moves_eff=round(moves_eff, 4),
        turns_eff=round(turns_eff, 4), tools_eff=round(tools_eff, 4),
        par_moves=par_moves, total_moves=total_moves, turns=turns,
        tool_calls=tool_calls, weights=w, reason="solved",
    )
