"""Optional Kociemba integration: reference solution lengths used for scoring.

Kociemba's solver uses the same facelet layout as this package but reports moves
with *inverted* semantics (its ``U`` is a counterclockwise turn of the U face in
the standard convention). Everything here translates between the two.
"""

from __future__ import annotations

GODS_NUMBER: int = 20

try:  # pragma: no cover - exercised through tests
    import kociemba as _kociemba

    HAS_KOCIEMBA = True
except Exception:  # noqa: BLE001 - pragma: no cover; optional dependency
    _kociemba = None
    HAS_KOCIEMBA = False


def invert_move(move: str) -> str:
    if move.endswith("2"):
        return move
    return move[:-1] if move.endswith("'") else move + "'"


def solve_standard(facelet_string: str) -> list[str] | None:
    """Return a solution in standard Singmaster semantics, or ``None`` if the
    kociemba solver is unavailable or the state is invalid."""
    if not HAS_KOCIEMBA:
        return None
    try:
        raw = _kociemba.solve(facelet_string)
    except ValueError:
        return None
    if not raw.strip():
        return []
    return [invert_move(m) for m in raw.split()]


def par_moves(facelet_string: str, is_solved: bool = False, default: int = GODS_NUMBER) -> int:
    """Best-effort reference ("par") move count for a state.

    Uses kociemba's near-optimal solver when available; otherwise falls back to
    God's number (20 moves for 3x3x3).
    """
    if is_solved:
        return 0
    solution = solve_standard(facelet_string)
    if solution is None:
        return default
    return len(solution)
