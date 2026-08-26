"""Scramble generation and parsing."""

from __future__ import annotations

import random

from .cube import Cube, parse_moves

_FACES = "URFDLB"


def random_scramble(rng: random.Random, length: int = 22) -> list[str]:
    """Generate a scramble with no immediate same-face repetitions."""
    if length < 0:
        raise ValueError("scramble length must be >= 0")
    moves: list[str] = []
    prev_face: str | None = None
    while len(moves) < length:
        face = rng.choice(_FACES)
        if face == prev_face:
            continue
        suffix = rng.choice(("", "'", "2"))
        moves.append(face + suffix)
        prev_face = face
    return moves


def scramble_from_string(text: str) -> list[str]:
    """Parse a scramble from a move string, e.g. ``"R U R' U'"``."""
    valid, invalid = parse_moves(text)
    if invalid:
        raise ValueError(f"invalid scramble tokens: {invalid!r}")
    return valid


def scramble_to_string(moves: list[str]) -> str:
    return " ".join(moves)


def solution_for_scramble(scramble: list[str]) -> list[str]:
    """Return the exact inverse of a scramble: replaying it solves the cube."""
    solution: list[str] = []
    for m in reversed(scramble):
        if m.endswith("'"):
            solution.append(m[:-1])
        elif m.endswith("2"):
            solution.append(m)
        else:
            solution.append(m + "'")
    return solution


def scramble_is_valid(moves: list[str]) -> bool:
    """Sanity check: replaying the inverse of a scramble returns to solved."""
    cube = Cube.solved()
    cube.scramble = moves
    cube.reset_to_scramble()
    for m in solution_for_scramble(moves):
        cube.apply([m])
    return cube.is_solved()
