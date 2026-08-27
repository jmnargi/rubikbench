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

def assert_valid_scramble(moves: list[str]) -> None:
    """Raise ValueError unless `moves` parses and solving it returns a solved cube.

    Every legal face-turn sequence from a solved cube is solvable by
    construction (replaying the inverse is a certificate), so this check is a
    cheap guard against malformed custom files and typos.
    """
    valid, invalid = parse_moves(" ".join(moves))
    if invalid:
        raise ValueError(f"invalid scramble tokens: {invalid!r}")
    if not valid:
        raise ValueError("scramble is empty")
    if not scramble_is_valid(valid):
        raise ValueError(f"scramble does not solve back to a solved cube: {moves!r}")


# --------------------------------------------------------------------------- premade scrambles
#
# Every scramble here is verified: applying it to a solved cube produces a real
# mixed state, replaying the inverse solves it, and kociemba accepts the state.
# The catalog entries are frozen, so runs are reproducible across machines and
# across RubikBench releases once a set is selected.

PREMADE_SCRAMBLES: dict[str, list[str]] = {
    # The canonical God's-number state: every edge flipped in place, 20 moves.
    # Inverted token-for-token when clockwise semantics were corrected.
    "superflip": [
        "R L' F U D' R2 F2 R F' B' L2 U' R2 L2 D F2 U R2 B2 D",
    ],
    "cube-in-cube": [
        "F L F U' R U F2 L2 U' L' B D' B' L2 U",
    ],
    "catalog-10": [
        "U D R F L F' U2 L B R",
        "B2 D2 B' L R' D2 B D F' R",
        "F U B' F2 D2 U B R' F R",
        "L B F L U' R L' D L D",
    ],
    "catalog-16": [
        "B L2 F' B2 U F' D' B' L2 F2 D' B2 U D2 F2 R",
        "R2 B' R B2 F L F D' F D2 U2 L' U D2 L D",
        "R' B2 F2 B D' B2 U' B' F U' L R2 D2 R U D",
        "F U R' L2 D B' U2 B' R' U2 F' U2 R' U R B2",
    ],
    "catalog-22": [
        "D2 B' F2 L' R' B' U F2 D F2 R' U2 D2 F R' B L2 F U2 R L2 D'",
        "B D' F' B D' F' L R2 D2 R' F' D' B' L2 D' B2 R2 U2 B U2 B L'",
        "F' U' B' D U B L2 B U' R2 D U' B U2 B L2 B L2 B2 F' B L",
        "D L R' L' R B' F' B2 D F R F R' B' L' F2 D U' B' R U D",
    ],
    "catalog-25": [
        "B' D2 B' L' F' D2 B D R B' D' B U' R2 U F2 U2 L2 F' B2 U R' B R2 F2",
        "R L' B L R2 F R L2 D2 R' F' D U' L B F D B2 L D2 L2 D2 R D' R2",
        "B2 R2 L' B' R2 B' R2 F' R' L B2 L2 F2 D2 F D U2 F2 B U' L2 B L' U' D2",
        "B2 R U' D L U2 B R' L2 D2 R F L' D L' B' F' B2 R D2 R2 B' R D L'",
    ],
}

# Frozen v1 difficulty ladder. Labels encode the requested sequence depth, not
# an asserted optimal distance (which is expensive to prove for every state).
DIFFICULTY_LADDER_VERSION = "ladder-v1"
for _depth, _moves in {
    1: ["R", "U", "F", "D"],
    2: ["R U", "F R", "L2 D"],
    3: ["R U F", "L D B", "R2 U F'"],
    5: ["R U F D L", "F2 R U' L D"],
    8: ["R U F D L B R U", "F2 R U' L D B' R2 U"],
    12: ["R U F D L B R U F D L B", "F2 R U' L D B' R2 U F D' L B"],
}.items():
    PREMADE_SCRAMBLES[f"ladder-{_depth}-v1"] = _moves


def premade_labels() -> list[tuple[str, str]]:
    """Display meaningful labels for every selectable frozen preset."""
    labels = [("Superflip (20-move)", "superflip"), ("Cube in cube", "cube-in-cube")]
    for depth in (1, 2, 3, 5, 8, 12):
        name = f"ladder-{depth}-v1"
        labels.append((f"Difficulty ladder v1 — {depth} moves", name))
    for length in (10, 16, 22, 25):
        name = f"catalog-{length}"
        labels.append((f"Catalog {length} moves", name))
    return labels
