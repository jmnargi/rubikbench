"""3x3x3 Rubik's cube model.

Facelet representation with an internal layout identical to Kociemba's solver:
54 facelets, faces in order U R F D L B (indices ``face*9 + row*3 + col``), each
face printed in the classic cross net::

            U0 U1 U2
            U3 U4 U5
            U6 U7 U8
    L0 L1 L2 F0 F1 F2 R0 R1 R2 B0 B1 B2
    L3 L4 L5 F3 F4 F5 R3 R4 R5 B3 B4 B5
    L6 L7 L8 F6 F7 F8 R6 R7 R8 B6 B7 B8
            D0 D1 D2
            D3 D4 D5
            D6 D7 D8

The face-turn permutation tables are derived *geometrically*: each facelet is
assigned a 3D position on a unit cube, a move rotates the stickers of its layer
about the face normal, and the resulting positions are mapped back to facelets.
This avoids hand-written permutation tables (and the errors that come with them).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

FACES = "URFDLB"

# Outward unit normal of each face. Axes: x right, y front, z up.
_NORMALS: dict[str, tuple[float, float, float]] = {
    "U": (0.0, 0.0, 1.0),
    "D": (0.0, 0.0, -1.0),
    "F": (0.0, 1.0, 0.0),
    "B": (0.0, -1.0, 0.0),
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
}

# Tangent frame per face: (right_axis, down_axis) matching the printed net.
_FRAME: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "U": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),    # right = R, down = F
    "D": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),   # right = R, down = back
    "F": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),   # right = R, down = D
    "B": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),  # right = L, down = D
    "R": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),  # right = B, down = D
    "L": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),   # right = F, down = D
}

_FACE_AT_AXIS = {0: {"+": "R", "-": "L"}, 1: {"+": "F", "-": "B"}, 2: {"+": "U", "-": "D"}}

_CELL = 2.0 / 3.0  # facelet center offset on the unit cube


def _sticker_positions() -> dict[int, tuple[float, float, float]]:
    """Facelet index -> 3D position of its center on the [-1, 1]^3 cube."""
    pos: dict[int, tuple[float, float, float]] = {}
    for fi, face in enumerate(FACES):
        n = _NORMALS[face]
        r_ax, d_ax = _FRAME[face]
        for row in range(3):
            for col in range(3):
                idx = fi * 9 + row * 3 + col
                pos[idx] = (
                    n[0] + (col - 1) * _CELL * r_ax[0] + (row - 1) * _CELL * d_ax[0],
                    n[1] + (col - 1) * _CELL * r_ax[1] + (row - 1) * _CELL * d_ax[1],
                    n[2] + (col - 1) * _CELL * r_ax[2] + (row - 1) * _CELL * d_ax[2],
                )
    return pos


_POSITIONS = _sticker_positions()


def _rotate(v: tuple[float, float, float], axis: tuple[float, float, float], theta_deg: float) -> tuple[float, float, float]:
    """Rotate vector `v` about `axis` by `theta_deg` (right-hand positive) via Rodrigues."""
    rad = math.radians(theta_deg)
    c, s = math.cos(rad), math.sin(rad)
    x, y, z = v
    ux, uy, uz = axis
    dot = x * ux + y * uy + z * uz
    cx = uy * z - uz * y
    cy = uz * x - ux * z
    cz = ux * y - uy * x
    return (
        x * c + cx * s + ux * dot * (1 - c),
        y * c + cy * s + uy * dot * (1 - c),
        z * c + cz * s + uz * dot * (1 - c),
    )


def _index_at(pos: tuple[float, float, float]) -> int:
    """Map a 3D position back to its facelet index."""
    axis = max(range(3), key=lambda i: abs(pos[i]))
    face = _FACE_AT_AXIS[axis]["+" if pos[axis] > 0 else "-"]
    fi = FACES.index(face)
    normal = _NORMALS[face]
    r_ax, d_ax = _FRAME[face]
    off = (pos[0] - normal[0], pos[1] - normal[1], pos[2] - normal[2])
    dc = off[0] * r_ax[0] + off[1] * r_ax[1] + off[2] * r_ax[2]
    dr = off[0] * d_ax[0] + off[1] * d_ax[1] + off[2] * d_ax[2]
    col = min(2, max(0, round(dc / _CELL) + 1))
    row = min(2, max(0, round(dr / _CELL) + 1))
    return fi * 9 + row * 3 + col


def _build_perm(face: str, angle: float) -> tuple[int, ...]:
    """Build a permutation table ``perm[i] == j`` meaning ``after[i] = before[j]``."""
    normal = _NORMALS[face]
    perm = list(range(54))
    for j, p in _POSITIONS.items():
        in_layer = p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] > 0.5
        if in_layer:
            rotated = _rotate(p, normal, angle)
            perm[_index_at(rotated)] = j
    return tuple(perm)


def _compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Permutation composition: apply `b` first, then `a`."""
    return tuple(a[b[i]] for i in range(54))


def _build_moves() -> dict[str, tuple[int, ...]]:
    moves: dict[str, tuple[int, ...]] = {}
    for face in FACES:
        cw = _build_perm(face, -90.0)   # clockwise when viewed at the face
        ccw = _build_perm(face, 90.0)
        double = _compose(cw, cw)
        moves[face] = cw
        moves[face + "'"] = ccw
        moves[face + "2"] = double
    return moves


MOVES: dict[str, tuple[int, ...]] = _build_moves()

#: All 18 legal Singmaster face turns.
ALL_MOVES: list[str] = [f + suffix for f in "URFDLB" for suffix in ("", "'", "2")]

_TOKEN_RE = re.compile(r"^[URFDLB][2']?$")


def parse_moves(text: str) -> tuple[list[str], list[str]]:
    """Parse whitespace-separated Singmaster moves from `text`.

    Returns ``(valid, invalid)``. Case-insensitive; unknown tokens are reported
    as invalid rather than raising.
    """
    valid: list[str] = []
    invalid: list[str] = []
    for raw in text.split():
        tok = raw.strip().upper()
        if _TOKEN_RE.match(tok) and tok in MOVES:
            valid.append(tok)
        else:
            invalid.append(raw)
    return valid, invalid


def moves_to_string(moves: list[str]) -> str:
    return " ".join(moves)

SOLVED_FACELETS = list("".join(f * 9 for f in FACES))


@dataclass
class Cube:
    """Mutable cube state plus a record of every move applied to it."""

    facelets: list[str] = field(default_factory=lambda: list(SOLVED_FACELETS))
    history: list[str] = field(default_factory=list)
    scramble: list[str] = field(default_factory=list)

    @classmethod
    def solved(cls) -> Cube:
        return cls(list(SOLVED_FACELETS))

    def reset_to_scramble(self) -> None:
        self.facelets = list(SOLVED_FACELETS)
        for m in self.scramble:
            self.facelets = [self.facelets[p] for p in MOVES[m]]
        self.history = []

    def apply(self, moves: list[str], record: bool = True) -> None:
        """Apply validated move tokens in sequence."""
        for m in moves:
            perm = MOVES[m]
            self.facelets = [self.facelets[p] for p in perm]
            if record:
                self.history.append(m)

    def apply_text(self, text: str, record: bool = True) -> tuple[list[str], list[str]]:
        """Parse and apply moves found in `text`. Returns `(applied, invalid)`."""
        valid, invalid = parse_moves(text)
        self.apply(valid, record=record)
        return valid, invalid

    def is_solved(self) -> bool:
        for fi in range(6):
            base = fi * 9
            color = self.facelets[base]
            for i in range(base + 1, base + 9):
                if self.facelets[i] != color:
                    return False
        return True

    def facelet_string(self) -> str:
        return "".join(self.facelets)

    def copy(self) -> Cube:
        c = Cube(list(self.facelets), list(self.history), list(self.scramble))
        return c
