"""Rubik's cube model for 2x2x2 and 3x3x3 cubes.

Facelet representation: 6 faces in Kociemba order U R F D L B, each face an
``n x n`` grid (``n = size``), so a 3x3x3 cube has 54 facelets and a 2x2x2
cube has 24. For 3x3x3 the internal layout matches Kociemba's solver; the
printed net is the classic cross::

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
This avoids hand-written permutation tables (and the errors that come with
them) and works for any face count ``n >= 2``.

Only the 18 legal Singmaster face turns exist (``U R F D L B`` with ``'`` and
``2`` suffixes). There is no code path that can apply a different turn: a
2x2x2 and a 3x3x3 cube have exactly the same 18 legal face turns.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

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

_SUPPORTED_SIZES = (2, 3)


def _step(size: int) -> float:
    return 2.0 / size


def _sticker_positions(size: int) -> dict[int, tuple[float, float, float]]:
    """Facelet index -> 3D position of its center on the [-1, 1]^3 cube."""
    step = _step(size)
    pos: dict[int, tuple[float, float, float]] = {}
    for fi, face in enumerate(FACES):
        normal = _NORMALS[face]
        r_ax, d_ax = _FRAME[face]
        for row in range(size):
            for col in range(size):
                idx = fi * size * size + row * size + col
                dc = (col - (size - 1) / 2) * step
                dr = (row - (size - 1) / 2) * step
                pos[idx] = (
                    normal[0] + dc * r_ax[0] + dr * d_ax[0],
                    normal[1] + dc * r_ax[1] + dr * d_ax[1],
                    normal[2] + dc * r_ax[2] + dr * d_ax[2],
                )
    return pos


_POSITIONS = _sticker_positions(3)


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


def _index_at(pos: tuple[float, float, float], size: int) -> int:
    """Map a 3D position back to its facelet index (nearest-sticker mapping)."""
    axis = max(range(3), key=lambda i: abs(pos[i]))
    face = _FACE_AT_AXIS[axis]["+" if pos[axis] > 0 else "-"]
    fi = FACES.index(face)
    normal = _NORMALS[face]
    r_ax, d_ax = _FRAME[face]
    off = (pos[0] - normal[0], pos[1] - normal[1], pos[2] - normal[2])
    dc = off[0] * r_ax[0] + off[1] * r_ax[1] + off[2] * r_ax[2]
    dr = off[0] * d_ax[0] + off[1] * d_ax[1] + off[2] * d_ax[2]
    step = _step(size)

    def nearest(d: float) -> int:
        best = 0
        best_dist = abs(d - (0 - (size - 1) / 2) * step)
        for c in range(1, size):
            dist = abs(d - (c - (size - 1) / 2) * step)
            if dist < best_dist - 1e-12:
                best, best_dist = c, dist
        return best

    return fi * size * size + nearest(dr) * size + nearest(dc)


def _build_perm(face: str, angle: float, size: int) -> tuple[int, ...]:
    """Build a permutation table ``perm[i] == j`` meaning ``after[i] = before[j]``."""
    normal = _NORMALS[face]
    positions = _sticker_positions(size)
    total = size * size * 6
    perm = list(range(total))
    for j, p in positions.items():
        in_layer = p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] >= 0.5 - 1e-9
        if in_layer:
            rotated = _rotate(p, normal, angle)
            perm[_index_at(rotated, size)] = j
    return tuple(perm)


def _compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Permutation composition: apply `b` first, then `a`."""
    return tuple(a[b[i]] for i in range(len(a)))


def _build_moves(size: int) -> dict[str, tuple[int, ...]]:
    moves: dict[str, tuple[int, ...]] = {}
    for face in FACES:
        cw = _build_perm(face, 90.0, size)    # clockwise when viewed at the face
        ccw = _build_perm(face, -90.0, size)
        double = _compose(cw, cw)
        moves[face] = cw
        moves[face + "'"] = ccw
        moves[face + "2"] = double
    return moves


_MOVES_BY_SIZE: dict[int, dict[str, tuple[int, ...]]] = {3: _build_moves(3)}


def moves_for(size: int) -> dict[str, tuple[int, ...]]:
    """The 18 move permutation tables for a cube of `size` (cached)."""
    if size not in _MOVES_BY_SIZE:
        _MOVES_BY_SIZE[size] = _build_moves(size)
    return _MOVES_BY_SIZE[size]


#: Backward-compatible alias: the 3x3x3 tables (54 facelets).
MOVES: dict[str, tuple[int, ...]] = _MOVES_BY_SIZE[3]

#: All 18 legal Singmaster face turns. Identical for 2x2x2 and 3x3x3.
ALL_MOVES: list[str] = [f + suffix for f in "URFDLB" for suffix in ("", "'", "2")]

_TOKEN_RE = re.compile(r"^[URFDLB][2']?$")


def parse_moves(text: str) -> tuple[list[str], list[str]]:
    """Parse whitespace-separated Singmaster moves from `text`.

    Returns ``(valid, invalid)``. Case-insensitive; unknown tokens (including
    slice or middle-layer moves such as ``M``, ``E``, ``S``, ``u``, ``Uw``)
    are reported as invalid rather than raising. Only face turns ``U D L R F B``
    with an optional ``'`` or ``2`` suffix are ever accepted, for any cube size.
    """
    valid: list[str] = []
    invalid: list[str] = []
    for raw in text.split():
        tok = raw.strip().upper()
        if _TOKEN_RE.match(tok) and tok in ALL_MOVES:
            valid.append(tok)
        else:
            invalid.append(raw)
    return valid, invalid


def moves_to_string(moves: list[str]) -> str:
    return " ".join(moves)


def solved_facelets(size: int) -> list[str]:
    """The solved facelet list for a cube of `size` (e.g. 24 for 2, 54 for 3)."""
    return list("".join(f * size * size for f in FACES))


#: Backward-compatible alias: the 3x3x3 solved state.
SOLVED_FACELETS = solved_facelets(3)


# Kociemba's position order and facelet indices (3x3x3 only). Piece identities
# are their solved color names; orientation is the standard cubie orientation
# at the current position.
_CORNER_NAMES = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
_CORNER_FACELETS = (
    (8, 9, 20), (6, 18, 38), (0, 36, 47), (2, 45, 11),
    (29, 26, 15), (27, 44, 24), (33, 53, 42), (35, 17, 51),
)
_CORNER_COLORS = tuple(tuple(name) for name in _CORNER_NAMES)
_EDGE_NAMES = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")
_EDGE_FACELETS = (
    (5, 10), (7, 19), (3, 37), (1, 46), (32, 16), (28, 25),
    (30, 43), (34, 52), (23, 12), (21, 41), (50, 39), (48, 14),
)
_EDGE_COLORS = tuple(tuple(name) for name in _EDGE_NAMES)


@dataclass
class Cube:
    """Mutable cube state plus a record of every move applied to it.

    ``size`` is the face count (2 or 3); ``facelets`` has ``6 * size * size``
    entries. Any attempt to build a cube whose facelet count does not match its
    size raises ValueError.
    """

    facelets: list[str] = field(default_factory=lambda: list(SOLVED_FACELETS))
    history: list[str] = field(default_factory=list)
    scramble: list[str] = field(default_factory=list)
    size: int = 3

    def __post_init__(self) -> None:
        expected = 6 * self.size * self.size
        if len(self.facelets) != expected:
            raise ValueError(
                f"size {self.size} cube needs {expected} facelets, got {len(self.facelets)}"
            )
        if self.size not in _SUPPORTED_SIZES:
            raise ValueError(f"unsupported cube size: {self.size} (use 2 or 3)")

    @classmethod
    def solved(cls, size: int = 3) -> Cube:
        return cls(list(solved_facelets(size)), size=size)

    def reset_to_scramble(self) -> None:
        self.facelets = list(solved_facelets(self.size))
        tables = moves_for(self.size)
        for m in self.scramble:
            self.facelets = [self.facelets[p] for p in tables[m]]
        self.history = []

    def apply(self, moves: list[str], record: bool = True) -> None:
        """Apply validated move tokens in sequence.

        Only the 18 legal face turns are known to the move tables; any other
        token raises KeyError, so an illegal move can never alter the state.
        """
        tables = moves_for(self.size)
        for m in moves:
            perm = tables[m]
            self.facelets = [self.facelets[p] for p in perm]
            if record:
                self.history.append(m)

    def apply_text(self, text: str, record: bool = True) -> tuple[list[str], list[str]]:
        """Parse and apply moves found in `text`. Returns `(applied, invalid)`."""
        valid, invalid = parse_moves(text)
        self.apply(valid, record=record)
        return valid, invalid

    def is_solved(self) -> bool:
        per_face = self.size * self.size
        for fi in range(6):
            base = fi * per_face
            color = self.facelets[base]
            for i in range(base + 1, base + per_face):
                if self.facelets[i] != color:
                    return False
        return True

    def facelet_string(self) -> str:
        return "".join(self.facelets)

    def copy(self) -> Cube:
        c = Cube(list(self.facelets), list(self.history), list(self.scramble), size=self.size)
        return c

    def cubie_state(self) -> dict[str, Any]:
        """Derived Kociemba-style cubie inventory (defined for 3x3x3 only)."""
        if self.size != 3:
            raise ValueError("cubie state is only defined for a 3x3x3 cube")
        corners: list[dict[str, Any]] = []
        for position, indices in zip(_CORNER_NAMES, _CORNER_FACELETS, strict=True):
            colors = tuple(self.facelets[index] for index in indices)
            orientation = next(i for i, color in enumerate(colors) if color in "UD")
            identity = next(
                "".join(piece)
                for piece in _CORNER_COLORS
                if piece[1] == colors[(orientation + 1) % 3]
                and piece[2] == colors[(orientation + 2) % 3]
            )
            corners.append(
                {"position": position, "piece": identity, "orientation": orientation}
            )

        edges: list[dict[str, Any]] = []
        for position, indices in zip(_EDGE_NAMES, _EDGE_FACELETS, strict=True):
            colors = tuple(self.facelets[index] for index in indices)
            identity = orientation = None
            for candidate, piece in zip(_EDGE_NAMES, _EDGE_COLORS, strict=True):
                if colors == piece:
                    identity, orientation = candidate, 0
                    break
                if colors == (piece[1], piece[0]):
                    identity, orientation = candidate, 1
                    break
            if identity is None:
                raise ValueError("facelets do not describe a legal edge cubie")
            edges.append(
                {"position": position, "piece": identity, "orientation": orientation}
            )
        return {
            "version": "cubie-v1",
            "orientation_convention": (
                "Corners: orientation is the index (0, 1, 2) of the U/D sticker "
                "in the listed position order. Edges: 0 preserves the piece-name "
                "color order and 1 reverses it."
            ),
            "legal": True,
            "corners": corners,
            "edges": edges,
        }
