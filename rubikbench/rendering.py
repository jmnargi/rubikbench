"""ASCII / colored rendering of the cube net, for 2x2x2 and 3x3x3 cubes."""

from __future__ import annotations

import math

from rich.text import Text

# Face letter -> vivid hex background color for each sticker.
# These render as solid colored blocks even on dark / 256-color terminals.
FACE_COLORS = {
    "U": "#FFFFFF",  # white
    "R": "#FF3333",  # red
    "F": "#00E000",  # green
    "D": "#FFFF33",  # yellow
    "L": "#FF8800",  # orange
    "B": "#3366FF",  # blue
}

# Human-readable color name for each face.
FACE_NAMES = {
    "U": "white",
    "R": "red",
    "F": "green",
    "D": "yellow",
    "L": "orange",
    "B": "blue",
}

FACES_ORDER = "URFDLB"


def cube_size(facelets: list[str]) -> int:
    """Face count (2 or 3) for a facelet list; raises ValueError on garbage."""
    if len(facelets) % 6 != 0:
        raise ValueError(f"facelet count must be a multiple of 6, got {len(facelets)}")
    size = math.isqrt(len(facelets) // 6)
    if size * size * 6 != len(facelets) or size not in (2, 3):
        raise ValueError(f"unsupported facelet count: {len(facelets)}")
    return size


def _cells(facelets: list[str], face: str, row: int, size: int) -> list[str]:
    start = FACES_ORDER.index(face) * size * size
    return [facelets[start + row * size + col] for col in range(size)]


def _row(facelets: list[str], faces: str, row: int, size: int) -> str:
    return "  ".join(
        " ".join(_cells(facelets, face, row, size)) for face in faces
    )


def _indent(size: int) -> str:
    # Aligns the U/D block under the F column of the 4-wide middle row
    # (2n+2 spaces: n cells + separators for one face, plus one).
    return " " * (2 * size + 2)


def render_plain(facelets: list[str]) -> str:
    """Human/LLM-readable ASCII net using face letters."""
    size = cube_size(facelets)
    lines = []
    for row in range(size):
        lines.append(_indent(size) + _row(facelets, "U", row, size))
    for row in range(size):
        lines.append(_row(facelets, "LFRB", row, size))
    for row in range(size):
        lines.append(_indent(size) + _row(facelets, "D", row, size))
    return "\n".join(lines)


def render_colored(facelets: list[str]) -> Text:
    """Rich ``Text`` net with each sticker a colored block."""
    size = cube_size(facelets)
    text = Text()
    for row in range(size):
        text.append(_indent(size), style="default")
        text.append_text(_row_colored(facelets, "U", row, size))
        text.append("\n")
    for row in range(size):
        text.append_text(_row_colored(facelets, "LFRB", row, size))
        text.append("\n")
    for row in range(size):
        text.append(_indent(size), style="default")
        text.append_text(_row_colored(facelets, "D", row, size))
        text.append("\n")
    return text


def render_faces(facelets: list[str]) -> str:
    """A labeled grid for each face; easier for models to read than the compact net."""
    size = cube_size(facelets)
    lines = []
    for face in FACES_ORDER:
        start = FACES_ORDER.index(face) * size * size
        lines.append(f"{face} ({FACE_NAMES[face]}):")
        for row in range(size):
            lines.append(" ".join(facelets[start + row * size : start + row * size + size]))
    return "\n".join(lines)


def _row_colored(facelets: list[str], faces: str, row: int, size: int) -> Text:
    blocks: list[Text] = []
    for face in faces:
        group = Text()
        for col in range(size):
            start = FACES_ORDER.index(face) * size * size
            letter = facelets[start + row * size + col]
            bg = FACE_COLORS[letter]
            # Two spaces painted with the sticker color as the cell background.
            # This avoids dark foreground colors making the block look black.
            group.append("  ", style=f"on {bg}")
        blocks.append(group)
    sep = Text("  ")
    out = blocks[0].copy()
    for b in blocks[1:]:
        out.append_text(sep)
        out.append_text(b)
    return out
