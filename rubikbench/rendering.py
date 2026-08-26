"""ASCII / colored rendering of the cube net."""

from __future__ import annotations

from rich.text import Text

# Face letter -> (rich color for the sticker background, dark text color)
FACE_COLORS = {
    "U": ("white", "black"),
    "R": ("red", "white"),
    "F": ("green", "black"),
    "D": ("yellow", "black"),
    "L": ("orange1", "black"),
    "B": ("blue", "white"),
}

_INDENT = "        "  # 8 spaces: aligns U/D under the F column of the 4-wide row

def _row(facelets: list[str], faces: str, row: int) -> str:
    cells = []
    for face in faces:
        for col in range(3):
            cells.append(facelets[FACES_ORDER.index(face) * 9 + row * 3 + col])
    return "  ".join(
        " ".join(chunk) for chunk in (cells[0:3], cells[3:6], cells[6:9], cells[9:12])
    )


FACES_ORDER = "URFDLB"


def render_plain(facelets: list[str]) -> str:
    """Human/LLM-readable ASCII net using face letters."""
    lines = []
    for row in range(3):
        lines.append(_INDENT + _row(facelets, "U", row))
    for row in range(3):
        lines.append(_row(facelets, "LFRB", row))
    for row in range(3):
        lines.append(_INDENT + _row(facelets, "D", row))
    return "\n".join(lines)


def render_colored(facelets: list[str]) -> Text:
    """Rich ``Text`` net with each sticker a colored block."""
    text = Text()
    for row in range(3):
        text.append("        ", style="default")
        text.append_text(_row_colored(facelets, "U", row))
        text.append("\n")
    for row in range(3):
        text.append_text(_row_colored(facelets, "LFRB", row))
        text.append("\n")
    for row in range(3):
        text.append("        ", style="default")
        text.append_text(_row_colored(facelets, "D", row))
        text.append("\n")
    return text


def _row_colored(facelets: list[str], faces: str, row: int) -> Text:
    blocks: list[Text] = []
    for face in faces:
        group = Text()
        for col in range(3):
            letter = facelets[FACES_ORDER.index(face) * 9 + row * 3 + col]
            bg, fg = FACE_COLORS[letter]
            group.append("██", style=f"bold {fg} on {bg}")
        blocks.append(group)
    sep = Text("  ")
    out = blocks[0].copy()
    for b in blocks[1:]:
        out.append_text(sep)
        out.append_text(b)
    return out
