"""Shared TUI widgets."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from ..rendering import render_colored


class CubeNet(Static):
    """Renders the cube net as colored blocks; updates atomically."""

    def set_state(self, facelets: list[str] | str) -> None:
        net = render_colored(facelets)
        legend = Text(
            "\nU up · D down · F front · B back · R right · L left",
            style="dim",
        )
        self.update(net + legend)
