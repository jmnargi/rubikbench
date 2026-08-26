"""Shared TUI widgets."""

from __future__ import annotations

from textual.widgets import Static

from ..rendering import render_colored


class CubeNet(Static):
    """Renders the cube net as colored blocks; updates atomically."""

    def set_state(self, facelets: list[str]) -> None:
        self.update(render_colored(facelets))
