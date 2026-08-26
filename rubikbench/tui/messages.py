"""Textual messages posted from the benchmark worker to the UI thread."""

from __future__ import annotations

from typing import Any

from textual.message import Message


class SolveStartedMsg(Message):
    def __init__(self, index: int, total: int, scramble: str) -> None:
        super().__init__()
        self.index = index
        self.total = total
        self.scramble = scramble


class StateMsg(Message):
    """Live cube state after any move batch."""

    def __init__(
        self,
        facelets: list[str],
        history: list[str],
        total_moves: int,
        turns: int,
        tool_calls: int,
        solved: bool,
    ) -> None:
        super().__init__()
        self.facelets = facelets
        self.history = history
        self.total_moves = total_moves
        self.turns = turns
        self.tool_calls = tool_calls
        self.solved = solved


class LogMsg(Message):
    def __init__(self, text: str, level: str = "info") -> None:
        super().__init__()
        self.text = text
        self.level = level


class TurnMsg(Message):
    def __init__(self, turn: int, content: str | None, tool_call_names: list[str], latency: float) -> None:
        super().__init__()
        self.turn = turn
        self.content = content
        self.tool_call_names = tool_call_names
        self.latency = latency


class StreamMsg(Message):
    """One streaming chunk of the model's reply (content / tool-call fragments)."""

    def __init__(
        self,
        turn: int,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None,
        usage: dict[str, int] | None,
        finish_reason: str | None,
        ttft: float | None,
    ) -> None:
        super().__init__()
        self.turn = turn
        self.content = content
        self.tool_calls = tool_calls
        self.usage = usage
        self.finish_reason = finish_reason
        self.ttft = ttft


class ToolCallMsg(Message):
    """A tool call was issued and is about to execute."""

    def __init__(self, turn: int, name: str, arguments: dict[str, Any], action: str) -> None:
        super().__init__()
        self.turn = turn
        self.name = name
        self.arguments = arguments
        self.action = action


class ToolResultMsg(Message):
    """A tool call finished and returned its text result."""

    def __init__(self, turn: int, name: str, content: str) -> None:
        super().__init__()
        self.turn = turn
        self.name = name
        self.content = content


class SolveDoneMsg(Message):
    def __init__(self, index: int, total: int, solve: Any) -> None:
        super().__init__()
        self.index = index
        self.total = total
        self.solve = solve


class BenchFinishedMsg(Message):
    def __init__(self, result: Any) -> None:
        super().__init__()
        self.result = result
