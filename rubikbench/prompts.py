"""Prompt templates and tool schemas driving the model."""

from __future__ import annotations

from .cube import Cube
from .rendering import render_faces, render_plain

SYSTEM_PROMPT = """You are solving a 3x3x3 Rubik's cube programmatically.

Singmaster notation: U, D, L, R, F, B are 90° clockwise quarter turns of the Up, Down, Left,
Right, Front, Back faces. An apostrophe means counter-clockwise (e.g. R'), and a 2 means a half
turn (e.g. U2). Move lists are space-separated, like "R U R' U'".

Goal: solve the cube. You are given the current cube state in every message; derive the solution
yourself. Do not ask for help. Do not explain your reasoning in the final response.

You have exactly one tool:
- apply_moves: apply one or more moves to the cube. You may call it with a single move or with many
  moves at once. This is the ONLY way to change the cube. Move text in your reply is not applied.

Scoring depends on total moves, turns, and tool calls used. Fewer is better."""


def initial_user_prompt(cube: Cube, presentation_mode: str = "stickers-v1") -> str:
    """The first user message: derived state only, never the scramble."""
    if presentation_mode == "cubie-v1":
        return (
            "Solve the cube from this legal structured cubie state. Position and piece names "
            "use the fixed U/R/F/D/L/B face orientation. Every corner gives the occupying "
            "piece and a numeric orientation (0, 1, or 2); every edge gives the occupying "
            "piece and orientation 0 or 1. The state supplies the precise convention.\n"
            f"{cube.cubie_state()}\n\n"
            "This is the current state. Every tool result supersedes it."
        )
    return (
        "Solve the cube from this state. Face orientation is fixed: U is up, D down, "
        "F faces you, B faces away, R is right, and L is left.\n"
        f"Facelet string (faces in order U R F D L B):\n{cube.facelet_string()}\n\n"
        f"Faces (each is a 3x3 grid, rows top-to-bottom):\n{render_faces(cube.facelets)}\n\n"
        f"Compact net:\n{render_plain(cube.facelets)}\n\n"
        "The state above is the current state. Every tool result will include the updated state."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "apply_moves",
            "description": (
                "Apply one or more Singmaster moves to the cube in a single call. Moves are "
                "space-separated, e.g. \"R U R' U'\". Valid faces: U D L R F B with optional ' or 2 "
                "suffix. Invalid tokens are ignored and reported. Returns the new cube state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {
                        "type": "string",
                        "description": "Space-separated Singmaster moves to apply, e.g. \"R U R' U'\".",
                    }
                },
                "required": ["moves"],
            },
        },
    },
]
