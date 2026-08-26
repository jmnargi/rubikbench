"""Prompt templates and tool schemas driving the model."""

from __future__ import annotations

from .cube import Cube
from .rendering import render_plain
from .scramble import scramble_to_string

SYSTEM_PROMPT = """You are solving a 3x3x3 Rubik's cube programmatically and will be benchmarked on it.

The cube uses standard Singmaster notation: U, D, L, R, F, B are quarter turns of the Up, Down,
Left, Right, Front, Back faces; an apostrophe means counter-clockwise (e.g. R'), and a 2 means a
half turn (e.g. U2). Move lists are space-separated, like "R U R' U'".

You have these tools:
- get_cube_state: return the exact current state of the cube (facelet string, colored net, move
  history and counts). Call it whenever you need to observe or verify the cube.
- apply_moves: apply one or more moves in a single call. This is the ONLY way to change the cube.
  Move text you write in your reply is not applied.
- reset_cube: restore the original scramble (discards all moves so far; costs you turns).

Scoring (0-1000, solved runs only):
- 50% efficiency = par_moves / total_moves, where par_moves is a near-optimal solution length
  (~20 moves; God's number). Efficient human methods take 50-70 moves; beginner methods 120-200.
- 30% turn economy = how few of your replies (turns) you used.
- 20% tool economy = how few tool calls you made (ideally one batch per turn).
You want the cube SOLVED with as few total moves, as few turns, and as few tool calls as possible.

Strategy guidance:
- Think carefully before calling apply_moves: plan a sequence of several moves that advances your
  strategy, and verify with get_cube_state when in doubt.
- Batch related moves into a single apply_moves call rather than one move per call.
- If you mess up, reset_cube is cheaper than wandering (it costs turns but no moves).
- Verify the cube is solved with get_cube_state before declaring success.
- When you are confident the cube is solved, reply briefly, e.g. "The cube is solved."."""


def initial_user_prompt(scramble: list[str], cube: Cube) -> str:
    state = cube.facelet_string()
    return (
        f"Scramble ({len(scramble)} moves): {scramble_to_string(scramble)}\n"
        f"Facelet string (order U R F D L B):\n{state}\n"
        f"Cube net:\n{render_plain(cube.facelets)}\n\n"
        "Solve the cube using the tools. You may call get_cube_state at any time to verify."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cube_state",
            "description": (
                "Return the exact current state of the Rubik's cube: all 54 facelets, the colored "
                "net, the move history, and current counts. Use this to observe or verify the cube."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
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
    {
        "type": "function",
        "function": {
            "name": "reset_cube",
            "description": (
                "Restore the original scramble, discarding all moves applied so far. Useful when "
                "the solve has gone off track. Costs extra turns (and thus score), but no moves."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
