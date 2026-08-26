"""In-process OpenAI-compatible mock server used by tests.

Implements POST /v1/chat/completions (plain JSON and SSE streaming) with a
scripted agent: it reads the scramble from the first user message and solves the
cube by calling get_cube_state plus apply_moves with the inverse sequence,
finally replying with plain text. Lets us exercise the full benchmark loop
against a real HTTP endpoint without any external service.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from rubikbench.scramble import solution_for_scramble

SCRAMBLE_RE = re.compile(r"Scramble \(\d+ moves\): (.+)")
FACELET_RE = re.compile(r"Facelet string \([^)]*\):\s*([URFDLB]{54})")


def _solve_for(messages: list[dict[str, Any]]) -> list[str]:
    """Solve the state described in the first user message, from the facelet
    string when present (current prompt) or the scramble line (legacy prompts).
    """
    for m in messages:
        if m["role"] != "user":
            continue
        content = m.get("content") or ""
        facelet_match = FACELET_RE.search(content)
        if facelet_match:
            from rubikbench.solver_ref import solve_standard

            solution = solve_standard(facelet_match.group(1))
            if solution is not None:
                return solution
        scramble_match = SCRAMBLE_RE.search(content)
        if scramble_match:
            return solution_for_scramble(scramble_match.group(1).split())
    return []


def default_agent(body: dict[str, Any]) -> dict[str, Any]:
    """Return chat completion fields (content/tool_calls) for a request body."""
    messages = body.get("messages", [])
    solution = _solve_for(messages)
    has_tool_result = any(m.get("role") == "tool" for m in messages)
    if not has_tool_result and solution:
        tool_calls = [
            {
                "id": "call_observe",
                "type": "function",
                "function": {"name": "get_cube_state", "arguments": "{}"},
            },
            {
                "id": "call_apply",
                "type": "function",
                "function": {
                    "name": "apply_moves",
                    "arguments": json.dumps({"moves": " ".join(solution)}),
                },
            },
        ]
        return {"content": None, "tool_calls": tool_calls}
    if solution:
        return {"content": "The cube is solved.", "tool_calls": None}
    # Could not derive a solution (e.g. a direct client test with placeholder
    # content): just observe, so the loop stays non-degenerate.
    return {
        "content": None,
        "tool_calls": [{
            "id": "call_observe",
            "type": "function",
            "function": {"name": "get_cube_state", "arguments": "{}"},
        }],
    }


class MockApiError(Exception):
    """Raised by agents to force a specific HTTP status (for retry tests)."""

    def __init__(self, status: int = 500, detail: str = "mock failure") -> None:
        self.status = status
        self.detail = detail
        super().__init__(detail)


def build_response(reply: dict[str, Any], model: str = "mock") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    if reply.get("content") is not None:
        message["content"] = reply["content"]
    if reply.get("tool_calls"):
        message["tool_calls"] = reply["tool_calls"]
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": reply.get("finish_reason") or ("tool_calls" if reply.get("tool_calls") else "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }


def _sse_chunk(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/1.0"

    def log_message(self, *args: Any) -> None:  # silence request logging
        pass

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "bad json")
            return
        self.server.seen_bodies.append(body)  # type: ignore[attr-defined]
        agent = self.server.agent  # type: ignore[attr-defined]
        try:
            reply = agent(body)
        except MockApiError as exc:
            self.send_error(exc.status, exc.detail)
            return
        if body.get("stream"):
            self._send_sse(reply)
        else:
            data = json.dumps(build_response(reply, body.get("model", "mock"))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    # -- streaming ----------------------------------------------------------
    def _send_sse(self, reply: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()


        def emit(data: dict[str, Any]) -> None:
            self.wfile.write(_sse_chunk(data))
            self.wfile.flush()

        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        emit({"choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}]})
        content = reply.get("content")
        if content:
            emit({"choices": [{"delta": {"content": content}, "index": 0}]})
        for i, tc in enumerate(reply.get("tool_calls") or []):
            fn = tc["function"]
            args = fn["arguments"]
            half = max(1, len(args) // 2)
            first = {"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": fn["name"], "arguments": args[:half]},
            }]}, "index": 0}
            emit({"choices": [first]})
            rest = {"delta": {"tool_calls": [{
                "index": i, "function": {"arguments": args[half:]},
            }]}, "index": 0}
            emit({"choices": [rest]})
        finish = reply.get("finish_reason") or ("tool_calls" if reply.get("tool_calls") else "stop")
        emit({"choices": [{"delta": {}, "index": 0, "finish_reason": finish}], "usage": usage})
        self.wfile.write(b"data\n")
        self.wfile.flush()


def start_mock_server(
    agent: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    fail_first_n: int = 0,
    status: int = 500,
) -> tuple[ThreadingHTTPServer, str]:
    """Start the mock server; returns (server, base_url)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.seen_bodies: list[dict[str, Any]] = []  # type: ignore[attr-defined]
    effective = agent or default_agent
    failures = {"left": fail_first_n}

    def wrapped(body: dict[str, Any]) -> dict[str, Any]:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise MockApiError(status)
        return effective(body)

    server.agent = wrapped  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"
