"""End-to-end tests of the client, tool loop, and runner against the mock server,
plus fake-client tests for the text-moves fallback, turn budget, and abort paths."""

from __future__ import annotations

import json
import re
import threading
import time

import pytest
from mock_openai import MockApiError, start_mock_server

from rubikbench.benchmark import (
    BenchmarkRunner,
    SolveContext,
    execute_tool,
    export_jsonl,
    run_solve,
)
from rubikbench.config import BenchmarkConfig
from rubikbench.llm import AssistantTurn, LLMError, OpenAICompatibleClient
from rubikbench.scramble import scramble_to_string


def make_client(url: str, **kw) -> OpenAICompatibleClient:
    params = {"base_url": url, "api_key": "test-key", "model": "mock", "max_retries": 0}
    params.update(kw)
    return OpenAICompatibleClient(**params)


def base_cfg(**kw) -> BenchmarkConfig:
    params = {"model": "mock", "num_solves": 1, "max_turns": 30, "seed": 1}
    params.update(kw)
    return BenchmarkConfig(**params)


@pytest.fixture()
def mock():
    server, url = start_mock_server()
    yield server, url
    server.shutdown()


def fail_first(n: int):
    from mock_openai import default_agent

    state = {"left": n}

    def agent(body):
        if state["left"] > 0:
            state["left"] -= 1
            raise MockApiError(500, "boom")
        return default_agent(body)

    return agent


# --------------------------------------------------------------------------- client

def test_client_parses_tool_calls(mock):
    server, url = mock
    client = make_client(url, extra_body={"reasoning_effort": "high"})
    turn = client.complete(
        [{"role": "user", "content": "Scramble (2 moves): R U"}],
        [{"type": "function", "function": {"name": "apply_moves", "parameters": {"type": "object", "properties": {}}}}],
    )
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "apply_moves"
    assert turn.tool_calls[0].arguments["moves"] == "U' R'"
    assert turn.latency >= 0
    # extra_body is flattened into the request body by the SDK
    assert server.seen_bodies[0]["reasoning_effort"] == "high"


def test_client_streaming_assembles_tool_args(mock):
    _, url = mock
    client = make_client(url, stream=True)
    turn = client.complete(
        [{"role": "user", "content": "Scramble (2 moves): R U"}],
        [{"type": "function", "function": {"name": "apply_moves", "parameters": {"type": "object", "properties": {}}}}],
    )
    assert len(turn.tool_calls) == 1
    applied = turn.tool_calls[0]
    assert applied.name == "apply_moves"
    assert applied.arguments == {"moves": "U' R'"}
    assert applied.raw == json.dumps({"moves": "U' R'"})
    assert turn.prompt_tokens == 12


def test_client_streaming_on_chunk_reports_deltas(mock):
    """on_chunk fires per SSE chunk with content, tool-call, and usage deltas."""
    _, url = mock
    client = make_client(url, stream=True)

    content_chunks: list[dict] = []
    turn = client.complete(
        [
            {"role": "user", "content": "Scramble (2 moves): R U"},
            {"role": "tool", "tool_call_id": "call_x", "content": "ok"},
        ],
        [],
        on_chunk=content_chunks.append,
    )
    assert turn.content == "The cube is solved."
    # reasoning content streams through and lands on the turn
    assert turn.reasoning == "I inverted the scramble to solve it."
    joined = "".join(c.get("content") or "" for c in content_chunks)
    assert joined == "The cube is solved."
    reasoning_joined = "".join(c.get("reasoning") or "" for c in content_chunks)
    assert reasoning_joined == "I inverted the scramble to solve it."
    assert any(c.get("usage") and c["usage"]["total"] == 19 for c in content_chunks)
    assert content_chunks[-1]["finish_reason"] == "stop"

    tool_chunks: list[dict] = []
    client.complete(
        [{"role": "user", "content": "Scramble (2 moves): R U"}],
        [{"type": "function", "function": {"name": "apply_moves", "parameters": {"type": "object", "properties": {}}}}],
        on_chunk=tool_chunks.append,
    )
    deltas = [tc for c in tool_chunks for tc in (c.get("tool_calls") or [])]
    assert deltas
    assert any(d.get("name") for d in deltas)
    assert any(d.get("arguments") for d in deltas)


def test_client_always_sends_stream_options(mock):
    """Streaming requests always request per-stream usage via stream_options."""
    server, url = mock
    client = make_client(url, stream=True)
    client.complete(
        [{"role": "user", "content": "Scramble (2 moves): R U"}],
        [{"type": "function", "function": {"name": "apply_moves", "parameters": {"type": "object", "properties": {}}}}],
    )
    assert server.seen_bodies[0]["stream_options"] == {"include_usage": True}

def test_client_sends_top_p_top_level(mock):
    """top_p is a standard OpenAI parameter and goes top-level, not extra_body."""
    server, url = mock
    client = make_client(url, stream=True, top_p=0.9)
    client.complete(
        [{"role": "user", "content": "Scramble (2 moves): R U"}],
        [{"type": "function", "function": {"name": "apply_moves", "parameters": {"type": "object", "properties": {}}}}],
    )
    body = server.seen_bodies[0]
    assert body["top_p"] == 0.9
    assert "top_p" not in (body.get("extra_body") or {})


def test_client_sends_sampling_params_via_extra_body(mock):
    """repetition_penalty and top_k are vLLM knobs; they ride in extra_body."""
    server, url = mock
    client = make_client(
        url,
        stream=True,
        extra_body={"repetition_penalty": 1.15, "top_k": 40, "reasoning_effort": "high"},
    )
    client.complete(
        [{"role": "user", "content": "Scramble (2 moves): R U"}],
        [],
    )
    body = server.seen_bodies[0]
    assert body["repetition_penalty"] == 1.15
    assert body["top_k"] == 40
    assert body["reasoning_effort"] == "high"


def test_loop_detection_aborts_looping_stream(mock):
    """A stream that repeats a short pattern is killed and retried."""
    server, url = mock
    server.agent = lambda body: {"content": "R U R U R U R U R U R U", "finish_reason": "stop"}
    client = make_client(url, stream=True)
    with pytest.raises(LLMError, match="looping"):
        client.complete([{"role": "user", "content": "Scramble (2 moves): R U"}], [])


def test_loop_detection_can_be_disabled(mock):
    """With loop_detection off, a repeating stream completes normally."""
    server, url = mock
    server.agent = lambda body: {"content": "R U R U R U R U R U R U", "finish_reason": "stop"}
    client = make_client(url, stream=True, loop_detection=False)
    turn = client.complete([{"role": "user", "content": "Scramble (2 moves): R U"}], [])
    assert turn.content == "R U R U R U R U R U R U"

def test_idle_timeout_aborts_stalled_stream():
    """A stream that stops producing chunks after the header is killed and retried."""
    import http.server

    class StallingHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0}]}\n\n'
            )
            self.wfile.flush()
            time.sleep(2)  # stall: client must abort before this returns

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StallingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    client = make_client(f"http://{host}:{port}/v1", stream=True, stream_idle_timeout=0.2)
    try:
        with pytest.raises(LLMError, match="watchdog"):
            client.complete([{"role": "user", "content": "hi"}], [])
    finally:
        server.shutdown()


def test_client_surfaces_http_errors():
    server, url = start_mock_server(fail_first_n=1, status=429)
    try:
        client = make_client(url)
        with pytest.raises(LLMError, match="429"):
            client.complete([{"role": "user", "content": "hi"}], [])
    finally:
        server.shutdown()


def test_client_surfaces_connection_errors():
    client = make_client("http://127.0.0.1:1/v1", timeout=2)
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "hi"}], [])


# --------------------------------------------------------------------------- tool execution

def test_execute_tool_apply_moves_invalid_tokens():
    ctx = SolveContext(["R", "U"])
    result = execute_tool("apply_moves", {"moves": "R X junk2"}, ctx)
    assert "1 move(s)" in result
    assert "invalid" in result
    assert ctx.total_moves == 1
    assert ctx.cube.history == ["R"]


def test_execute_tool_reset_is_rejected():
    # reset_cube was removed; it should now be treated as an unknown tool.
    ctx = SolveContext(["R", "U", "F"])
    ctx.cube.apply(["R", "U", "F"])
    out = execute_tool("reset_cube", {}, ctx)
    assert "Unknown tool" in out
    assert ctx.cube.history == ["R", "U", "F"]


def test_execute_tool_unknown_tool():
    ctx = SolveContext([])
    assert "Unknown tool" in execute_tool("frobnicate", {}, ctx)


# --------------------------------------------------------------------------- solve loop (real HTTP)

def test_solve_loop_e2e_solves_via_tools(mock):
    _, url = mock
    client = make_client(url)
    scramble = ["R", "U", "F'", "D2"]
    result = run_solve(0, scramble, base_cfg(reasoning_effort="high"), client)
    assert result.solved
    assert result.error is None
    assert result.turns == 1
    assert result.tool_calls == 1
    assert result.total_moves == len(scramble)
    assert result.par >= 1
    assert result.score > 0
    assert result.breakdown["solved"] is True
    assert len(result.transcript) >= 2  # assistant + 1 tool result
    assert result.scramble == scramble


def test_solve_loop_streaming_e2e(mock):
    _, url = mock
    result = run_solve(0, ["R", "U"], base_cfg(stream=True), make_client(url, stream=True))
    assert result.solved
    assert result.turns == 1


def test_solve_loop_streams_chunks_and_tool_events(mock):
    """The emitter receives per-chunk stream events plus tool call/result events."""
    _, url = mock
    events: list[tuple[str, dict]] = []

    def emitter(kind: str, payload: dict) -> None:
        events.append((kind, payload))

    result = run_solve(0, ["R", "U"], base_cfg(stream=True), make_client(url, stream=True), emitter=emitter)
    assert result.solved
    kinds = [k for k, _ in events]
    assert "stream" in kinds
    # tool-call deltas flowed through as chunks (the mock solves via tools)
    chunks = [p for k, p in events if k == "stream"]
    assert any(c.get("tool_calls") for c in chunks)
    # the final chunk carries usage for live token counters
    assert any(c.get("usage") and c["usage"]["total"] > 0 for c in chunks)
    assert "tool_call" in kinds and "tool_result" in kinds
    tool_events = [p for k, p in events if k == "tool_call"]
    assert {t["name"] for t in tool_events} == {"apply_moves"}


def test_retry_then_success(mock):
    server, url = mock
    server.agent = fail_first(3)
    cfg = base_cfg(max_retries=5)
    result = run_solve(0, ["R", "U"], cfg, make_client(url))
    assert result.solved
    assert result.error is None


def test_retry_exhausted(mock):
    server, url = mock
    server.agent = fail_first(5)
    result = run_solve(0, ["R", "U"], base_cfg(max_retries=1), make_client(url))
    assert not result.solved
    assert result.error is not None
    assert "API error" in result.error


def test_abort_before_first_turn(mock):
    _, url = mock
    cancel = threading.Event()
    cancel.set()
    result = run_solve(0, ["R", "U"], base_cfg(), make_client(url), cancel_event=cancel)
    assert result.error == "aborted by user"
    assert not result.solved


# --------------------------------------------------------------------------- runner

def test_runner_solves_three_cubes(mock):
    _, url = mock
    cfg = base_cfg(num_solves=3, seed=7)
    runner = BenchmarkRunner(cfg, make_client(url))
    result = runner.run()
    assert len(result.solves) == 3
    assert all(s.solved for s in result.solves)
    agg = result.aggregates()
    assert agg["solve_rate"] == 1.0
    assert agg["solves"] == 3
    assert agg["avg_moves"] > 0
    assert result.duration >= 0


def test_runner_cancel_stops_early(mock):
    _, url = mock
    cancel = threading.Event()
    cancel.set()
    cfg = base_cfg(num_solves=5)
    runner = BenchmarkRunner(cfg, make_client(url))
    result = runner.run(cancel_event=cancel)
    assert len(result.solves) == 0


def test_runner_custom_scrambles(mock):
    _, url = mock
    cfg = base_cfg(num_solves=2, scrambles=["R U", "F' B2 R"])
    runner = BenchmarkRunner(cfg, make_client(url))
    result = runner.run()
    assert [s.scramble for s in result.solves] == [["R", "U"], ["F'", "B2", "R"]]


def test_export_jsonl_roundtrip(mock, tmp_path):
    _, url = mock
    cfg = base_cfg(num_solves=2)
    result = BenchmarkRunner(cfg, make_client(url)).run()
    out = tmp_path / "results.jsonl"
    export_jsonl(result, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    header = json.loads(lines[0])
    assert header["event"] == "benchmark"
    assert header["aggregates"]["solves"] == 2
    for line in lines[1:]:
        solve = json.loads(line)
        assert solve["event"] == "solve"
        assert solve["solved"] is True
        assert "transcript" in solve


# --------------------------------------------------------------------------- fake clients

class FixedContentClient:
    def __init__(self, content: str):
        self.content = content

    def complete(self, messages, tools, extra_body, on_chunk=None):
        return AssistantTurn(content=self.content)


class TextSolverClient:
    """Solves by writing the moves as plain text (no tool calls)."""

    def complete(self, messages, tools, extra_body, on_chunk=None):
        solution = self._solution(messages)
        return AssistantTurn(content=scramble_to_string(solution))

    @staticmethod
    def _solution(messages):
        from rubikbench.solver_ref import solve_standard

        for m in messages:
            if m.get("role") == "user":
                match = re.search(r"Facelet string \([^)]*\):\s*([URFDLB]{54})", m.get("content") or "")
                if match:
                    standard = solve_standard(match.group(1))
                    if standard is not None:
                        return standard
        raise AssertionError("no cube state found in messages")


def test_text_moves_fallback_solves(mock):
    result = run_solve(0, ["R", "U", "F'"], base_cfg(protocol_mode="text_compat"), TextSolverClient())
    assert result.solved
    assert result.tool_calls == 0
    assert result.total_moves == 3
    assert result.turns == 1
    assert result.score > 0


def test_text_moves_disabled_does_not_apply(mock):
    result = run_solve(0, ["R", "U"], base_cfg(allow_text_moves=False), TextSolverClient())
    assert not result.solved
    assert result.total_moves == 0
    assert result.turns >= 1


def test_junk_turns_hit_budget():
    cfg = base_cfg(max_turns=5)
    result = run_solve(0, ["R", "U"], cfg, FixedContentClient("Hmm, let me think about this..."))
    assert not result.solved
    assert result.turns == 5
    assert result.score == 0.0
    assert result.error is None


def test_raw_tool_call_markup_is_rejected_and_nudged():
    cfg = base_cfg(max_turns=3, allow_text_moves=False)
    result = run_solve(0, ["R", "U"], cfg, FixedContentClient("<apply_moves>moves: R U R'</apply_moves>"))
    assert not result.solved
    assert result.total_moves == 0
    assert result.tool_calls == 0
    assert any(
        m["role"] == "user" and "apply_moves tool" in m.get("content", "")
        for m in result.transcript
    )


def test_empty_turn_is_rejected_and_nudged():
    cfg = base_cfg(max_turns=3, allow_text_moves=False)
    result = run_solve(0, ["R", "U"], cfg, FixedContentClient(""))
    assert not result.solved
    assert result.total_moves == 0
    assert result.tool_calls == 0
    assert any(
        m["role"] == "user" and "empty" in m.get("content", "")
        for m in result.transcript
    )


# --------------------------------------------------------------------------- token caps

from rubikbench.benchmark import estimate_message_tokens, trim_messages


def _msg(role: str, content: str = "", tool_calls=None) -> dict:
    m = {"role": role}
    if content:
        m["content"] = content
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


def _tool_calls(arguments: str) -> list:
    return [{"id": "c1", "type": "function", "function": {"name": "apply_moves", "arguments": arguments}}]


def test_trim_disabled_without_cap():
    messages = [_msg("system", "s"), _msg("user", "u"), _msg("assistant", "a")]
    out, trimmed = trim_messages(messages, None)
    assert out == messages
    assert trimmed is False


def test_trim_noop_under_cap():
    messages = [_msg("system", "s" * 100), _msg("user", "u" * 100)]
    out, trimmed = trim_messages(messages, 1000)
    assert out == messages
    assert trimmed is False


def test_trim_keeps_system_and_initial_user():
    messages = [
        _msg("system", "s" * 100),
        _msg("user", "u" * 100),
        _msg("assistant", "a" * 1000),
        _msg("tool", "t" * 1000),
        _msg("assistant", "b" * 1000),
        _msg("tool", "t" * 1000),
        _msg("assistant", "c" * 1000),
    ]
    out, trimmed = trim_messages(messages, 200)
    assert trimmed is True
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert estimate_message_tokens(out[0]) + estimate_message_tokens(out[1]) <= 200
    # every remaining assistant message has its own tool results, no orphans
    for i, m in enumerate(out):
        if m["role"] == "tool":
            assert out[i - 1]["role"] == "assistant"


def test_trim_removes_complete_turn_units():
    messages = [
        _msg("system", "s"),
        _msg("user", "u"),
        _msg("assistant", "a" * 500),
        _msg("tool", "t" * 500),
        _msg("assistant", "b" * 500),
        _msg("tool", "t" * 500),
    ]
    out, trimmed = trim_messages(messages, 400)
    assert trimmed is True
    # only the first (oldest) turn unit is removed; the second remains
    assert all(m["role"] not in ("assistant", "tool") or m["content"] != "a" * 500 for m in out)
    assert any(m.get("content") == "b" * 500 for m in out)


def test_trim_small_cap_raises_when_prefix_cannot_fit():
    """A budget too small for the immutable system/initial prefix raises clearly."""
    messages = [
        _msg("system", "s" * 100),
        _msg("user", "u" * 100),
        _msg("assistant", "a" * 500),
    ]
    with pytest.raises(ValueError, match="cannot fit"):
        trim_messages(messages, 60)

class RecordingClient:
    """Wraps another client and records every messages list it receives."""

    def __init__(self, inner):
        self.inner = inner
        self.seen: list[list[dict]] = []

    def complete(self, messages, tools, extra_body, on_chunk=None):
        self.seen.append(list(messages))
        return self.inner.complete(messages, tools, extra_body, on_chunk=on_chunk)


def test_max_output_tokens_sent_in_body(mock):
    server, url = mock
    cfg = base_cfg(max_output_tokens=512, max_input_tokens=None)
    client = make_client(url, max_output_tokens=cfg.max_output_tokens)
    run_solve(0, ["R", "U"], cfg, client)
    assert server.seen_bodies[0]["max_tokens"] == 512
    assert "max_tokens" not in (client._extra_body or {})  # passed top-level, not in extra_body


def test_max_input_tokens_trims_history(mock):
    """Trimming preserves the immutable prefix plus the latest-state checkpoint."""
    _mock = mock
    # Small but sufficient: fits system + initial user + current-state
    # checkpoint, but never the growing transcript, so every request once the
    # history exceeds the cap collapses to that checkpoint prefix.
    cfg = base_cfg(max_turns=10, max_input_tokens=700, allow_text_moves=False)
    recording = RecordingClient(FixedContentClient("x" * 2000))
    run_solve(0, ["R", "U"], cfg, recording)
    from rubikbench.cube import Cube

    expected = Cube.solved()
    expected.apply(["R", "U"])
    assert len(recording.seen) == 10
    for messages in recording.seen[1:]:  # first request has no history to trim
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["content"].startswith("[CURRENT STATE CHECKPOINT]")
        assert expected.facelet_string() in messages[2]["content"]


def test_runner_premade_catalog_cycles(mock):
    from rubikbench.scramble import PREMADE_SCRAMBLES, scramble_from_string

    _, url = mock
    cfg = base_cfg(num_solves=6, scramble_preset="catalog-10")
    result = BenchmarkRunner(cfg, make_client(url)).run()
    expected = [scramble_from_string(s) for s in PREMADE_SCRAMBLES["catalog-10"]]
    assert [s.scramble for s in result.solves] == expected + expected[: 6 - 4]


def test_runner_premade_superflip(mock):
    from rubikbench.scramble import PREMADE_SCRAMBLES, scramble_from_string

    _, url = mock
    cfg = base_cfg(num_solves=1, scramble_preset="superflip")
    result = BenchmarkRunner(cfg, make_client(url)).run()
    assert result.solves[0].scramble == scramble_from_string(PREMADE_SCRAMBLES["superflip"][0])
    assert result.solves[0].solved


def test_initial_prompt_does_not_leak_scramble():
    from rubikbench.cube import Cube
    from rubikbench.prompts import SYSTEM_PROMPT, initial_user_prompt

    scramble = ["R", "U", "F'", "D2"]
    cube = Cube.solved()
    cube.apply(scramble)
    prompt = initial_user_prompt(cube)
    assert "Scramble" not in prompt
    assert " ".join(scramble) not in prompt
    assert cube.facelet_string() in prompt  # the state IS given
    assert "Goal: solve the cube" in SYSTEM_PROMPT
    assert "apply_moves" in SYSTEM_PROMPT
    assert "reset_cube" not in SYSTEM_PROMPT.lower()


def test_loop_first_message_has_no_scramble(mock):
    server, url = mock
    cfg = base_cfg(max_turns=3)
    run_solve(0, ["R", "U", "F"], cfg, make_client(url))
    first_user = next(m for m in server.seen_bodies[0]["messages"] if m["role"] == "user")
    assert "Scramble" not in first_user["content"]
    assert "R U F" not in first_user["content"]
