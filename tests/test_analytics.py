"""Analytics: cached tokens, finish_reason, truncation, retries, timeline."""

from __future__ import annotations

import json

import pytest
from mock_openai import default_agent, start_mock_server

from rubikbench.benchmark import BenchmarkRunner, export_jsonl, run_solve
from rubikbench.config import BenchmarkConfig
from rubikbench.cube import Cube
from rubikbench.llm import OpenAICompatibleClient
from rubikbench.scramble import PREMADE_SCRAMBLES

scramble = PREMADE_SCRAMBLES["catalog-16"][0].split()


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


# --------------------------------------------------------------------------- config

def test_cache_retention_merged_into_extra_body():
    cfg = base_cfg(cache_retention=3600, extra_body={"temperature": 0.2})
    body = cfg.effective_extra_body()
    assert body["prompt_cache_retention"] == 3600
    assert body["temperature"] == 0.2


def test_cache_retention_ignored_when_unset():
    assert "prompt_cache_retention" not in base_cfg().effective_extra_body()


def test_cache_retention_validation():
    with pytest.raises(ValueError):
        base_cfg(cache_retention=0).validate()
    with pytest.raises(ValueError):
        base_cfg(cache_retention=-5).validate()


# --------------------------------------------------------------------------- client

def test_client_captures_cached_tokens_and_finish_reason(mock):
    _, url = mock
    turn = make_client(url).complete([{"role": "user", "content": "go"}], [], {})
    assert turn.cached_tokens == 4
    assert turn.total_tokens == 19
    assert turn.prompt_tokens == 12
    assert turn.completion_tokens == 7
    assert turn.finish_reason in ("tool_calls", "stop")


def test_client_captures_finish_reason_from_sse(mock):
    _, url = mock
    turn = make_client(url, stream=True).complete([{"role": "user", "content": "go"}], [], {})
    assert turn.total_tokens == 19
    assert turn.cached_tokens == 4
    assert turn.finish_reason in ("tool_calls", "stop")


def test_client_records_ttft_on_stream(mock):
    _, url = mock
    turn = make_client(url, stream=True).complete([{"role": "user", "content": "hello"}], [], {})
    assert turn.ttft > 0  # first delta precedes the final chunk


# --------------------------------------------------------------------------- loop / runner

def test_run_solve_collects_analytics_and_timeline(mock):
    _, url = mock
    cfg = base_cfg(base_url=url, max_turns=30)
    result = run_solve(0, scramble, cfg, make_client(url))
    cube = Cube.solved()
    cube.apply(scramble)
    expected = cube.facelet_string()
    assert result.total_tokens > 0
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    assert result.finish_reasons and all(
        r in ("tool_calls", "stop") for r in result.finish_reasons
    )
    assert result.truncated is False
    assert result.retries == 0

    # timeline starts at the scrambled state and ends solved
    assert result.timeline[0]["facelets"] == expected
    assert result.timeline[0]["action"] == "start"
    assert result.timeline[0]["moves"] == []
    assert result.timeline[-1]["solved"] is True
    seen = {e["i"] for e in result.timeline}
    assert seen == set(range(len(result.timeline)))
    for entry in result.timeline:
        assert entry["t"] >= 0
        assert len(entry["facelets"]) == 54
        if entry["action"] == "apply":
            assert entry["moves"]

    # transcript entries are enriched with per-turn usage
    assistant = [e for e in result.transcript if e.get("role") == "assistant"]
    assert assistant
    assert all(e.get("prompt_tokens") and e.get("finish_reason") for e in assistant)
    assert all("cached_tokens" in e and "ttft" in e for e in assistant)
def test_aggregates_include_avg_time_and_cache_totals(mock):
    _, url = mock
    result = BenchmarkRunner(base_cfg(base_url=url, num_solves=2), make_client(url)).run()
    agg = result.aggregates()
    assert agg["avg_time"] > 0
    assert agg["total_cached_tokens"] > 0


def test_sdk_internal_retries_disabled():
    """The loop owns retries; the SDK must not multiply them."""
    import rubikbench.llm as llm_mod

    client = llm_mod.OpenAICompatibleClient(
        base_url="http://localhost:1/v1", api_key="k", model="m", max_retries=3
    )
    assert client._client.max_retries == 0  # SDK must not double-retry

def test_non_dict_tool_arguments_do_not_crash_run():
    """A model sending valid JSON with the wrong shape must not kill the run."""
    import json as _json
    import re as _re

    from rubikbench.solver_ref import solve_standard

    def weird_agent(body):
        messages = body.get("messages", [])
        if not any(m.get("role") == "tool" for m in messages):
            # First completion: apply_moves with LIST arguments (invalid shape).
            return {
                "content": None,
                "tool_calls": [{
                    "id": "call_weird",
                    "type": "function",
                    "function": {
                        "name": "apply_moves",
                        "arguments": _json.dumps(["R U R' U'"]),  # a list, not an object
                    },
                }],
            }
        # Recover: solve the (unchanged) initial state from the first user message.
        solution: list[str] = []
        for m in messages:
            if m.get("role") != "user":
                continue
            match = _re.search(r"Facelet string \([^)]*\):\s*([URFDLB]{54})", m.get("content") or "")
            if match:
                solution = solve_standard(match.group(1)) or []
                break
        return {
            "content": None,
            "tool_calls": [{
                "id": "call_recover",
                "type": "function",
                "function": {
                    "name": "apply_moves",
                    "arguments": _json.dumps({"moves": " ".join(solution)}),
                },
            }],
        }

    server, url = start_mock_server(agent=weird_agent)
    try:
        result = run_solve(0, scramble, base_cfg(base_url=url), make_client(url))
        assert result.solved
        assert any(
            e.get("role") == "tool" and "No moves provided" in e.get("content", "")
            for e in result.transcript
        ), "the model should have received a recoverable error message"
    finally:
        server.shutdown()


def test_retries_counted_in_solve():
    server, url = start_mock_server(fail_first_n=1)
    try:
        cfg = base_cfg(base_url=url, max_retries=3)
        result = run_solve(0, scramble, cfg, make_client(url))
        assert result.retries == 1
    finally:
        server.shutdown()


def test_truncation_detected_end_to_end():
    state = {"calls": 0}

    def length_agent(body):
        if state["calls"] == 0:
            state["calls"] += 1
            return {
                "content": "I need more room for reasoning...",
                "tool_calls": None,
                "finish_reason": "length",
            }
        return default_agent(body)

    server, url = start_mock_server(agent=length_agent)
    try:
        cfg = base_cfg(base_url=url, num_solves=1, max_turns=10)
        result = BenchmarkRunner(cfg, make_client(url)).run()
        solve = result.solves[0]
        assert "length" in solve.finish_reasons
        assert solve.truncated is True
        assert result.aggregates()["truncated_solves"] == 1
    finally:
        server.shutdown()


def test_export_includes_analytics(mock, tmp_path):
    _, url = mock
    result = BenchmarkRunner(base_cfg(base_url=url), make_client(url)).run()
    out = export_jsonl(result, tmp_path / "run.jsonl")
    lines = out.read_text().strip().splitlines()
    solve = json.loads(lines[1])
    assert solve["event"] == "solve"
    for key in ("cached_tokens", "total_tokens", "retries", "truncated",
                "finish_reasons", "timeline", "transcript"):
        assert key in solve, key
    assert solve["truncated"] is False
    assert solve["timeline"][0]["action"] == "start"
    assert solve["timeline"][0]["moves"] == []
    assert solve["timeline"][0]["facelets"]


def test_timeline_states_match_transcript_actions(mock):
    """Every apply/reset timeline entry matches a tool transcript entry."""
    _, url = mock
    result = run_solve(0, scramble, base_cfg(base_url=url), make_client(url))
    tool_entries = [e for e in result.transcript if e.get("role") == "tool"]
    timeline_tool = [e for e in result.timeline if e["action"] in ("apply", "reset")]
    assert len(timeline_tool) == len(tool_entries)
    for entry, tool in zip(timeline_tool, tool_entries):
        if entry["action"] == "apply":
            assert tool["name"] == "apply_moves"
            assert "moves" in tool["arguments"]
        elif entry["action"] == "reset":
            assert tool["name"] == "reset_cube"
