"""TUI smoke tests: drive the full app with Textual's Pilot against the mock server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mock_openai import start_mock_server
from textual.widgets import DataTable

from rubikbench.config import BenchmarkConfig
from rubikbench.tui.app import RubikBenchApp
from rubikbench.tui.results_screen import ResultsScreen


@pytest.fixture()
def mock():
    server, url = start_mock_server()
    yield server, url
    server.shutdown()


async def wait_for(app: RubikBenchApp, pred, timeout: float = 30.0) -> None:
    for _ in range(int(timeout * 10)):
        if pred(app):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"condition not met after {timeout}s (screen={app.screen.__class__.__name__})")


async def test_full_flow_through_tui(mock, tmp_path):
    """The view-only TUI starts the run immediately and lands on results."""
    server, url = mock
    cfg = BenchmarkConfig(
        base_url=url, api_key="k", model="mock", num_solves=2, max_turns=20, seed=3,
        max_output_tokens=512, max_input_tokens=4096, cache_retention=3600,
        scramble_preset="superflip",
    )
    app = RubikBenchApp(config_path=tmp_path / "cfg.json")
    app.config = cfg

    async with app.run_test(size=(140, 44)) as pilot:
        # the run starts automatically on mount - no clicks needed
        await wait_for(app, lambda a: a.result is not None)
        from rubikbench.scramble import PREMADE_SCRAMBLES

        assert app.result.solves[0].scramble == PREMADE_SCRAMBLES["superflip"][0].split()
        assert server.seen_bodies[0]["max_tokens"] == 512
        assert server.seen_bodies[0]["prompt_cache_retention"] == 3600

        # results table populated
        await wait_for(app, lambda a: a.screen.query_one("#table", DataTable).row_count == 2)
        results = app.screen
        table = results.query_one("#table", DataTable)
        assert table.row_count == 2
        agg = app.result.aggregates()
        assert agg["solves"] == 2
        assert agg["solve_rate"] == 1.0
        assert agg["avg_score"] > 0

        # detail pane populated for the first row
        transcript = results.query_one("#detail-transcript")
        assert transcript is not None
        # selecting a row refreshes the detail pane
        table.focus()
        await pilot.press("down")
        await pilot.press("enter")
        detail_summary = results.query_one("#detail-summary")
        assert len(detail_summary.lines) > 0

        # export
        await pilot.click("#export-btn")
        exports = list(Path("rubikbench_results").glob("bench_*.jsonl"))
        assert len(exports) >= 1
        lines = exports[-1].read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[1])["solved"] is True


def test_app_rejects_invalid_config(tmp_path):
    """A broken config file surfaces as a clear error before the UI starts."""
    from rubikbench.tui.app import RubikBenchApp

    bad = tmp_path / "cfg.json"
    bad.write_text('{"base_url": "http://x/v1", "model": "m", "num_solves": 0}')
    with pytest.raises(ValueError):
        RubikBenchApp(config_path=bad)


async def test_results_screen_for_empty_run(mock):
    """Aborted/empty runs render a sane results screen with no rows."""
    from rubikbench.benchmark import BenchmarkResult

    cfg = BenchmarkConfig(base_url="http://x/v1", model="mock", num_solves=3)
    empty = BenchmarkResult(config=cfg, solves=[], started_at_iso="now", duration=0.0)
    path = Path("tui_empty_test.json")
    try:
        app = RubikBenchApp(config_path=path, auto_run=False)
        app.result = empty
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ResultsScreen(empty))
            await pilot.pause(0.2)
            table = app.screen.query_one("#table", DataTable)
            assert table.row_count == 1  # placeholder "—" row for empty runs
    finally:
        path.unlink(missing_ok=True)


def _model_text(app) -> str:
    """Plain text of the run screen's right-hand model pane."""
    from textual.widgets import Static

    screen = app.screen
    history = str(screen.query_one("#history-text", Static).render())
    live = str(screen.query_one("#live-text", Static).render())
    return f"{history}\n{live}"


async def test_run_screen_streams_live_output(mock, tmp_path):
    """Streaming chunks paint into the model pane, token counters, and history."""
    from textual.widgets import Label, Static

    from rubikbench.cube import Cube
    from rubikbench.tui.messages import (
        SolveDoneMsg,
        StateMsg,
        StreamMsg,
        ToolCallMsg,
        ToolResultMsg,
        TurnMsg,
        TurnStartedMsg,
    )
    from rubikbench.tui.run_screen import RunScreen

    class IdleRunScreen(RunScreen):
        """Run screen without a real benchmark worker."""

        def _drive(self) -> None:
            pass

    cfg = BenchmarkConfig(base_url="http://x/v1", model="mock", num_solves=1, max_output_tokens=512)
    app = RubikBenchApp(config_path=tmp_path / "cfg.json", auto_run=False)
    app.config = cfg
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(IdleRunScreen(cfg))
        await pilot.pause()
        screen = app.screen
        assert screen.query_one("#history-text", Static) is not None
        assert screen.query_one("#live-text", Static) is not None
        assert screen.query_one("#model-scroll") is not None

        # config info lives in the left panel, not the model pane
        assert "vitruvix" in str(screen.query_one("#info-model", Label).render()) or "mock" in str(
            screen.query_one("#info-model", Label).render()
        )
        assert "512" in str(screen.query_one("#info-tokens", Label).render())

        # a sent request shows a live waiting indicator only at the bottom left
        screen.post_message(TurnStartedMsg(1))
        await pilot.pause()
        assert "waiting for model" in str(screen.query_one("#run-status", Static).render())
        assert "waiting for model" not in _model_text(app)

        # the cube pane renders colored stickers from a state event
        screen.post_message(StateMsg(list(Cube.solved().facelets), [], 0, 1, 1, False))
        await pilot.pause()
        cube_text = screen.query_one("#cube-net").render()
        styles = {str(span.style) for span in cube_text.spans}
        assert "on rgb(255,51,51)" in styles   # R face
        assert "on rgb(51,102,255)" in styles  # B face
        assert "on rgb(0,224,0)" in styles      # F face

        # reasoning content streams in, rendered grey, and Ctrl+T collapses it
        screen.post_message(StreamMsg(1, None, None, None, None, None, "Let me think about the state..."))
        await pilot.pause()
        assert "Let me think about the state..." in _model_text(app)
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert "Let me think about the state..." not in _model_text(app)
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert "Let me think about the state..." in _model_text(app)

        # Model output is literal text, even when it resembles Textual markup.
        markup_like = "B [ ^jl=+] R and move [R2]"
        screen.post_message(StreamMsg(1, None, None, None, None, None, markup_like))
        await pilot.pause()
        assert markup_like in _model_text(app)

        # Usage commonly arrives only in the final chunk; rate still updates
        # from the visible reasoning/content while the response is streaming.
        screen._live_stream_started -= 1.0
        screen._tick_stats()
        speed = str(screen.query_one("#st-speed", Label).render())
        assert speed.endswith(" tok/s")
        assert speed != "0 tok/s"

        # content streams in across chunks, with a first-token time
        screen.post_message(StreamMsg(1, "Let me ", None, None, None, 0.5))
        await pilot.pause()
        screen.post_message(
            StreamMsg(1, "observe the cube.", None, {"prompt": 10, "completion": 2, "cached": 1, "total": 13}, None, None)
        )
        await pilot.pause()
        assert "Let me observe the cube." in _model_text(app)
        assert str(screen.query_one("#st-tok-in", Label).render()) == "10"
        assert str(screen.query_one("#st-tok-out", Label).render()) == "2"
        assert str(screen.query_one("#st-tok-cached", Label).render()) == "1"

        # a tool call builds up from argument fragments
        screen.post_message(
            StreamMsg(2, None, [{"index": 0, "id": "c1", "name": "apply_moves", "arguments": '{"moves": "'}], None, None, None)
        )
        await pilot.pause()
        screen.post_message(StreamMsg(2, None, [{"index": 0, "id": None, "name": None, "arguments": "R U'}"}], None, None, None))
        await pilot.pause()
        assert "apply_moves" in _model_text(app)
        assert 'R U\'' in _model_text(app)

        # the executed tool call and its result land in the history
        screen.post_message(ToolCallMsg(2, "apply_moves", {"moves": "R U'"}, "apply"))
        await pilot.pause()
        screen.post_message(ToolResultMsg(2, "apply_moves", "Applied 2 move(s). The cube is SOLVED."))
        await pilot.pause()
        text = _model_text(app)
        assert "apply_moves" in text
        assert "The cube is SOLVED" in text

        # turn-level diagnostics: finish reason, output use versus output cap
        screen.post_message(TurnMsg(3, "wrapped up", [], 0.3, None, 10, 128, 5, 3, 146, "stop"))
        await pilot.pause()
        history_text = str(screen.query_one("#history-text", Static).render())
        assert "turn 3" in history_text
        assert "· stop" in history_text
        assert str(screen.query_one("#st-out-cap", Label).render()) == "128/512"

        # text actions are counted separately from tool calls
        screen.post_message(StateMsg(list(Cube.solved().facelets), [], 0, 3, 1, False, 2))
        await pilot.pause()
        assert str(screen.query_one("#st-text", Label).render()) == "2"

        # solve summary: finish-reason list and reported-vs-estimated token metrics
        from rubikbench.benchmark import SolveResult

        sr = SolveResult(
            index=0, scramble=["R", "U"], solved=True, turns=1, tool_calls=1, text_actions=0,
            total_moves=2, elapsed=0.5, prompt_tokens=10, completion_tokens=128,
            reasoning_tokens=5, estimated_reasoning_tokens=9, cached_tokens=3,
            estimated_cacheable_tokens=200, total_tokens=146, finish_reasons=["tool_calls", "stop"],
            truncated=True, timeline=[], par=4, score=900.0, breakdown={}, transcript=[],
        )
        screen.post_message(SolveDoneMsg(0, 1, sr))
        await pilot.pause()
        final_text = str(screen.query_one("#history-text", Static).render())
        assert "finish=tool_calls,stop" in final_text
        assert "(truncated)" in final_text
        assert "est: reasoning=9" in final_text and "cacheable=200" in final_text
async def test_replay_screen_steps_and_plays(mock, tmp_path):
    """Replay screen walks a real solve's timeline and toggles playback."""
    from textual.widgets import Button, OptionList, Select

    from rubikbench.benchmark import BenchmarkResult, run_solve
    from rubikbench.llm import OpenAICompatibleClient
    from rubikbench.scramble import PREMADE_SCRAMBLES
    from rubikbench.tui.replay_screen import ReplayScreen

    _, url = mock
    scramble = PREMADE_SCRAMBLES["catalog-10"][0].split()
    cfg = BenchmarkConfig(base_url=url, model="mock", num_solves=2, max_turns=15, seed=5)
    client = OpenAICompatibleClient(base_url=url, api_key="k", model="mock", max_retries=0)
    s0 = run_solve(0, scramble, cfg, client)
    s1 = run_solve(1, scramble, cfg, client)
    result = BenchmarkResult(config=cfg, solves=[s0, s1], started_at_iso="now", duration=1.0)

    app = RubikBenchApp(config_path=tmp_path / "none.json", auto_run=False)
    async with app.run_test(size=(110, 40)) as pilot:
        app.push_screen(ReplayScreen(result))
        await pilot.pause()
        assert isinstance(app.screen, ReplayScreen)

        opts = app.screen.query_one("#timeline", OptionList)
        assert opts.option_count == len(s0.timeline)
        info = app.screen.query_one("#step-info")
        assert "step 0/" in str(info.render())
        assert "start" in str(info.render())

        await pilot.press("right")
        assert "step 1/" in str(app.screen.query_one("#step-info").render())

        await pilot.press("space")
        assert app.screen.query_one("#btn-play", Button).label == "Pause"
        await pilot.press("space")
        assert app.screen.query_one("#btn-play", Button).label == "Play"

        solve_sel = app.screen.query_one("#solve-select", Select)
        solve_sel.value = s1.index
        await pilot.pause()
        assert "step 0/" in str(app.screen.query_one("#step-info").render())
        assert app.screen.query_one("#timeline", OptionList).option_count == len(s1.timeline)
