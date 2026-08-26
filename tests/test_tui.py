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


def _log_text(app) -> str:
    """Plain text of everything written to the run screen's history log."""
    from textual.widgets import RichLog

    log = app.screen.query_one("#log", RichLog)
    return "".join(seg.text for strip in log.lines for seg in strip)


async def test_run_screen_streams_live_output(mock, tmp_path):
    """Streaming chunks paint into the live panel, token counters, and history."""
    from textual.widgets import Label, Static

    from rubikbench.tui.messages import StreamMsg, ToolCallMsg, ToolResultMsg
    from rubikbench.tui.run_screen import RunScreen

    class IdleRunScreen(RunScreen):
        """Run screen without a real benchmark worker."""

        def _drive(self) -> None:
            pass

    cfg = BenchmarkConfig(base_url="http://x/v1", model="mock", num_solves=1)
    app = RubikBenchApp(config_path=tmp_path / "cfg.json", auto_run=False)
    app.config = cfg
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(IdleRunScreen(cfg))
        await pilot.pause()
        screen = app.screen
        assert screen.query_one("#live", Static) is not None
        assert screen.query_one("#log") is not None

        # reasoning content streams in and renders in the live panel
        screen.post_message(StreamMsg(1, None, None, None, None, None, "Let me think about the state..."))
        await pilot.pause()
        live = str(screen.query_one("#live", Static).render())
        assert "Let me think about the state..." in live

        # content streams in across chunks, with a first-token time
        screen.post_message(StreamMsg(1, "Let me ", None, None, None, 0.5))
        await pilot.pause()
        screen.post_message(
            StreamMsg(1, "observe the cube.", None, {"prompt": 10, "completion": 2, "cached": 1, "total": 13}, None, None)
        )
        await pilot.pause()
        live = str(screen.query_one("#live", Static).render())
        assert "Let me observe the cube." in live
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
        live = str(screen.query_one("#live", Static).render())
        assert "apply_moves" in live
        assert 'R U\'' in live

        # the executed tool call and its result land in the history log
        screen.post_message(ToolCallMsg(2, "apply_moves", {"moves": "R U'"}, "apply"))
        await pilot.pause()
        screen.post_message(ToolResultMsg(2, "apply_moves", "Applied 2 move(s). The cube is SOLVED."))
        await pilot.pause()
        text = _log_text(app)
        assert "apply_moves" in text
        assert "The cube is SOLVED" in text
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
