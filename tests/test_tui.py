"""TUI smoke tests: drive the full app with Textual's Pilot against the mock server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mock_openai import start_mock_server
from textual.widgets import DataTable, Input

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
    server, url = mock
    cfg = BenchmarkConfig(base_url=url, api_key="k", model="mock", num_solves=2, max_turns=20, seed=3)
    config_file = tmp_path / "cfg.json"
    app = RubikBenchApp(config_path=config_file)
    app.config = cfg

    async with app.run_test(size=(140, 44)) as pilot:
        await wait_for(app, lambda a: a.screen.__class__.__name__ == "ConfigScreen")
        # exercise the token cap fields through the form
        app.screen.query_one("#max_output_tokens", Input).value = "512"
        app.screen.query_one("#max_input_tokens", Input).value = "4096"
        await pilot.click("#start_btn")

        saved = json.loads(config_file.read_text())
        assert saved["model"] == "mock"
        assert saved["max_output_tokens"] == 512
        assert saved["max_input_tokens"] == 4096
        assert server.seen_bodies[0]["max_tokens"] == 512

        # results table populated
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

        # back to config
        await pilot.click("#back-btn")
        await wait_for(app, lambda a: a.screen.__class__.__name__ == "ConfigScreen")


async def test_config_validation_shows_error(mock, tmp_path):
    app = RubikBenchApp(config_path=tmp_path / "none.json")
    app.config = BenchmarkConfig(base_url="", model="", num_solves=3)
    async with app.run_test(size=(140, 44)) as pilot:
        await wait_for(app, lambda a: a.screen.__class__.__name__ == "ConfigScreen")
        await pilot.click("#start_btn")
        await asyncio.sleep(0.3)
        assert isinstance(app.screen, ResultsScreen) is False
        status = app.screen.query_one("#status")
        assert "Configuration error" in str(status.render())


async def test_results_screen_for_empty_run(mock):
    """Aborted/empty runs render a sane results screen with no rows."""
    from rubikbench.benchmark import BenchmarkResult

    cfg = BenchmarkConfig(base_url="http://x/v1", model="mock", num_solves=3)
    empty = BenchmarkResult(config=cfg, solves=[], started_at_iso="now", duration=0.0)
    path = Path("tui_empty_test.json")
    try:
        app = RubikBenchApp(config_path=path)
        app.result = empty
        async with app.run_test(size=(120, 40)) as pilot:
            app.switch_screen(ResultsScreen(empty))
            await pilot.pause(0.2)
            table = app.screen.query_one("#table", DataTable)
            assert table.row_count == 1  # placeholder "—" row for empty runs
    finally:
        path.unlink(missing_ok=True)
