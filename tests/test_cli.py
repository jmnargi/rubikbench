"""Tests for the headless CLI (``rubikbench run``) against the mock server."""

from __future__ import annotations

import json

import pytest
from mock_openai import start_mock_server

from rubikbench.cli import main
from rubikbench.config import BenchmarkConfig, save_config


@pytest.fixture()
def mock():
    server, url = start_mock_server()
    yield server, url
    server.shutdown()


def _write_config(tmp_path, url: str) -> str:
    cfg = BenchmarkConfig(base_url=url, api_key="k", model="mock", num_solves=2, max_turns=15, seed=9)
    path = tmp_path / "cfg.json"
    save_config(cfg, path)
    return str(path)


def test_headless_run(mock, tmp_path, capsys):
    _, url = mock
    cfg_path = _write_config(tmp_path, url)
    out = tmp_path / "out.jsonl"
    code = main(["run", "--config", cfg_path, "-o", str(out), "--no-color"])
    assert code == 0
    captured = capsys.readouterr()
    stdout = captured.out[: captured.out.index("results written")]
    agg = json.loads(stdout)
    assert agg["solves"] == 2
    assert agg["solve_rate"] == 1.0
    assert agg["avg_score"] > 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    solve = json.loads(lines[1])
    assert solve["solved"] is True
    assert "transcript" in solve
    assert "SOLVED" in captured.err


def test_headless_run_missing_config(tmp_path, capsys):
    code = main(["run", "--config", str(tmp_path / "nope.json")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_headless_validate(mock, tmp_path, capsys):
    _, url = mock
    cfg_path = _write_config(tmp_path, url)
    code = main(["validate", "--config", cfg_path])
    assert code == 0
    assert "config OK" in capsys.readouterr().out


def test_presets_list(capsys):
    code = main(["presets"])
    assert code == 0
    out = capsys.readouterr().out
    assert "OpenAI" in out
    assert "vLLM (local)" in out


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "rubikbench" in capsys.readouterr().out
