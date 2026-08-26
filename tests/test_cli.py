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


@pytest.fixture(autouse=True)
def _clean_rubikbench_env(monkeypatch):
    """Isolate tests from any real .env the developer may have."""
    for name in (
        "RUBIKBENCH_API_KEY",
        "RUBIKBENCH_BASE_URL",
        "RUBIKBENCH_MODEL",
        "RUBIKBENCH_SOLVES",
        "RUBIKBENCH_MAX_TURNS",
        "RUBIKBENCH_SCRAMBLE_LEN",
        "RUBIKBENCH_SEED",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_headless_run_from_env_no_config(mock, tmp_path, capsys, monkeypatch):
    """``rubikbench run`` with no config file works from .env-style variables."""
    _, url = mock
    monkeypatch.setenv("RUBIKBENCH_BASE_URL", url)
    monkeypatch.setenv("RUBIKBENCH_API_KEY", "env-key")
    monkeypatch.setenv("RUBIKBENCH_MODEL", "mock")
    monkeypatch.setenv("RUBIKBENCH_SOLVES", "2")
    out = tmp_path / "out.jsonl"
    code = main(["run", "-o", str(out), "--no-color"])
    assert code == 0
    stdout = capsys.readouterr().out
    agg = json.loads(stdout[: stdout.index("results written")])
    assert agg["solves"] == 2
    assert agg["solve_rate"] == 1.0
    assert out.is_file()


def test_headless_run_from_env_missing_key(tmp_path, capsys, monkeypatch):
    """A remote preset without any key in .env is reported clearly."""
    code = main(["run", "-o", str(tmp_path / "out.jsonl")])
    assert code == 2
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err


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
