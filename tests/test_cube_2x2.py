"""2x2x2 cube mode: geometry, solver, config/CLI wiring, and end-to-end solves."""

from __future__ import annotations

import json
import random

import pytest

from rubikbench.config import BenchmarkConfig, apply_env_overrides, save_config
from rubikbench.cube import ALL_MOVES, Cube, parse_moves, solved_facelets
from rubikbench.prompts import SYSTEM_PROMPT, build_system_prompt, initial_user_prompt
from rubikbench.rendering import render_faces, render_plain
from rubikbench.scramble import (
    assert_valid_scramble,
    scramble_from_string,
    scramble_to_string,
)
from rubikbench.solver_ref import GODS_NUMBER_2X2, par_moves_2x2, solve_2x2


def _random_moves(rng: random.Random, length: int) -> list[str]:
    seq: list[str] = []
    prev = ""
    while len(seq) < length:
        face = rng.choice("URFDLB")
        if face != prev:
            seq.append(face + rng.choice(("", "'", "2")))
            prev = face
    return seq


# --------------------------------------------------------------------------- geometry

def test_solved_2x2_has_24_facelets():
    cube = Cube.solved(size=2)
    assert len(cube.facelets) == 24
    assert len(cube.facelet_string()) == 24
    assert cube.is_solved()
    assert cube.facelet_string() == "U" * 4 + "R" * 4 + "F" * 4 + "D" * 4 + "L" * 4 + "B" * 4


def test_solved_facelets_helper():
    assert "".join(solved_facelets(2)) == "U" * 4 + "R" * 4 + "F" * 4 + "D" * 4 + "L" * 4 + "B" * 4


def test_each_2x2_move_breaks_solved_and_inverse_restores():
    """Every one of the 18 face moves rotates the 2x2 cube; its inverse undoes it."""
    for move in ALL_MOVES:
        cube = Cube.solved(size=2)
        cube.apply([move])
        assert not cube.is_solved(), move
        assert cube.history == [move]
        inverse = move[:-1] if move.endswith("'") else move + "'"
        if move.endswith("2"):
            inverse = move
        cube.apply([inverse])
        assert cube.is_solved(), move


def test_2x2_move_applied_four_times_is_identity():
    for move in ALL_MOVES:
        start = Cube.solved(size=2)
        cube = Cube.solved(size=2)
        for _ in range(4):
            cube.apply([move])
        assert cube.facelet_string() == start.facelet_string(), move


def test_2x2_scramble_reset_and_round_trip():
    seq = _random_moves(random.Random(7), 12)
    cube = Cube.solved(size=2)
    cube.scramble = seq
    cube.reset_to_scramble()
    assert not cube.is_solved()
    # Applying the same scramble from the solved state reproduces the same state.
    other = Cube.solved(size=2)
    other.apply(seq)
    assert other.facelet_string() == cube.facelet_string()


def test_2x2_facelet_length_is_validated():
    for bad_len in (22, 25, 54):
        with pytest.raises(ValueError, match="24 facelets"):
            Cube(list("U" * bad_len), size=2)
    with pytest.raises(ValueError, match="size 3 cube needs 54"):
        Cube(list("U" * 24), size=3)


def test_cube_size_is_validated():
    for bad_size in (1, 4, 0):
        with pytest.raises(ValueError, match="unsupported cube size"):
            Cube.solved(size=bad_size)


def test_legacy_positional_constructor_keeps_size_last():
    cube = Cube(list(solved_facelets(2)), ["R"], ["R"], 2)
    assert cube.size == 2
    assert cube.is_solved()


def test_2x2_rendering_shapes():
    cube = Cube.solved(size=2)
    net = render_plain(cube.facelets)
    assert len(net.splitlines()) == 6  # 3 rows of faces x 2 rows each
    faces = render_faces(cube.facelets)
    assert "U (white)" in faces and "B (blue)" in faces
    assert faces.count("U U") == 2  # the U grid occupies 2 rows of 2 facelets


def test_3x3_still_solved_default():
    cube = Cube.solved()
    assert cube.size == 3
    assert len(cube.facelets) == 54
    assert cube.is_solved()


# --------------------------------------------------------------------------- solver

def test_solve_2x2_solved_and_invalid_inputs():
    assert solve_2x2(Cube.solved(size=2).facelet_string()) == []
    assert solve_2x2("U" * 54) is None          # 3x3 facelet count
    assert solve_2x2("X" * 24) is None          # illegal colors
    assert solve_2x2("URFDLB" * 4) is None      # not a legal piece arrangement


def test_solve_2x2_single_moves_are_optimal():
    for move in ALL_MOVES:
        cube = Cube.solved(size=2)
        cube.apply([move])
        solution = solve_2x2(cube.facelet_string())
        assert solution is not None
        assert len(solution) == 1, (move, solution)
        restored = cube.copy()
        restored.apply(solution)
        assert restored.is_solved(), (move, solution)


def test_solve_2x2_round_trip_random_states():
    for trial in range(4):
        cube = Cube.solved(size=2)
        cube.apply(_random_moves(random.Random(1000 + trial), 22))
        solution = solve_2x2(cube.facelet_string())
        assert solution is not None, trial
        assert 0 < len(solution) <= GODS_NUMBER_2X2, (trial, len(solution))
        restored = cube.copy()
        restored.apply(solution)
        assert restored.is_solved(), (trial, solution)
        assert 0 < par_moves_2x2(cube.facelet_string()) <= GODS_NUMBER_2X2


# --------------------------------------------------------------------------- config

def test_config_cube_size_validation():
    cube = BenchmarkConfig(cube_size=2)
    cube.validate()
    with pytest.raises(ValueError, match="cube_size must be 2 or 3"):
        BenchmarkConfig(cube_size=4).validate()
    with pytest.raises(ValueError, match="cube_size 2"):
        BenchmarkConfig(cube_size=2, presentation_mode="cubie-v1").validate()
    BenchmarkConfig(cube_size=2, presentation_mode="stickers-v1").validate()


def test_env_knob_rubikbench_cube_size(monkeypatch):
    monkeypatch.setenv("RUBIKBENCH_CUBE_SIZE", "2")
    cfg = apply_env_overrides(BenchmarkConfig())
    assert cfg.cube_size == 2
    monkeypatch.setenv("RUBIKBENCH_CUBE_SIZE", "3")
    assert apply_env_overrides(BenchmarkConfig(cube_size=2)).cube_size == 3


def test_config_roundtrip_preserves_cube_size(tmp_path):
    path = tmp_path / "cfg.json"
    save_config(BenchmarkConfig(cube_size=2), path)
    loaded = json.loads(path.read_text())
    assert loaded["cube_size"] == 2


# --------------------------------------------------------------------------- prompts / scramble

def test_build_system_prompt_size():
    assert "2x2x2" in build_system_prompt(2)
    assert "3x3x3" in build_system_prompt(3)
    assert build_system_prompt(3) == SYSTEM_PROMPT  # byte-identical default


def test_initial_prompt_2x2_has_24_facelets_and_2x2_grid():
    cube = Cube.solved(size=2)
    prompt = initial_user_prompt(cube, "stickers-v1")
    assert cube.facelet_string() in prompt
    assert "2x2 grid" in prompt
    assert "3x3" not in prompt


def test_scramble_validation_respects_size():
    moves = ["R", "U", "F'", "R2"]
    assert_valid_scramble(moves, size=2)
    assert_valid_scramble(moves, size=3)
    with pytest.raises(ValueError, match="invalid scramble tokens"):
        assert_valid_scramble(["S"], size=2)
    assert parse_moves(scramble_to_string(moves))[1] == []
    assert scramble_from_string(scramble_to_string(moves)) == moves


# --------------------------------------------------------------------------- CLI + end-to-end

@pytest.fixture(autouse=True)
def _clean_rubikbench_env(monkeypatch):
    """Isolate tests from any real .env the developer may have."""
    for name in (
        "RUBIKBENCH_BASE_URL", "RUBIKBENCH_API_KEY", "RUBIKBENCH_MODEL",
        "RUBIKBENCH_SOLVES", "RUBIKBENCH_CUBE_SIZE", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(name, "")


def _read_run_header(out):
    return json.loads(out.read_text().splitlines()[0])


def test_cli_run_2x2_against_mock(tmp_path):
    """A full 2x2 benchmark run solves via the 2x2 reference solver, headless."""
    from mock_openai import start_mock_server

    from rubikbench.cli import main

    server, url = start_mock_server()
    try:
        cfg_path = tmp_path / "cfg.json"
        save_config(
            BenchmarkConfig(base_url=url, api_key="k", model="mock",
                            num_solves=2, max_turns=20, seed=3, cube_size=2),
            cfg_path,
        )
        out = tmp_path / "out.jsonl"
        code = main(["run", "--config", str(cfg_path), "-o", str(out), "--no-color"])
        assert code == 0
        header = _read_run_header(out)
        assert header["config"]["cube_size"] == 2
        lines = out.read_text().splitlines()[1:]
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert record["solved"] is True
            assert record["total_moves"] > 0
    finally:
        server.shutdown()


def test_cli_cube_size_flag_overrides_config(tmp_path):
    """--cube-size beats the config file value."""
    from mock_openai import start_mock_server

    from rubikbench.cli import main

    server, url = start_mock_server()
    try:
        cfg_path = tmp_path / "cfg.json"
        save_config(
            BenchmarkConfig(base_url=url, api_key="k", model="mock",
                            num_solves=1, max_turns=20, seed=3, cube_size=3),
            cfg_path,
        )
        out = tmp_path / "out.jsonl"
        code = main(["run", "--config", str(cfg_path), "--cube-size", "2",
                     "-o", str(out), "--no-color"])
        assert code == 0
        assert _read_run_header(out)["config"]["cube_size"] == 2
    finally:
        server.shutdown()


def test_cli_env_cube_size_wins_over_config(tmp_path, monkeypatch):
    """RUBIKBENCH_CUBE_SIZE beats the config file value; the flag beats both."""
    from mock_openai import start_mock_server

    from rubikbench.cli import main

    server, url = start_mock_server()
    try:
        cfg_path = tmp_path / "cfg.json"
        save_config(
            BenchmarkConfig(base_url=url, api_key="k", model="mock",
                            num_solves=1, max_turns=20, seed=3, cube_size=3),
            cfg_path,
        )
        monkeypatch.setenv("RUBIKBENCH_CUBE_SIZE", "2")
        out = tmp_path / "out.jsonl"
        assert main(["run", "--config", str(cfg_path), "-o", str(out), "--no-color"]) == 0
        assert _read_run_header(out)["config"]["cube_size"] == 2

        out2 = tmp_path / "out2.jsonl"
        assert main(["run", "--config", str(cfg_path), "--cube-size", "3",
                     "-o", str(out2), "--no-color"]) == 0
        assert _read_run_header(out2)["config"]["cube_size"] == 3
    finally:
        server.shutdown()


def test_cli_rejects_invalid_cube_size(tmp_path, capsys):
    from rubikbench.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--cube-size", "4"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "4" in err


def test_2x2_config_env_without_config_file(tmp_path, monkeypatch):
    """cube_size also reaches a run configured purely from the environment."""
    from mock_openai import start_mock_server

    from rubikbench.cli import main

    server, url = start_mock_server()
    try:
        monkeypatch.setenv("RUBIKBENCH_BASE_URL", url)
        monkeypatch.setenv("RUBIKBENCH_API_KEY", "env-key")
        monkeypatch.setenv("RUBIKBENCH_MODEL", "mock")
        monkeypatch.setenv("RUBIKBENCH_SOLVES", "1")
        monkeypatch.setenv("RUBIKBENCH_CUBE_SIZE", "2")
        out = tmp_path / "out.jsonl"
        code = main(["run", "-o", str(out), "--no-color"])
        assert code == 0
        assert _read_run_header(out)["config"]["cube_size"] == 2
    finally:
        server.shutdown()
