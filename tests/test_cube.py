"""Tests for the cube model, scramble generation, solver reference, and rendering."""

from __future__ import annotations

import random

import pytest

from rubikbench.cube import ALL_MOVES, MOVES, Cube, parse_moves
from rubikbench.rendering import render_colored, render_plain
from rubikbench.scramble import (
    random_scramble,
    scramble_from_string,
    scramble_is_valid,
    solution_for_scramble,
)
from rubikbench.solver_ref import par_moves, solve_standard

pytest.importorskip("kociemba")
import kociemba


def apply_seq(seq: list[str]) -> Cube:
    cube = Cube.solved()
    cube.apply(seq)
    return cube


def apply_kociemba_solution(cube: Cube, solution: str) -> None:
    """Apply a kociemba solution string, mirroring its inverted semantics."""
    for raw in solution.split():
        if raw.endswith("2"):
            cube.apply([raw])
        elif raw.endswith("'"):
            cube.apply([raw[:-1]])
        else:
            cube.apply([raw + "'"])


def test_all_moves_are_valid_permutations():
    for name, perm in MOVES.items():
        assert len(perm) == 54
        assert sorted(perm) == list(range(54)), name


def test_four_quarter_turns_are_identity():
    for name in ALL_MOVES:
        if name.endswith("2"):
            continue
        cube = Cube.solved()
        for _ in range(4):
            cube.apply([name])
        assert cube.is_solved(), name


def test_double_move_equals_two_quarters():
    for face in "URFDLB":
        a = Cube.solved()
        a.apply([face + "2"])
        b = Cube.solved()
        b.apply([face, face])
        assert a.facelet_string() == b.facelet_string(), face


def test_prime_is_inverse():
    for face in "URFDLB":
        for seq in ([face, face + "'"], [face + "'", face]):
            cube = Cube.solved()
            cube.apply(seq)
            assert cube.is_solved(), seq


def test_opposite_faces_commute():
    for f1, f2 in (("U", "D"), ("L", "R"), ("F", "B")):
        a = Cube.solved()
        a.apply([f1, f2])
        b = Cube.solved()
        b.apply([f2, f1])
        assert a.facelet_string() == b.facelet_string(), (f1, f2)


def test_sexy_move_six_times_is_identity():
    cube = Cube.solved()
    for _ in range(6):
        cube.apply(["R", "U", "R'", "U'"])
    assert cube.is_solved()


def test_history_and_reset():
    cube = Cube.solved()
    cube.apply(["R", "U"])
    assert cube.history == ["R", "U"]
    cube.scramble = ["F", "D'"]
    cube.reset_to_scramble()
    assert cube.history == []
    expected = Cube.solved()
    expected.apply(["F", "D'"])
    assert cube.facelet_string() == expected.facelet_string()
    assert not cube.is_solved()


def test_parse_moves():
    valid, invalid = parse_moves("R U R' U' F2  r D  junk UU")
    assert valid == ["R", "U", "R'", "U'", "F2", "R", "D"]
    assert parse_moves("") == ([], [])
    valid, invalid = parse_moves("b2 l' f")
    assert valid == ["B2", "L'", "F"]
    assert not invalid


def test_scramble_generation():
    rng = random.Random(7)
    for _ in range(100):
        s = random_scramble(rng, 22)
        assert len(s) == 22
        for prev, cur in zip(s, s[1:]):  # noqa: RUF007 - clarity
            assert prev[0] != cur[0]
        assert scramble_is_valid(s)


def test_scramble_parse_and_solution():
    s = scramble_from_string("R U R' U' F2")
    assert s == ["R", "U", "R'", "U'", "F2"]
    cube = apply_seq(s)
    assert cube.is_solved() is False
    for m in solution_for_scramble(s):
        cube.apply([m])
    assert cube.is_solved()
    with pytest.raises(ValueError):
        scramble_from_string("R U X")


def test_kociemba_roundtrip_scrambles():
    rng = random.Random(1234)
    for _ in range(25):
        scramble = random_scramble(rng, 22)
        cube = apply_seq(scramble)
        apply_kociemba_solution(cube, kociemba.solve(cube.facelet_string()))
        assert cube.is_solved(), scramble


def test_kociemba_agrees_on_single_moves():
    for move in ALL_MOVES:
        cube = apply_seq([move])
        apply_kociemba_solution(cube, kociemba.solve(cube.facelet_string()))
        assert cube.is_solved(), move


def test_solver_ref():
    cube = apply_seq(random_scramble(random.Random(5), 22))
    solution = solve_standard(cube.facelet_string())
    assert solution is not None
    cube.apply(solution)
    assert cube.is_solved()
    assert par_moves("x" * 54, is_solved=True) == 0
    solved = Cube.solved()
    assert par_moves(solved.facelet_string(), is_solved=True) == 0


def test_rendering():
    cube = Cube.solved()
    plain = render_plain(cube.facelets)
    flat = plain.replace(" ", "").replace("\n", "")
    assert len(flat) == 54
    assert sorted(flat) == sorted("URFDLB" * 9)
    colored = render_colored(cube.facelets)
    assert colored.plain.replace("█", "").strip() == ""
    assert any("on " in (span.style or "") for span in colored.spans)

    scrambled = apply_seq(["R", "U", "F"])
    assert render_plain(scrambled.facelets) != render_plain(cube.facelets)
