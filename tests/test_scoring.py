"""Tests for the scoring module."""

from __future__ import annotations

import pytest

from rubikbench.scoring import Weights, compute_score


def test_unsolved_scores_zero():
    b = compute_score(solved=False, par_moves=20, total_moves=30, turns=5, tool_calls=6, max_turns=40)
    assert b.score == 0.0
    assert b.reason == "unsolved"
    assert b.moves_eff == 0.0


def test_perfect_solve_scores_1000():
    b = compute_score(solved=True, par_moves=20, total_moves=20, turns=1, tool_calls=1, max_turns=40)
    assert b.score == 1000.0
    assert b.moves_eff == 1.0
    assert b.turns_eff == 1.0
    assert b.tools_eff == 1.0


def test_moves_penalty_scales_with_par():
    b = compute_score(solved=True, par_moves=20, total_moves=60, turns=1, tool_calls=1, max_turns=40)
    assert b.moves_eff == pytest.approx(20 / 60, abs=1e-3)
    b2 = compute_score(solved=True, par_moves=20, total_moves=120, turns=1, tool_calls=1, max_turns=40)
    assert b2.score < b.score


def test_better_than_par_capped_at_one():
    b = compute_score(solved=True, par_moves=20, total_moves=10, turns=1, tool_calls=1, max_turns=40)
    assert b.moves_eff == 1.0


def test_turns_penalty():
    a = compute_score(solved=True, par_moves=20, total_moves=30, turns=3, tool_calls=3, max_turns=40)
    b = compute_score(solved=True, par_moves=20, total_moves=30, turns=12, tool_calls=3, max_turns=40)
    assert b.score < a.score
    assert 0.0 < b.turns_eff <= a.turns_eff


def test_tool_calls_penalty():
    a = compute_score(solved=True, par_moves=20, total_moves=30, turns=5, tool_calls=5, max_turns=40)
    b = compute_score(solved=True, par_moves=20, total_moves=30, turns=5, tool_calls=14, max_turns=40)
    assert b.score < a.score
    assert b.tools_eff < a.tools_eff


def test_weights_normalize():
    w = Weights(50, 30, 20).validate()
    assert w.moves == pytest.approx(0.5)
    assert w.turns == pytest.approx(0.3)
    assert w.tools == pytest.approx(0.2)


def test_invalid_weights():
    with pytest.raises(ValueError):
        Weights(0, 0, 0).validate()
    with pytest.raises(ValueError):
        Weights(-1, 1, 1).validate()


def test_zero_moves_solved_is_perfect_moves():
    b = compute_score(solved=True, par_moves=20, total_moves=0, turns=1, tool_calls=1, max_turns=40)
    assert b.moves_eff == 1.0
