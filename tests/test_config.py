"""Tests for config persistence, validation, and presets."""

from __future__ import annotations

import pytest

from rubikbench.config import (
    DEFAULT_CONFIG_PATH,
    PRESETS,
    BenchmarkConfig,
    load_config,
    preset_defaults,
    save_config,
)


def test_roundtrip_dict():
    cfg = BenchmarkConfig(
        base_url="http://localhost:8000/v1",
        model="my-model",
        max_turns=12,
        seed=7,
        extra_body={"top_p": 0.9, "max_tokens": 4096},
    )
    restored = BenchmarkConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_unknown_keys_ignored_and_extra_body_string_coerced():
    data = {
        "base_url": "http://x/v1",
        "model": "m",
        "legacy_thing": "whatever",
        "extra_body": '{"reasoning_effort": "low"}',
    }
    cfg = BenchmarkConfig.from_dict(data)
    assert cfg.extra_body == {"reasoning_effort": "low"}
    assert not hasattr(cfg, "legacy_thing")


def test_extra_body_bad_string_graceful():
    cfg = BenchmarkConfig.from_dict({"extra_body": "not json"})
    assert cfg.extra_body == {}


def test_effective_extra_body_merges_reasoning_effort():
    cfg = BenchmarkConfig(extra_body={"temperature": 0.3}, reasoning_effort="high")
    body = cfg.effective_extra_body()
    assert body == {"temperature": 0.3, "reasoning_effort": "high"}
    # explicit field wins over a stray extra_body entry
    cfg2 = BenchmarkConfig(extra_body={"reasoning_effort": "low"}, reasoning_effort="high")
    assert cfg2.effective_extra_body()["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: setattr(c, "base_url", ""), "base URL"),
        (lambda c: setattr(c, "model", ""), "model"),
        (lambda c: setattr(c, "num_solves", 0), "num_solves"),
        (lambda c: setattr(c, "max_turns", 0), "max_turns"),
        (lambda c: setattr(c, "scramble_len", -1), "scramble_len"),
        (lambda c: setattr(c, "timeout", -5.0), "timeout"),
        (lambda c: setattr(c, "par_strategy", "nonsense"), "par_strategy"),
        (lambda c: setattr(c, "seed", -3), "seed"),
        (lambda c: setattr(c, "extra_body", [1, 2]), "extra_body"),
    ],
)
def test_validation_errors(mutate, message):
    cfg = BenchmarkConfig()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_presets_are_sane():
    assert "OpenAI" in PRESETS
    assert PRESETS["OpenAI"]["base_url"] == "https://api.openai.com/v1"
    cfg = preset_defaults("vLLM (local)")
    assert cfg.base_url == "http://localhost:8000/v1"
    with pytest.raises(KeyError):
        preset_defaults("Nope")


def test_save_and_load(tmp_path):
    path = tmp_path / "cfg.json"
    cfg = BenchmarkConfig(model="gpt-4o", seed=42, extra_body={"top_p": 0.5})
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg
    # no stray artifacts at the default path
    assert DEFAULT_CONFIG_PATH.exists() is False or DEFAULT_CONFIG_PATH.name != "rubikbench_config.json"


def test_token_caps_roundtrip():
    cfg = BenchmarkConfig(max_input_tokens=16384, max_output_tokens=4096)
    restored = BenchmarkConfig.from_dict(cfg.to_dict())
    assert restored.max_input_tokens == 16384
    assert restored.max_output_tokens == 4096


@pytest.mark.parametrize(
    "field,value",
    [("max_input_tokens", 0), ("max_input_tokens", -3), ("max_output_tokens", 0)],
)
def test_token_cap_validation(field, value):
    cfg = BenchmarkConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_max_output_tokens_merged_into_body():
    cfg = BenchmarkConfig(max_output_tokens=2048, extra_body={"temperature": 0.4})
    body = cfg.effective_extra_body()
    assert body["max_tokens"] == 2048
    # explicit field wins over a stray extra_body entry
    cfg2 = BenchmarkConfig(max_output_tokens=1024, extra_body={"max_tokens": 99})
    assert cfg2.effective_extra_body()["max_tokens"] == 1024
