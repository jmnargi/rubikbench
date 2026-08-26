"""Tests for config persistence, validation, and presets."""

from __future__ import annotations

import os

import pytest

from rubikbench.config import (
    DEFAULT_CONFIG_PATH,
    OPENROUTER_FREE_MODEL,
    PRESETS,
    BenchmarkConfig,
    apply_env_overrides,
    config_from_env,
    load_config,
    load_env,
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


def test_scramble_preset_validation():
    from rubikbench.config import BenchmarkConfig

    with pytest.raises(ValueError, match="scramble_preset"):
        BenchmarkConfig(scramble_preset="nope").validate()
    with pytest.raises(ValueError, match="requires a non-empty scrambles list"):
        BenchmarkConfig(scramble_preset="file").validate()
    BenchmarkConfig(scramble_preset="superflip").validate()  # does not raise


def test_legacy_scrambles_migration():
    cfg = BenchmarkConfig.from_dict({"scrambles": ["R U F"], "base_url": "http://x/v1", "model": "m"})
    assert cfg.scramble_preset == "file"


# --------------------------------------------------------------------------- .env


def test_load_env_parses_file(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\n"
        "PLAIN=value1\n"
        "export EXPORTED=value2\n"
        "SPACED = value3\n"
        'QUOTED="value 4"\n'
        "SINGLE='v5'\n"
        "INLINE=value6 # trailing comment\n"
        'HASH_IN_QUOTES="a#b"\n'
        "EMPTY=\n"
        "NOEQUALS\n"
    )
    for key in ("PLAIN", "EXPORTED", "SPACED", "QUOTED", "SINGLE", "INLINE", "HASH_IN_QUOTES", "EMPTY", "NOEQUALS"):
        monkeypatch.delenv(key, raising=False)
    assert load_env(dotenv) is True
    assert os.environ["PLAIN"] == "value1"
    assert os.environ["EXPORTED"] == "value2"
    assert os.environ["SPACED"] == "value3"
    assert os.environ["QUOTED"] == "value 4"
    assert os.environ["SINGLE"] == "v5"
    assert os.environ["INLINE"] == "value6"
    assert os.environ["HASH_IN_QUOTES"] == "a#b"
    assert os.environ["EMPTY"] == ""
    assert "NOEQUALS" not in os.environ


def test_load_env_missing_file(tmp_path):
    assert load_env(tmp_path / "nope.env") is False


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAIN", "already-set")
    (tmp_path / ".env").write_text("PLAIN=from-file\n")
    assert load_env(tmp_path / ".env") is True
    assert os.environ["PLAIN"] == "already-set"


def test_config_from_env_defaults_to_openrouter_free(monkeypatch):
    monkeypatch.delenv("RUBIKBENCH_BASE_URL", raising=False)
    monkeypatch.delenv("RUBIKBENCH_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = config_from_env()
    assert cfg.base_url == PRESETS["OpenRouter"]["base_url"]
    assert cfg.model == OPENROUTER_FREE_MODEL
    assert cfg.api_key == "sk-test"


def test_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("RUBIKBENCH_BASE_URL", "http://mock/v1")
    monkeypatch.setenv("RUBIKBENCH_MODEL", "mock-model")
    monkeypatch.setenv("RUBIKBENCH_API_KEY", "k")
    monkeypatch.setenv("RUBIKBENCH_SOLVES", "3")
    monkeypatch.setenv("RUBIKBENCH_MAX_TURNS", "10")
    monkeypatch.setenv("RUBIKBENCH_SEED", "7")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = config_from_env()
    assert (cfg.base_url, cfg.model, cfg.api_key) == ("http://mock/v1", "mock-model", "k")
    assert (cfg.num_solves, cfg.max_turns, cfg.seed) == (3, 10, 7)


def test_config_from_env_bad_knob(monkeypatch):
    monkeypatch.setenv("RUBIKBENCH_SOLVES", "many")
    with pytest.raises(ValueError, match="RUBIKBENCH_SOLVES"):
        config_from_env()


def test_apply_env_overrides_preset_key_wins_over_file(monkeypatch):
    cfg = BenchmarkConfig(base_url=PRESETS["OpenRouter"]["base_url"], api_key="stored", model="stored-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    monkeypatch.delenv("RUBIKBENCH_API_KEY", raising=False)
    monkeypatch.delenv("RUBIKBENCH_MODEL", raising=False)
    out = apply_env_overrides(cfg)
    assert out.api_key == "from-env"
    assert out.model == "stored-model"
    assert out.base_url == cfg.base_url


def test_apply_env_overrides_generic_key_first(monkeypatch):
    cfg = BenchmarkConfig(base_url=PRESETS["OpenRouter"]["base_url"], api_key="stored")
    monkeypatch.setenv("RUBIKBENCH_API_KEY", "generic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "preset")
    out = apply_env_overrides(cfg)
    assert out.api_key == "generic"


def test_apply_env_overrides_local_server_untouched(monkeypatch):
    cfg = BenchmarkConfig(base_url="http://localhost:11434/v1", api_key="")
    monkeypatch.setenv("OPENROUTER_API_KEY", "whatever")
    monkeypatch.delenv("RUBIKBENCH_API_KEY", raising=False)
    out = apply_env_overrides(cfg)
    assert out.api_key == ""
