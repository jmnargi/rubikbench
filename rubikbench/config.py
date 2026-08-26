"""Benchmark configuration: dataclass, presets, JSON persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_CONFIG_PATH = Path("rubikbench_config.json")

#: A current OpenRouter free model with tool-calling support. Used as the
#: default when running from ``.env`` without a config file, and when applying
#: the OpenRouter preset. Override with ``RUBIKBENCH_MODEL`` in ``.env``.
OPENROUTER_FREE_MODEL = "google/gemma-4-31b-it:free"

#: Provider presets: name -> base_url, suggested model, env var for the API key.
PRESETS: dict[str, dict[str, str]] = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o", "env": "OPENAI_API_KEY"},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": OPENROUTER_FREE_MODEL, "env": "OPENROUTER_API_KEY"},
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY"},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile", "env": "GROQ_API_KEY"},
    "Mistral": {"base_url": "https://api.mistral.ai/v1", "model": "mistral-large-latest", "env": "MISTRAL_API_KEY"},
    "vLLM (local)": {"base_url": "http://localhost:8000/v1", "model": "", "env": ""},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "llama3.1", "env": ""},
    "LM Studio (local)": {"base_url": "http://localhost:1234/v1", "model": "", "env": ""},
}


@dataclass
class BenchmarkConfig:
    # --- Endpoint ----------------------------------------------------------
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float | None = None
    stream: bool = False
    timeout: float = 120.0
    max_retries: int = 2
    tool_choice: str = "auto"
    #: Extra JSON body params merged into every chat completion request
    #: (e.g. ``{"reasoning_effort": "high"}`` or provider-specific knobs).
    extra_body: dict[str, Any] = field(default_factory=dict)
    #: Convenience field merged into ``extra_body`` under ``reasoning_effort``.
    reasoning_effort: str | None = None
    #: Optional prompt-cache retention request, merged as ``prompt_cache_retention``
    #: (seconds). Cached-token usage is always recorded when the server reports it.
    cache_retention: int | None = None
    #: Output token cap forwarded as ``max_tokens`` in the request body.
    max_output_tokens: int | None = None
    #: Approximate cap on the request context (tokens). Older conversation turns
    #: are trimmed to stay below this value; ``None`` disables trimming.
    max_input_tokens: int | None = None

    # --- Benchmark ---------------------------------------------------------
    num_solves: int = 5
    max_turns: int = 40
    scramble_len: int = 22
    #: RNG seed for scrambles; ``None`` = fresh random run.
    seed: int | None = None
    #: Accept moves written in plain assistant text (not via apply_moves).
    allow_text_moves: bool = True
    #: Custom scrambles file list (used when ``scramble_preset == "file"``).
    scrambles: list[str] = field(default_factory=list)
    #: "" = random; "file" = custom scrambles list; otherwise a premade set name.
    scramble_preset: str = ""

    # --- Scoring -----------------------------------------------------------
    weight_moves: float = 0.5
    weight_turns: float = 0.3
    weight_tools: float = 0.2
    #: "auto" -> kociemba par when available, else "god"; "god" -> God's number; "fixed" -> par_fixed.
    par_strategy: str = "auto"
    par_fixed: int = 20

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base URL is required")
        if not self.model.strip():
            raise ValueError("model name is required")
        if self.num_solves < 1:
            raise ValueError("num_solves must be >= 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.scramble_len < 0:
            raise ValueError("scramble_len must be >= 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.par_strategy not in ("auto", "god", "fixed"):
            raise ValueError("par_strategy must be one of: auto, god, fixed")
        if self.par_fixed < 1:
            raise ValueError("par_fixed must be >= 1")
        if self.tool_choice not in ("auto", "none", "required"):
            raise ValueError("tool_choice must be one of: auto, none, required")
        if not isinstance(self.extra_body, dict):
            raise ValueError("extra_body must be a JSON object")  # noqa: TRY004 - user-facing message
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError("seed must be a non-negative integer or null")
        if self.max_input_tokens is not None and self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1 or null")
        if self.cache_retention is not None and self.cache_retention < 1:
            raise ValueError("cache_retention must be >= 1 or null")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1 or null")
        from .scramble import PREMADE_SCRAMBLES

        if self.scramble_preset not in ("", "file", *PREMADE_SCRAMBLES):
            known = ", ".join(sorted(PREMADE_SCRAMBLES))
            raise ValueError(f"scramble_preset must be '', 'file', or one of: {known}")
        if self.scramble_preset == "file" and not self.scrambles:
            raise ValueError("scramble_preset 'file' requires a non-empty scrambles list")

    # --- Serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        known = {f.name for f in fields(cls)}
        cleaned = {k: v for k, v in data.items() if k in known}
        # Legacy configs used a non-empty scrambles list without a preset name.
        if "scrambles" in data and data["scrambles"] and "scramble_preset" not in data:
            cleaned["scramble_preset"] = "file"
        # Keep configs written by older versions/other tools working.
        if "extra_body" in cleaned and isinstance(cleaned["extra_body"], str):
            try:
                cleaned["extra_body"] = json.loads(cleaned["extra_body"])
            except json.JSONDecodeError:
                cleaned["extra_body"] = {}
        cfg = cls(**cleaned)
        cfg.validate()
        return cfg

    def effective_extra_body(self) -> dict[str, Any]:
        """``extra_body`` plus the explicit reasoning effort and token cap fields."""
        body = dict(self.extra_body or {})
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.cache_retention:
            body["prompt_cache_retention"] = self.cache_retention
        if self.max_output_tokens:
            body["max_tokens"] = self.max_output_tokens
        return body


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> BenchmarkConfig:
    with open(path, encoding="utf-8") as fh:
        return BenchmarkConfig.from_dict(json.load(fh))


def save_config(cfg: BenchmarkConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")


def preset_defaults(name: str, key_from_env: bool = True) -> BenchmarkConfig:
    """Config pre-filled from a preset name. Raises KeyError for unknown presets."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name}")
    p = PRESETS[name]
    cfg = BenchmarkConfig(base_url=p["base_url"], model=p["model"])
    if key_from_env and p.get("env"):
        cfg.api_key = os.environ.get(p["env"], "")
    return cfg


# --------------------------------------------------------------------------- .env

def load_env(path: str | Path = ".env") -> bool:
    """Load ``KEY=value`` pairs from a dotenv file into ``os.environ``.

    Existing environment variables are never overridden; the file only fills
    in values that are not already set. Handles blank lines, ``#`` comments,
    an optional ``export`` prefix, single/double quotes around values, inline
    comments after the value, and whitespace around ``=``. Returns True if the
    file was found and read.
    """
    p = Path(path)
    if not p.is_file():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Cut an inline comment: a '#' that is not inside any quotes.
        quote: str | None = None
        cut = len(value)
        for idx, ch in enumerate(value):
            if ch in "'\"":
                if quote is None:
                    quote = ch
                elif ch == quote:
                    quote = None
            elif ch == "#" and quote is None:
                cut = idx
                break
        value = value[:cut].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return True


#: Host fragments -> env var holding the API key, for matching provider
#: endpoints and any proxy hosted under their domains.
_HOST_ENV: dict[str, str] = {
    "openai.com": "OPENAI_API_KEY",
    "openrouter.ai": "OPENROUTER_API_KEY",
    "deepseek.com": "DEEPSEEK_API_KEY",
    "groq.com": "GROQ_API_KEY",
    "mistral.ai": "MISTRAL_API_KEY",
}

#: Local hosts never need an API key.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def preset_env_for(base_url: str) -> str | None:
    """API key env var declared by the preset matching ``base_url``, if any.

    Matches the exact preset URL first, then by host fragment (so a proxy
    hosted under a provider's domain still resolves to its env var).
    """
    for p in PRESETS.values():
        if p.get("base_url") == base_url and p.get("env"):
            return p["env"]
    host = urlparse(base_url).hostname or ""
    for fragment, env in _HOST_ENV.items():
        if fragment in host:
            return env
    return None


def api_key_source(base_url: str) -> str | None:
    """Env var that should hold the API key for ``base_url``, if any.

    ``RUBIKBENCH_API_KEY`` always wins; otherwise the preset/host-matched
    provider var. For any other endpoint (e.g. a custom LiteLLM proxy)
    ``OPENAI_API_KEY`` is the assumed default, since RubikBench only speaks
    OpenAI-compatible APIs. Returns None only for local servers.
    """
    env = preset_env_for(base_url)
    if env:
        return env
    host = urlparse(base_url).hostname or ""
    if host in _LOCAL_HOSTS or host.endswith(".local"):
        return None
    return "OPENAI_API_KEY"


def api_key_from_env(base_url: str) -> tuple[str, str | None]:
    """``(API key, env var it came from)`` for ``base_url``.

    ``RUBIKBENCH_API_KEY`` always wins over the provider-specific variable
    (e.g. ``OPENAI_API_KEY``). Returns ``("", None)`` when no key is set or
    the endpoint needs none.
    """
    source = api_key_source(base_url)
    if source is None:
        return "", None
    if os.environ.get("RUBIKBENCH_API_KEY"):
        return os.environ["RUBIKBENCH_API_KEY"], "RUBIKBENCH_API_KEY"
    return os.environ.get(source, ""), source


#: ``RUBIKBENCH_*`` variables mapped to config fields and their parsers.
_ENV_KNOBS: dict[str, tuple[str, Any]] = {
    "RUBIKBENCH_MAX_INPUT_TOKENS": ("max_input_tokens", int),
    "RUBIKBENCH_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
    "RUBIKBENCH_SOLVES": ("num_solves", int),
    "RUBIKBENCH_MAX_TURNS": ("max_turns", int),
    "RUBIKBENCH_SCRAMBLE_LEN": ("scramble_len", int),
    "RUBIKBENCH_SEED": ("seed", int),
    "RUBIKBENCH_MAX_RETRIES": ("max_retries", int),
    "RUBIKBENCH_TIMEOUT": ("timeout", float),
    "RUBIKBENCH_TEMPERATURE": ("temperature", float),
}


def apply_env_overrides(cfg: BenchmarkConfig) -> BenchmarkConfig:
    """Copy of ``cfg`` with values from the environment taking precedence.

    Credentials never have to be stored in the config file: an API key in the
    environment (e.g. ``OPENROUTER_API_KEY`` from ``.env``) wins over any key
    saved in the file. ``RUBIKBENCH_*`` variables override the base URL, model,
    and the common benchmark knobs. Raises ValueError on a malformed knob.
    """
    overrides: dict[str, Any] = {}
    url = os.environ.get("RUBIKBENCH_BASE_URL")
    if url:
        overrides["base_url"] = url
    model = os.environ.get("RUBIKBENCH_MODEL")
    if model:
        overrides["model"] = model
    final_url = overrides.get("base_url", cfg.base_url)
    key, _ = api_key_from_env(final_url)
    if key:
        overrides["api_key"] = key
    for env_name, (field_name, cast) in _ENV_KNOBS.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            overrides[field_name] = cast(raw)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a number, got {raw!r}") from exc
    return replace(cfg, **overrides)


def config_from_env() -> BenchmarkConfig:
    """Build a full config from the environment, with no config file needed.

    Defaults to the OpenAI endpoint and model (the project default); override
    the endpoint, model, key, and any knob via ``RUBIKBENCH_*`` variables in
    ``.env`` (e.g. ``RUBIKBENCH_BASE_URL=...``, ``RUBIKBENCH_MODEL=...``,
    ``OPENAI_API_KEY=...`` or ``RUBIKBENCH_API_KEY=...``). Raises ValueError
    on malformed ``RUBIKBENCH_*`` values.
    """
    openai = PRESETS["OpenAI"]
    base_url = os.environ.get("RUBIKBENCH_BASE_URL") or openai["base_url"]
    model = os.environ.get("RUBIKBENCH_MODEL") or openai["model"]
    return apply_env_overrides(BenchmarkConfig(base_url=base_url, model=model))
