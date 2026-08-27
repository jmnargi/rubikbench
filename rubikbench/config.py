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
    #: Nucleus sampling (standard OpenAI ``top_p``). ``None`` omits the field.
    top_p: float | None = None
    #: vLLM/llama.cpp-style repetition penalty, merged into ``extra_body``.
    repetition_penalty: float | None = None
    #: vLLM-style top-k sampling, merged into ``extra_body``.
    top_k: int | None = None
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
    #: Output cap forwarded as the standard ``max_tokens`` parameter.
    max_output_tokens: int | None = None
    #: Model context-window capacity. Input is budgeted after reserving output.
    context_window_tokens: int | None = None
    #: Optional output reserve; defaults to ``max_output_tokens`` when unset.
    output_token_reserve: int | None = None
    #: Approximate input budget override for legacy configurations. New configs
    #: should set ``context_window_tokens`` instead.
    max_input_tokens: int | None = None
    #: Margin reserved for provider request framing.
    context_safety_margin: int = 128
    #: Abort a streaming request and retry if no chunk arrives for this many
    #: seconds (idle watchdog). ``None`` disables it and relies on ``timeout``.
    stream_idle_timeout: float | None = None
    #: Abort a streaming request when the model's output loops (repeats a short
    #: pattern). Safe to leave on; only fires on clear, tight repetition.
    loop_detection: bool = True

    # --- Benchmark ---------------------------------------------------------
    num_solves: int = 5
    max_turns: int = 40
    scramble_len: int = 22
    #: RNG seed for scrambles; ``None`` = fresh random run.
    seed: int | None = None
    #: Strict tool-only is the benchmark protocol. Text parsing is compatibility-only.
    allow_text_moves: bool = False
    #: Persisted protocol label, preventing comparison of incompatible runs.
    protocol_mode: str = "tool_only"
    #: Sticker is the spatial benchmark; cubie-v1 is an explicitly labeled view.
    presentation_mode: str = "stickers-v1"
    #: Named run profile; explicit values remain authoritative.
    profile: str = "full"
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
        if self.context_window_tokens is not None and self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be >= 1 or null")
        if self.output_token_reserve is not None and self.output_token_reserve < 1:
            raise ValueError("output_token_reserve must be >= 1 or null")
        if self.context_safety_margin < 0:
            raise ValueError("context_safety_margin must be >= 0")
        if self.protocol_mode not in ("tool_only", "text_compat"):
            raise ValueError("protocol_mode must be 'tool_only' or 'text_compat'")
        if self.presentation_mode not in ("stickers-v1", "cubie-v1"):
            raise ValueError("presentation_mode must be 'stickers-v1' or 'cubie-v1'")
        if self.profile not in RUN_PROFILES:
            raise ValueError(f"profile must be one of: {', '.join(RUN_PROFILES)}")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1 or null")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1] or null")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or null")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0 or null")
        if self.stream_idle_timeout is not None and self.stream_idle_timeout <= 0:
            raise ValueError("stream_idle_timeout must be > 0 or null")
        from .scramble import PREMADE_SCRAMBLES

        if self.scramble_preset not in ("", "file", *PREMADE_SCRAMBLES):
            known = ", ".join(sorted(PREMADE_SCRAMBLES))
            raise ValueError(f"scramble_preset must be '', 'file', or one of: {known}")
        if self.scramble_preset == "file" and not self.scrambles:
            raise ValueError("scramble_preset 'file' requires a non-empty scrambles list")

    def to_dict(self, *, include_credentials: bool = False) -> dict[str, Any]:
        """Safe persistence/export representation; credentials never leave memory."""
        data = asdict(self)
        if not include_credentials:
            data.pop("api_key", None)
        return data
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
        """``extra_body`` plus explicit reasoning/cache knobs.

        ``max_output_tokens`` is intentionally omitted: for OpenAI-compatible
        endpoints it is a standard top-level ``max_tokens`` parameter, passed
        directly to ``chat.completions.create`` rather than buried in
        ``extra_body``.
        """
        body = dict(self.extra_body or {})
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.cache_retention:
            body["prompt_cache_retention"] = self.cache_retention
        if self.repetition_penalty is not None:
            body["repetition_penalty"] = self.repetition_penalty
        if self.top_k is not None:
            body["top_k"] = self.top_k
        return body


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> BenchmarkConfig:
    with open(path, encoding="utf-8") as fh:
        return BenchmarkConfig.from_dict(json.load(fh))


def save_config(cfg: BenchmarkConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist configuration without writing credentials to disk."""
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")


RUN_PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"num_solves": 1, "max_turns": 12, "max_output_tokens": 2048},
    "diagnostic": {"num_solves": 2, "max_turns": 30, "max_output_tokens": 4096},
    "full": {"num_solves": 5, "max_turns": 40, "max_output_tokens": 8192},
    "research": {"num_solves": 10, "max_turns": 80, "max_output_tokens": 8192},
}


def apply_profile(cfg: BenchmarkConfig, profile: str) -> BenchmarkConfig:
    """Apply a named conservative profile; callers may override fields afterward."""
    if profile not in RUN_PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    return replace(cfg, profile=profile, **RUN_PROFILES[profile])


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
    "RUBIKBENCH_TOP_P": ("top_p", float),
    "RUBIKBENCH_REPETITION_PENALTY": ("repetition_penalty", float),
    "RUBIKBENCH_TOP_K": ("top_k", int),
    "RUBIKBENCH_STREAM_IDLE_TIMEOUT": ("stream_idle_timeout", float),
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
    loop_opt = os.environ.get("RUBIKBENCH_LOOP_DETECTION", "").lower()
    no_loop_opt = os.environ.get("RUBIKBENCH_NO_LOOP_DETECTION", "").lower()
    if loop_opt in ("1", "true", "yes"):
        overrides["loop_detection"] = True
    elif no_loop_opt in ("1", "true", "yes"):
        overrides["loop_detection"] = False
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
