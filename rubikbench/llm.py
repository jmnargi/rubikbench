"""OpenAI-compatible chat client used by the benchmark loop.

Wraps the ``openai`` SDK with ``base_url``/``api_key`` overrides so any
OpenAI-compatible endpoint works (OpenAI, OpenRouter, DeepSeek, vLLM, Ollama,
LM Studio, ...). Extra body parameters (``reasoning_effort``, provider knobs)
are merged into every request via ``extra_body``, which compatible servers
surface as plain JSON body fields.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


class LLMError(Exception):
    """Raised for any failure talking to the chat endpoint."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
    ) -> AssistantTurn: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 0,
        stream: bool = False,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        tool_choice: str = "auto",
    ) -> None:
        if not api_key:
            api_key = "EMPTY"  # accepted by local servers (vLLM, Ollama, LM Studio)
        self._base_url = base_url
        self._model = model
        self._stream = stream
        self._temperature = temperature
        self._extra_body = dict(extra_body or {})
        self._tool_choice = tool_choice
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,  # we do our own retries with backoff in the loop
        )

    # -- public -------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
    ) -> AssistantTurn:
        body = {**self._extra_body, **(extra_body or {})}
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, "extra_body": body}
        if tools:
            kwargs["tools"] = tools
        if self._tool_choice and self._tool_choice != "auto":
            kwargs["tool_choice"] = self._tool_choice
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        started = time.monotonic()
        try:
            if self._stream:
                turn = self._complete_stream(**kwargs)
            else:
                response = self._client.chat.completions.create(**kwargs)
                turn = self._parse_response(response)
        except (APIError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError) as exc:
            raise LLMError(_describe_api_error(exc)) from exc
        except Exception as exc:
            raise LLMError(f"request failed: {exc}") from exc
        turn.latency = time.monotonic() - started
        return turn

    # -- internals ----------------------------------------------------------
    def _complete_stream(self, **kwargs: Any) -> AssistantTurn:
        content_parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        prompt_tokens = completion_tokens = 0
        order: list[int] = []
        stream = self._client.chat.completions.create(stream=True, **kwargs)
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens) or completion_tokens
            for choice in getattr(chunk, "choices", []) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", 0)
                    slot = slots.get(idx)
                    if slot is None:
                        slot = slots[idx] = {"id": "", "name": "", "args": ""}
                        order.append(idx)
                    tc_id = getattr(tc, "id", None)
                    if tc_id:
                        slot["id"] = tc_id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        name = getattr(fn, "name", None)
                        if name:
                            slot["name"] += name
                        args = getattr(fn, "arguments", None)
                        if args:
                            slot["args"] += args
        tool_calls: list[ToolCall] = []
        for idx in order:
            slot = slots[idx]
            raw = slot["args"]
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_unparsed": raw}
            tool_calls.append(ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"], arguments=arguments, raw=raw))
        content = "".join(content_parts) if content_parts else None
        return AssistantTurn(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _parse_response(self, response: Any) -> AssistantTurn:
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        content = getattr(message, "content", None) if message is not None else None
        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw = getattr(fn, "arguments", "") or "" if fn is not None else ""
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                arguments = {"_unparsed": raw}
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or f"call_{len(tool_calls)}",
                    name=getattr(fn, "name", "") if fn is not None else "",
                    arguments=arguments,
                    raw=raw,
                )
            )
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage is not None else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage is not None else 0
        return AssistantTurn(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _describe_api_error(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict) and body.get("error"):
        err = body["error"]
        if isinstance(err, dict):
            detail = err.get("message") or err.get("type") or str(err)
        else:
            detail = str(err)
    message = str(exc) if not detail else detail
    name = type(exc).__name__.replace("Error", "")
    return f"{name}: {message}" if name else message
