"""OpenAI-compatible chat client used by the benchmark loop.

Wraps the ``openai`` SDK with ``base_url``/``api_key`` overrides so any
OpenAI-compatible endpoint works (OpenAI, OpenRouter, DeepSeek, vLLM, Ollama,
LM Studio, ...). Extra body parameters (``reasoning_effort``, provider knobs)
are merged into every request via ``extra_body``, which compatible servers
surface as plain JSON body fields.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
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
    #: Chain-of-thought from reasoning models (reasoning_content / reasoning).
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    latency: float = 0.0
    ttft: float = 0.0


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
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
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stream_idle_timeout: float | None = None,
        loop_detection: bool = True,
        extra_body: dict[str, Any] | None = None,
        tool_choice: str = "auto",
    ) -> None:
        if not api_key:
            api_key = "EMPTY"  # accepted by local servers (vLLM, Ollama, LM Studio)
        self._base_url = base_url
        self._model = model
        self._stream = stream
        self._temperature = temperature
        self._top_p = top_p
        self._max_output_tokens = max_output_tokens
        self._stream_idle_timeout = stream_idle_timeout
        self._loop_detection = loop_detection
        self._extra_body = dict(extra_body or {})
        self._tool_choice = tool_choice
        # The SDK must not retry internally: retries with backoff are owned by
        # the benchmark turn loop (``benchmark.run_solve``), which counts them
        # and bounds them. Internal SDK retries would multiply the configured
        # ``max_retries`` (retryable errors would be retried twice).
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    # -- public -------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
    ) -> AssistantTurn:
        body = {**self._extra_body, **(extra_body or {})}
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, "extra_body": body}
        if tools:
            kwargs["tools"] = tools
        if self._tool_choice and self._tool_choice != "auto":
            kwargs["tool_choice"] = self._tool_choice
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._max_output_tokens:
            # Standard OpenAI-compatible parameter; passed top-level (not
            # inside extra_body) so vLLM/SGLang and OpenAI both accept it.
            kwargs["max_tokens"] = self._max_output_tokens
        if self._stream:
            # Avoid HTTP/gateway buffering of SSE responses.
            kwargs["extra_headers"] = {"Cache-Control": "no-cache"}
            # OpenAI v1 chat completions: request per-stream usage so token
            # counts (prompt/completion/cached/reasoning) arrive in the final
            # chunk. Every OpenAI-compatible server MUST support this.
            kwargs["stream_options"] = {"include_usage": True}

        started = time.monotonic()
        self._request_started = started
        try:
            if self._stream:
                turn = self._complete_stream(on_chunk=on_chunk, **kwargs)
            else:
                response = self._client.chat.completions.create(**kwargs)
                turn = self._parse_response(response)
                # Loop detection is not streaming-only: a non-streaming answer
                # that repeats a short pattern is equally stuck. Raising here
                # lets the benchmark turn loop retry, exactly as with streams.
                if self._loop_detection and self._is_looping(
                    (turn.reasoning or "") + (turn.content or "")
                ):
                    raise LLMError("watchdog: model output is looping; aborted answer")
        except LLMError:
            raise
        except (APIError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError) as exc:
            raise LLMError(_describe_api_error(exc)) from exc
        except Exception as exc:
            raise LLMError(f"request failed: {exc}") from exc
        turn.latency = time.monotonic() - started
        return turn

    @staticmethod
    def _usage_fields(usage: Any) -> tuple[int, int, int, int, int]:
        """(prompt, completion, reasoning, cached, total) provider-reported usage."""
        def get(value: Any, name: str) -> int:
            return (value.get(name, 0) if isinstance(value, dict) else getattr(value, name, 0)) or 0
        prompt = completion = reasoning = cached = total = 0
        if usage is not None:
            prompt, completion, total = get(usage, "prompt_tokens"), get(usage, "completion_tokens"), get(usage, "total_tokens")
            details = get(usage, "prompt_tokens_details")
            if details:
                cached = get(details, "cached_tokens")
            details = get(usage, "completion_tokens_details")
            if details:
                reasoning = get(details, "reasoning_tokens")
        return prompt, completion, reasoning, cached, total

    # -- internals ----------------------------------------------------------
    def _complete_stream(
        self,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> AssistantTurn:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        prompt_tokens = completion_tokens = reasoning_tokens = cached_tokens = total_tokens = 0
        finish_reason: str | None = None
        ttft = 0.0
        order: list[int] = []
        seen_payload = False
        stream = self._client.chat.completions.create(stream=True, **kwargs)

        # Watchdog. A daemon thread closes the response when no meaningful chunk
        # has arrived for ``stream_idle_timeout`` seconds, aborting the blocking
        # iteration below with a retryable error. Loop detection runs inline on
        # the accumulated output and aborts on tight, sustained repetition.
        idle_timeout = self._stream_idle_timeout
        stop = threading.Event()
        fired: dict[str, str | None] = {"reason": None}
        last_activity = time.monotonic()
        activity_lock = threading.Lock()
        if idle_timeout is not None and idle_timeout > 0:
            def _watchdog() -> None:
                while not stop.is_set():
                    stop.wait(0.25)
                    with activity_lock:
                        idle_for = time.monotonic() - last_activity
                    if idle_for > idle_timeout:
                        fired["reason"] = f"no output for {idle_for:.0f}s (idle timeout {idle_timeout:.0f}s)"
                        stream.close()
                        return

            threading.Thread(target=_watchdog, daemon=True, name="rubikbench-watchdog").start()

        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    candidate = self._usage_fields(usage)
                    # Providers may emit partial/cumulative usage only in the
                    # final chunk; never erase a previously reported value.
                    prompt_tokens = max(prompt_tokens, candidate[0])
                    completion_tokens = max(completion_tokens, candidate[1])
                    reasoning_tokens = max(reasoning_tokens, candidate[2])
                    cached_tokens = max(cached_tokens, candidate[3])
                    total_tokens = max(total_tokens, candidate[4])
                delta_content: str | None = None
                delta_reasoning: str | None = None
                tool_deltas_out: list[dict[str, Any]] = []
                delta_finish: str | None = None
                for choice in getattr(chunk, "choices", []) or []:
                    reason = getattr(choice, "finish_reason", None)
                    if reason:
                        finish_reason = reason
                        delta_finish = reason
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    content = getattr(delta, "content", None)
                    if content:
                        content_parts.append(content)
                        delta_content = content
                    # Reasoning models (DeepSeek-R1, QwQ, o-series) stream the
                    # chain of thought under ``reasoning_content`` (vLLM/SGLang)
                    # or ``reasoning``.
                    reasoning = None
                    for attr in ("reasoning_content", "reasoning"):
                        value = getattr(delta, attr, None)
                        if value:
                            reasoning = value
                            break
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        delta_reasoning = reasoning
                    tool_deltas = getattr(delta, "tool_calls", None) or []
                    if (content or reasoning or tool_deltas) and not seen_payload:
                        seen_payload = True
                        ttft = time.monotonic() - self._request_started
                    for tc in tool_deltas:
                        idx = getattr(tc, "index", 0)
                        slot = slots.get(idx)
                        if slot is None:
                            slot = slots[idx] = {"id": "", "name": "", "args": ""}
                            order.append(idx)
                        tc_id = getattr(tc, "id", None)
                        if tc_id:
                            slot["id"] = tc_id
                        fn = getattr(tc, "function", None)
                        frag_name = frag_args = None
                        if fn is not None:
                            name = getattr(fn, "name", None)
                            if name:
                                slot["name"] += name
                                frag_name = name
                            args = getattr(fn, "arguments", None)
                            if args:
                                slot["args"] += args
                                frag_args = args
                        tool_deltas_out.append(
                            {"index": idx, "id": tc_id, "name": frag_name, "arguments": frag_args}
                        )

                if delta_content or delta_reasoning or tool_deltas_out or prompt_tokens or completion_tokens or cached_tokens or total_tokens:
                    with activity_lock:
                        last_activity = time.monotonic()

                if self._loop_detection and self._is_looping(
                    "".join(reasoning_parts) + "".join(content_parts)
                ):
                    stream.close()
                    raise LLMError("watchdog: model output is looping; aborted stream")

                if on_chunk is not None:
                    on_chunk({
                        "content": delta_content,
                        "reasoning": delta_reasoning,
                        "tool_calls": tool_deltas_out or None,
                        "usage": {
                            "prompt": prompt_tokens,
                            "completion": completion_tokens,
                            "reasoning": reasoning_tokens,
                            "cached": cached_tokens,
                            "total": total_tokens,
                        },
                        "finish_reason": delta_finish,
                        "ttft": ttft if seen_payload else None,
                    })
        except LLMError:
            raise
        except Exception as exc:
            reason = fired["reason"]
            if reason:
                raise LLMError(f"watchdog: {reason}") from exc
            raise
        finally:
            stop.set()

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
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        return AssistantTurn(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            ttft=ttft,
        )

    @staticmethod
    def _is_looping(text: str, max_repeats: int = 6, min_cycle: int = 2, max_cycle: int = 24) -> bool:
        """True when the tail of ``text`` repeats a short pattern ``max_repeats`` times.

        Whitespace is ignored so move lists like ``R U R U R U …`` (which arrive
        with a trailing separator that breaks exact-prefix matching) are caught.
        """
        compact = "".join(text.split())
        for cycle_len in range(min_cycle, max_cycle + 1):
            need = cycle_len * max_repeats
            if len(compact) < need:
                continue
            pattern = compact[-cycle_len:]
            if not pattern:
                continue
            if compact[-need:] == pattern * max_repeats:
                return True
        return False

    def _parse_response(self, response: Any) -> AssistantTurn:
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        content = getattr(message, "content", None) if message is not None else None
        reasoning = None
        if message is not None:
            for attr in ("reasoning_content", "reasoning"):
                value = getattr(message, attr, None)
                if value:
                    reasoning = value
                    break
        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
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
        (
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            cached_tokens,
            total_tokens,
        ) = self._usage_fields(getattr(response, "usage", None))
        return AssistantTurn(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
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
