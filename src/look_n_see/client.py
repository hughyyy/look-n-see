"""Model clients: HuggingFace Inference + OpenRouter, retries and fallbacks."""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from huggingface_hub import InferenceClient

from .config import Config
from .openrouter import OpenRouterTransport, RawResult, _content_to_text

# 408/425 are transient request/timeout states; 529 is provider overload.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
# A server-supplied Retry-After or exponential delay never exceeds this.
MAX_RETRY_DELAY = 60.0


class InferenceError(RuntimeError):
    """Raised when the model call ultimately fails.

    Carries the retry/latency metadata of the failed attempts so the benchmark
    harness can attribute failures instead of recording a bare `retries: 0`.
    """

    def __init__(
        self,
        message: str,
        *,
        retries: int = 0,
        latency_s: float = 0.0,
        deadline_exceeded: bool = False,
    ):
        super().__init__(message)
        self.retries = retries
        self.latency_s = latency_s
        self.deadline_exceeded = deadline_exceeded


@dataclass(frozen=True)
class Completion:
    """Result of one tool call: text plus the metadata needed for benchmarking."""

    text: str
    model: str
    provider: str
    transport: str
    retries: int
    latency_s: float
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "response", None)
    if status is not None:
        code = getattr(status, "status_code", None)
        if code is not None:
            return code in RETRYABLE_STATUS
    # Connection/timeout errors from the underlying HTTP layer. httpx raises
    # its own ConnectError (not a builtin ConnectionError), which is the
    # classic transient failure — retry it.
    return isinstance(exc, (ConnectionError, TimeoutError)) or exc.__class__.__name__ in {
        "ConnectionError",
        "ConnectError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
    }


def _retry_after_seconds(exc: Exception) -> float | None:
    """Parse a numeric Retry-After header (HTTP-date form is ignored)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # pragma: no cover - exotic header containers
        return None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _retry_delay(exc: Exception, attempt: int, base: float) -> float:
    """Exponential backoff with equal jitter, honoring Retry-After."""
    delay = max(0.0, base) * (2**attempt)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        delay = max(delay, retry_after)
    delay = min(delay, MAX_RETRY_DELAY)
    # Equal jitter: half fixed, half random, so parallel callers decorrelate.
    return delay / 2 + random.uniform(0, delay / 2)


def _remaining(deadline_at: float | None) -> float | None:
    return None if deadline_at is None else deadline_at - time.monotonic()


def _describe_error(exc: Exception) -> str:
    """Human-readable error text; prefers the API's own error message."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        message = body.get("error")
        if isinstance(message, dict):
            message = message.get("message")
        if message:
            return f"{response.status_code}: {message}"
        return str(exc)
    return str(exc)


class _HFTransport:
    """Adapter over huggingface_hub InferenceClient exposing RawResult."""

    def __init__(self, config: Config, client: InferenceClient | None = None):
        self.config = config
        # The constructor must not pin an HF model id: huggingface_hub maps
        # it to the HF router, and provider-prefixed or lowercase slugs are
        # rejected there. Per-call model ids are always passed explicitly, so
        # only a dedicated Endpoint URL belongs in the constructor.
        self._client = client or InferenceClient(
            model=config.endpoint,
            token=os.environ.get("HF_TOKEN"),
            timeout=config.timeout,
        )
        # InferenceClient takes its timeout from an instance attribute and
        # exposes no per-call override; serialize + restore it so a shrinking
        # deadline budget can bound each request without racing other calls.
        self._timeout_lock = threading.Lock()

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        timeout: float | None = None,
    ) -> RawResult:
        # A dedicated HF_ENDPOINT pins the target; per-call model ids then only
        # matter for other providers (OpenRouter).
        call_model = self.config.target if self.config.endpoint else model
        request_timeout = timeout if timeout is not None else self.config.timeout
        with self._timeout_lock:
            previous = getattr(self._client, "timeout", None)
            if previous is not None:
                self._client.timeout = request_timeout
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    model=call_model,
                    max_tokens=max_tokens,
                )
            finally:
                if previous is not None:
                    self._client.timeout = previous
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise InferenceError("HF response contained no choices")
        choice = choices[0]
        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            usage = usage or None
        elif usage is not None:
            usage = getattr(usage, "__dict__", None) or None
        return RawResult(
            text=_content_to_text(getattr(getattr(choice, "message", None), "content", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class VisionClient:
    def __init__(self, config: Config, client: InferenceClient | None = None):
        self.config = config
        self._hf = _HFTransport(config, client)
        self._openrouter: OpenRouterTransport | None = None
        self._transport_lock = threading.Lock()

    def _transport_for(self, provider: str) -> _HFTransport | OpenRouterTransport:
        if provider == "openrouter":
            with self._transport_lock:
                if self._openrouter is None:
                    if not self.config.openrouter_api_key:
                        raise InferenceError(
                            "OpenRouter model requested but OPENROUTER_API_KEY is not set"
                        )
                    self._openrouter = OpenRouterTransport(self.config)
            return self._openrouter
        return self._hf

    def close(self) -> None:
        """Release transport HTTP clients."""
        self._hf.close()
        if self._openrouter is not None:
            self._openrouter.close()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        fallbacks: tuple[str, ...] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run chat completion and return the assistant text.

        `messages` is OpenAI-style chat content (text + image parts), making
        this a pure function of the message list — the seam for future
        session-state history/context injection.
        """
        return self.complete_with_meta(
            messages, model=model, fallbacks=fallbacks, max_tokens=max_tokens
        ).text

    def complete_with_meta(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        fallbacks: tuple[str, ...] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Like `complete`, but returns text plus model/retry/usage metadata.

        The effective model chain is `model or config.model` followed by the
        configured fallbacks; each entry is tried with the full retry policy.
        Pass `fallbacks=()` to disable the configured chain (benchmarking).
        The whole chain shares one `VISION_DEADLINE` wall-clock budget.
        """
        chain = [self.config.normalize_model_id(model or self.config.model)]
        chain.extend(
            self.config.normalize_model_id(entry)
            for entry in (fallbacks if fallbacks is not None else self.config.fallbacks)
        )
        deadline_at = (
            time.monotonic() + self.config.deadline if self.config.deadline > 0 else None
        )
        seen: set[str] = set()
        errors: list[str] = []
        details: list[str] = []
        total_retries = 0
        total_latency = 0.0
        for candidate in chain:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                return self._complete_one(candidate, messages, max_tokens, deadline_at)
            except InferenceError as exc:
                total_retries += exc.retries
                total_latency += exc.latency_s
                if exc.deadline_exceeded:
                    raise InferenceError(
                        f"vision call deadline exceeded ({self.config.deadline:g}s) "
                        f"after trying {', '.join(f'[{m}]' for m in seen)}: {_describe_error(exc)}",
                        retries=total_retries,
                        latency_s=round(total_latency, 3),
                        deadline_exceeded=True,
                    ) from exc
                detail = _describe_error(exc)
                errors.append(f"{candidate}: {detail}")
                details.append(detail)
            except Exception as exc:  # noqa: BLE001 - chain policy decides
                detail = _describe_error(exc)
                errors.append(f"{candidate}: {detail}")
                details.append(detail)
        if len(seen) == 1:
            raise InferenceError(
                f"vision model call failed ({next(iter(seen))}): {details[0]}",
                retries=total_retries,
                latency_s=round(total_latency, 3),
            )
        raise InferenceError(
            "vision model call failed; tried "
            + ", ".join(f"[{m}]" for m in seen)
            + ": "
            + " | ".join(errors),
            retries=total_retries,
            latency_s=round(total_latency, 3),
        )

    def _complete_one(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        deadline_at: float | None = None,
    ) -> Completion:
        provider = self.config.provider_for(model_id)
        transport = self._transport_for(provider)
        bare_model = self.config.strip_prefix(model_id)
        last_error: Exception | None = None
        t0 = time.monotonic()
        attempts = max(0, self.config.retries) + 1
        for attempt in range(attempts):
            remaining = _remaining(deadline_at)
            if remaining is not None and remaining <= 0:
                raise InferenceError(
                    "vision call deadline exceeded",
                    retries=attempt,
                    latency_s=round(time.monotonic() - t0, 3),
                    deadline_exceeded=True,
                )
            request_timeout = self.config.timeout
            if remaining is not None:
                request_timeout = max(0.05, min(request_timeout, remaining))
            try:
                result = transport.chat_completion(
                    messages,
                    bare_model,
                    max_tokens or self.config.max_tokens,
                    timeout=request_timeout,
                )
                return Completion(
                    text=result.text,
                    model=model_id,
                    provider=provider,
                    transport=type(transport).__name__,
                    retries=attempt,
                    latency_s=round(time.monotonic() - t0, 3),
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                )
            except Exception as exc:  # noqa: BLE001 - retry policy decides
                last_error = exc
                if not _is_retryable(exc) or attempt >= self.config.retries:
                    break
                delay = _retry_delay(exc, attempt, self.config.retry_backoff)
                remaining = _remaining(deadline_at)
                if remaining is not None and delay >= remaining:
                    break
                time.sleep(delay)
        # Bare detail: the chain wrapper builds the final message.
        raise InferenceError(
            _describe_error(last_error),
            retries=attempt,
            latency_s=round(time.monotonic() - t0, 3),
        )
