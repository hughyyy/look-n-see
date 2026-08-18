"""HuggingFace Inference client wrapper with retry/backoff."""

from __future__ import annotations

import os
import time
from typing import Any

from huggingface_hub import InferenceClient

from .config import Config

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class InferenceError(RuntimeError):
    """Raised when the model call ultimately fails."""


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "response", None)
    if status is not None:
        code = getattr(status, "status_code", None)
        if code is not None:
            return code in RETRYABLE_STATUS
    # Connection/timeout errors from the underlying HTTP layer.
    return isinstance(exc, (ConnectionError, TimeoutError)) or exc.__class__.__name__ in {
        "ConnectionError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
    }


class VisionClient:
    def __init__(self, config: Config, client: InferenceClient | None = None):
        self.config = config
        self._client = client or InferenceClient(
            model=config.target,
            token=os.environ.get("HF_TOKEN"),
            timeout=config.timeout,
        )

    def complete(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        """Run chat completion and return the assistant text.

        `messages` is OpenAI-style chat content (text + image parts), making
        this a pure function of the message list — the seam for future
        session-state history/context injection.
        """
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    model=self.config.target,
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as exc:  # noqa: BLE001 - retry policy decides
                last_error = exc
                if not _is_retryable(exc) or attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_backoff * (2**attempt))
        raise InferenceError(
            f"vision model call failed ({self.config.target}): {last_error}"
        ) from last_error
