"""Environment-driven configuration with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"


@dataclass(frozen=True)
class Config:
    model: str
    endpoint: str | None
    max_tokens: int
    timeout: float
    retries: int
    retry_backoff: float

    @property
    def target(self) -> str:
        """The model id or dedicated endpoint URL passed to the client."""
        return self.endpoint or self.model


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def load_config() -> Config:
    return Config(
        model=os.environ.get("VISION_MODEL", DEFAULT_MODEL),
        endpoint=os.environ.get("HF_ENDPOINT") or None,
        max_tokens=_env_int("VISION_MAX_TOKENS", 1024),
        timeout=_env_float("VISION_TIMEOUT", 60.0),
        retries=_env_int("VISION_RETRIES", 3),
        retry_backoff=_env_float("VISION_RETRY_BACKOFF", 2.0),
    )
