"""Environment-driven configuration with sane defaults."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

DEFAULT_MODEL = "openrouter:google/gemma-4-26b-a4b-it"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_PREFIX = "openrouter:"
HF_PREFIX = "hf:"
# Default chain when VISION_FALLBACKS is unset: precision second opinion,
# then a larger/faster last resort. Set VISION_FALLBACKS="" to disable.
DEFAULT_FALLBACKS = (
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "openrouter:google/gemma-4-31b-it",
)


@dataclass(frozen=True)
class Config:
    model: str
    endpoint: str | None
    max_tokens: int
    timeout: float
    retries: int
    retry_backoff: float
    provider: str = "auto"
    fallbacks: tuple[str, ...] = ()
    openrouter_api_key: str | None = None
    openrouter_base_url: str = DEFAULT_OPENROUTER_URL
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None
    # Tool-surface roles: the three release variants are expressed as
    # (VISION_ENABLED, OCR_ENABLED) — both on = default, one off = focused.
    vision_enabled: bool = True
    ocr_enabled: bool = True
    ocr_engine: str = "vlm"  # vlm | rapidocr | mistral
    mistral_api_key: str | None = None

    @property
    def target(self) -> str:
        """The default model id or dedicated endpoint URL passed to the client."""
        return self.endpoint or self.model

    def provider_for(self, model_id: str) -> str:
        """Transport provider for a model id.

        An explicit `openrouter:`/`hf:` prefix always wins; otherwise the
        `provider` setting decides the interpretation of bare ids.
        """
        if model_id.startswith(OPENROUTER_PREFIX):
            return "openrouter"
        if model_id.startswith(HF_PREFIX):
            return "hf"
        if self.provider == "openrouter":
            return "openrouter"
        return "hf"

    @staticmethod
    def strip_prefix(model_id: str) -> str:
        if model_id.startswith(OPENROUTER_PREFIX):
            return model_id[len(OPENROUTER_PREFIX):]
        if model_id.startswith(HF_PREFIX):
            return model_id[len(HF_PREFIX):]
        return model_id


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    # NaN/inf pass a plain float() parse but are never valid timeouts.
    return value if math.isfinite(value) else default


def _env_positive_float(name: str, default: float) -> float:
    value = _env_float(name, default)
    return value if value > 0 else default


def _env_text(name: str) -> str | None:
    """Env value with surrounding whitespace removed; blank counts as unset."""
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_fallbacks(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_FALLBACKS
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return tuple(parts)


def _parse_engine(raw: str | None) -> str:
    engine = (raw or "vlm").strip().lower()
    return engine if engine in {"vlm", "rapidocr", "mistral"} else "vlm"


def load_config() -> Config:
    provider = os.environ.get("VISION_PROVIDER", "auto").strip().lower()
    if provider not in {"auto", "hf", "openrouter"}:
        provider = "auto"
    return Config(
        model=os.environ.get("VISION_MODEL", "").strip() or DEFAULT_MODEL,
        endpoint=_env_text("HF_ENDPOINT"),
        max_tokens=_env_positive_int("VISION_MAX_TOKENS", 2048),
        timeout=_env_positive_float("VISION_TIMEOUT", 60.0),
        retries=max(0, _env_int("VISION_RETRIES", 3)),
        retry_backoff=max(0.0, _env_float("VISION_RETRY_BACKOFF", 2.0)),
        provider=provider,
        fallbacks=_parse_fallbacks(os.environ.get("VISION_FALLBACKS")),
        openrouter_api_key=_env_text("OPENROUTER_API_KEY"),
        openrouter_base_url=_env_text("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_URL,
        openrouter_site_url=_env_text("OPENROUTER_SITE_URL"),
        openrouter_app_name=_env_text("OPENROUTER_APP_NAME"),
        vision_enabled=_env_bool("VISION_ENABLED", True),
        ocr_enabled=_env_bool("OCR_ENABLED", True),
        ocr_engine=_parse_engine(os.environ.get("OCR_ENGINE")),
        mistral_api_key=_env_text("MISTRAL_API_KEY"),
    )
