"""Environment-driven configuration with sane defaults."""

from __future__ import annotations

import logging
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

_log = logging.getLogger(__name__)


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
    # Wall-clock budget for one tool call across retries and fallbacks. The
    # per-request timeout is derived from the remaining budget, so a call can
    # never outlive its MCP client's request timeout by much.
    deadline: float = 100.0
    # URL image inputs default to public hosts only (SSRF guard); opt out for
    # local/dev servers that fetch loopback or LAN URLs.
    allow_private_urls: bool = False

    @property
    def target(self) -> str:
        """The default model id or dedicated endpoint URL passed to the client."""
        return self.endpoint or self.model

    @staticmethod
    def normalize_model_id(model_id: str) -> str:
        """Canonical model id: trimmed, with a case-insensitive provider prefix.

        Bare ids stay byte-identical (HF/OpenRouter slugs are case-sensitive);
        only the prefix spelling and surrounding whitespace are normalized.
        """
        value = model_id.strip()
        lowered = value.lower()
        for prefix in (OPENROUTER_PREFIX, HF_PREFIX):
            if lowered.startswith(prefix):
                return prefix + value[len(prefix):].strip()
        return value

    def provider_for(self, model_id: str) -> str:
        """Transport provider for a model id.

        An explicit `openrouter:`/`hf:` prefix always wins; otherwise the
        `provider` setting decides the interpretation of bare ids.
        """
        normalized = self.normalize_model_id(model_id)
        if normalized.startswith(OPENROUTER_PREFIX):
            return "openrouter"
        if normalized.startswith(HF_PREFIX):
            return "hf"
        if self.provider == "openrouter":
            return "openrouter"
        return "hf"

    @staticmethod
    def strip_prefix(model_id: str) -> str:
        normalized = Config.normalize_model_id(model_id)
        if normalized.startswith(OPENROUTER_PREFIX):
            return normalized[len(OPENROUTER_PREFIX):]
        if normalized.startswith(HF_PREFIX):
            return normalized[len(HF_PREFIX):]
        return normalized


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
    if engine not in {"vlm", "rapidocr", "mistral"}:
        _log.warning(
            "OCR_ENGINE=%r is not one of vlm/rapidocr/mistral; falling back to 'vlm'",
            raw,
        )
        return "vlm"
    return engine


def _parse_provider(raw: str | None) -> str:
    provider = (raw or "auto").strip().lower()
    if provider not in {"auto", "hf", "openrouter"}:
        _log.warning(
            "VISION_PROVIDER=%r is not auto/hf/openrouter; falling back to 'auto'",
            raw,
        )
        return "auto"
    return provider


def load_config() -> Config:
    return Config(
        model=os.environ.get("VISION_MODEL", "").strip() or DEFAULT_MODEL,
        endpoint=_env_text("HF_ENDPOINT"),
        max_tokens=_env_positive_int("VISION_MAX_TOKENS", 2048),
        timeout=_env_positive_float("VISION_TIMEOUT", 60.0),
        retries=max(0, _env_int("VISION_RETRIES", 3)),
        retry_backoff=max(0.0, _env_float("VISION_RETRY_BACKOFF", 2.0)),
        provider=_parse_provider(os.environ.get("VISION_PROVIDER")),
        fallbacks=_parse_fallbacks(os.environ.get("VISION_FALLBACKS")),
        openrouter_api_key=_env_text("OPENROUTER_API_KEY"),
        openrouter_base_url=_env_text("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_URL,
        openrouter_site_url=_env_text("OPENROUTER_SITE_URL"),
        openrouter_app_name=_env_text("OPENROUTER_APP_NAME"),
        vision_enabled=_env_bool("VISION_ENABLED", True),
        ocr_enabled=_env_bool("OCR_ENABLED", True),
        ocr_engine=_parse_engine(os.environ.get("OCR_ENGINE")),
        mistral_api_key=_env_text("MISTRAL_API_KEY"),
        deadline=_env_positive_float("VISION_DEADLINE", 100.0),
        allow_private_urls=_env_bool("VISION_ALLOW_PRIVATE_URLS", False),
    )
