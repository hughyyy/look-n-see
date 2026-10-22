"""OCR engines: VLM-as-OCR (default), local RapidOCR (optional extra), Mistral OCR.

The dispatcher normalizes every image to a data URI (shared with the vision
path) and routes to one engine:

- `vlm` (default): text extraction through the existing chat-completions
  client — zero new dependencies, works everywhere.
- `rapidocr`: fully-pip local ONNX engine (no torch). Requires the optional
  `ocr` extra: `pip install 'look-n-see[ocr]'`.
- `mistral`: dedicated cloud OCR via plain httpx (`MISTRAL_API_KEY`),
  per-page markdown + bounding boxes, document-grade accuracy.

Every engine returns a JSON object (text + engine + engine-specific
metadata) so the calling model can act on it.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .client import VisionClient, _describe_error, _is_retryable
from .config import Config
from .images import load_image
from .prompts import OCR_DEFAULT, build_messages

DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MISTRAL_MODEL = "mistral-ocr-latest"
OCR_ENGINES = ("vlm", "rapidocr", "mistral")


class OcrError(RuntimeError):
    """Raised when an OCR engine cannot produce a result."""


def _data_uri_to_bytes(uri: str) -> bytes:
    _, data = load_image(uri)
    return data


class VlmOcr:
    """VLM-as-OCR: send the image + an OCR prompt through the shared client."""

    name = "vlm"

    def __init__(self, client: VisionClient):
        self._client = client

    def extract(
        self, uri: str, *, prompt: str | None = None, model: str | None = None
    ) -> dict[str, Any]:
        messages = build_messages(uri, prompt, OCR_DEFAULT)
        return {
            "text": self._client.complete(messages, model=model),
            "engine": self.name,
        }


try:  # The rapidocr extra is optional; degrade gracefully when absent.
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    _RAPIDOCR_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in CI without the extra
    RapidOCR = None  # type: ignore[assignment]
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _RAPIDOCR_AVAILABLE = False


class RapidOcrEngine:
    """Local ONNX OCR (PP-OCR models); boxes + per-item confidence."""

    name = "rapidocr"

    def __init__(self) -> None:
        if not _RAPIDOCR_AVAILABLE:
            raise OcrError(
                "rapidocr engine requires the 'ocr' extra: "
                "pip install 'look-n-see[ocr]' (or set OCR_ENGINE=vlm/mistral)"
            )
        self._engine = RapidOCR()

    def extract(
        self, uri: str, *, prompt: str | None = None, model: str | None = None
    ) -> dict[str, Any]:
        data = _data_uri_to_bytes(uri)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)  # type: ignore[union-attr]
        if image is None:
            raise OcrError("image could not be decoded for local OCR")
        raw = self._engine(image)
        if isinstance(raw, tuple):
            raw = raw[0]  # 1.2.x returns (result, elapse)
        items = [
            {"text": text, "confidence": float(score), "box": box}
            for box, text, score in (raw or [])
        ]
        return {
            "text": "\n".join(item["text"] for item in items),
            "engine": self.name,
            "items": items,
        }


class MistralOcr:
    """Dedicated cloud OCR via Mistral's OpenAI-adjacent REST API (httpx-only)."""

    name = "mistral"

    def __init__(self, config: Config):
        self.config = config
        if not config.mistral_api_key:
            raise OcrError("MISTRAL_API_KEY is required for the mistral OCR engine")
        self._http = httpx.Client(
            base_url=DEFAULT_MISTRAL_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.mistral_api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    def extract(
        self, uri: str, *, prompt: str | None = None, model: str | None = None
    ) -> dict[str, Any]:
        payload = {
            "model": model or DEFAULT_MISTRAL_MODEL,
            "document": {"type": "document_url", "document_url": uri},
        }
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self._http.post("/ocr", json=payload)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"Mistral OCR returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                data = response.json()
                pages = data.get("pages") or []
                text = "\n\n".join(page.get("markdown", "") for page in pages)
                return {
                    "text": text,
                    "engine": self.name,
                    "model": data.get("model", model or DEFAULT_MISTRAL_MODEL),
                    "pages": len(pages),
                }
            except Exception as exc:  # noqa: BLE001 - retry policy decides
                last_error = exc
                if not _is_retryable(exc) or attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_backoff * (2**attempt))
        raise OcrError(_describe_error(last_error))


class OcrDispatcher:
    """Routes an extraction request to the configured or per-call engine."""

    def __init__(self, config: Config, vision_client: VisionClient | None = None):
        self.config = config
        self._vision = vision_client or VisionClient(config)
        self._engines: dict[str, Any] = {}

    def _engine(self, name: str | None) -> Any:
        engine_name = (name or self.config.ocr_engine).strip().lower()
        if engine_name not in OCR_ENGINES:
            raise OcrError(
                f"unknown OCR engine: {engine_name!r} (choose from {', '.join(OCR_ENGINES)})"
            )
        cached = self._engines.get(engine_name)
        if cached is None:
            if engine_name == "vlm":
                cached = VlmOcr(self._vision)
            elif engine_name == "rapidocr":
                cached = RapidOcrEngine()
            else:
                cached = MistralOcr(self.config)
            self._engines[engine_name] = cached
        return cached

    def extract(
        self,
        uri: str,
        *,
        prompt: str | None = None,
        model: str | None = None,
        engine: str | None = None,
    ) -> str:
        """Extract text from a data URI and return a JSON result string."""
        result = self._engine(engine).extract(uri, prompt=prompt, model=model)
        return json.dumps(result, ensure_ascii=False)
