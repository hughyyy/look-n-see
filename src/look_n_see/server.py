"""MCPServer exposing the vision and OCR tools."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from . import config as config_mod
from .client import InferenceError, VisionClient
from .images import ImageInputError, to_data_uri
from .ocr import OcrDispatcher, OcrError
from .prompts import ASK_DEFAULT, DESCRIBE_DEFAULT, SCREENSHOT_DEFAULT, build_messages
from .sanity import annotate


def _tool_error(exc: Exception, kind: str) -> ToolError:
    """Structured failure: MCP marks the result `isError` and keeps the shape."""
    payload = json.dumps(
        {
            "error": {
                "type": kind,
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        },
        ensure_ascii=False,
    )
    return ToolError(payload)


def _image_options(cfg: config_mod.Config) -> dict:
    """Per-call image input limits: one URL fetch must fit the call budget."""
    return {
        "timeout": min(cfg.timeout, cfg.deadline),
        "allow_private_urls": cfg.allow_private_urls,
    }


def _make_lifespan(ocr: OcrDispatcher, client: VisionClient, owns_client: bool):
    """Close engine/transport HTTP clients when the server shuts down."""

    @asynccontextmanager
    async def lifespan(_server: MCPServer):
        try:
            yield {}
        finally:
            ocr.close()
            if owns_client:
                client.close()

    return lifespan


def create_server(
    cfg: config_mod.Config | None = None,
    client: VisionClient | None = None,
) -> MCPServer:
    cfg = cfg or config_mod.load_config()
    owns_client = client is None
    client = client or VisionClient(cfg)
    ocr = OcrDispatcher(cfg, vision_client=client)

    mcp = MCPServer(
        name="look-n-see",
        version=__version__,
        lifespan=_make_lifespan(ocr, client, owns_client),
    )

    # Role-gated registration: each release variant enables a subset of the
    # surface here — both on = default, VISION_ENABLED=0 = OCR-only,
    # OCR_ENABLED=0 = vision-only. tools/list advertises exactly what is
    # registered, which is what MCP clients display. `health` is diagnostic
    # and present in every variant.
    if cfg.vision_enabled:
        _register_vision_tools(mcp, client, cfg)
    if cfg.ocr_enabled:
        _register_ocr_tool(mcp, ocr, cfg)
    _register_health_tool(mcp, cfg)
    return mcp


def _register_vision_tools(
    mcp: MCPServer, client: VisionClient, cfg: config_mod.Config
) -> None:
    options = _image_options(cfg)

    @mcp.tool()
    def describe_image(
        image: str, prompt: str | None = None, model: str | None = None
    ) -> str:
        """Describe an image in natural language.

        Args:
            image: File path, base64 data URI, or http(s) URL of the image.
            prompt: Optional instruction steering the description.
            model: Optional model id; `openrouter:<slug>` routes to OpenRouter,
                bare ids follow VISION_PROVIDER. Falls back through
                VISION_FALLBACKS on failure.
        """
        return _run(client, image, prompt, DESCRIBE_DEFAULT, model, options)

    @mcp.tool()
    def analyze_screenshot(
        image: str, prompt: str | None = None, model: str | None = None
    ) -> str:
        """Analyze a screenshot as a UI/state capture for an agent.

        Args:
            image: File path, base64 data URI, or http(s) URL of the screenshot.
            prompt: Optional instruction steering the analysis.
            model: Optional model id; `openrouter:<slug>` routes to OpenRouter,
                bare ids follow VISION_PROVIDER. Falls back through
                VISION_FALLBACKS on failure.
        """
        return _run(client, image, prompt, SCREENSHOT_DEFAULT, model, options)

    @mcp.tool()
    def ask_about_image(
        image: str, question: str, model: str | None = None
    ) -> str:
        """Ask a question about an image and get an answer.

        Args:
            image: File path, base64 data URI, or http(s) URL of the image.
            question: The question to answer about the image.
            model: Optional model id; `openrouter:<slug>` routes to OpenRouter,
                bare ids follow VISION_PROVIDER. Falls back through
                VISION_FALLBACKS on failure.
        """
        return _run(client, image, question, ASK_DEFAULT, model, options)


def _register_ocr_tool(
    mcp: MCPServer, ocr: OcrDispatcher, cfg: config_mod.Config
) -> None:
    options = _image_options(cfg)

    @mcp.tool()
    def extract_text(
        image: str,
        prompt: str | None = None,
        model: str | None = None,
        engine: str | None = None,
    ) -> str:
        """Extract all visible text from an image (OCR).

        Args:
            image: File path, base64 data URI, or http(s) URL of the image.
            prompt: Optional instruction steering extraction (vlm engine).
            model: Optional model id for the vlm engine (same syntax as the
                vision tools); the mistral engine treats it as the OCR model id.
            engine: Override the OCR engine for this call: `vlm` (default),
                `rapidocr` (local, needs the `ocr` extra), or `mistral`
                (needs MISTRAL_API_KEY).
        """
        try:
            uri = to_data_uri(image, **options)
        except ImageInputError as exc:
            raise _tool_error(exc, "input") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as an MCP tool error
            raise _tool_error(exc, "internal") from exc
        try:
            return ocr.extract(uri, prompt=prompt, model=model, engine=engine)
        except OcrError as exc:
            raise _tool_error(exc, "ocr") from exc
        except InferenceError as exc:
            raise _tool_error(exc, "inference") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as an MCP tool error
            raise _tool_error(exc, "internal") from exc


def _register_health_tool(mcp: MCPServer, cfg: config_mod.Config) -> None:
    @mcp.tool()
    def health() -> str:
        """Report tool roles and non-secret configuration (no network call)."""
        return json.dumps(
            {
                "ok": True,
                "version": __version__,
                "vision_enabled": cfg.vision_enabled,
                "ocr_enabled": cfg.ocr_enabled,
                "ocr_engine": cfg.ocr_engine,
                "model": cfg.model,
                "provider": cfg.provider,
                "fallbacks": list(cfg.fallbacks),
                "endpoint_configured": bool(cfg.endpoint),
                "deadline_s": cfg.deadline,
                "timeout_s": cfg.timeout,
                "allow_private_urls": cfg.allow_private_urls,
                "credentials": {
                    "openrouter": bool(cfg.openrouter_api_key),
                    "hf": bool(os.environ.get("HF_TOKEN")),
                    "mistral": bool(cfg.mistral_api_key),
                },
            },
            ensure_ascii=False,
        )


def _run(
    client: VisionClient,
    image: str,
    prompt: str | None,
    default: str,
    model: str | None = None,
    options: dict | None = None,
) -> str:
    try:
        uri = to_data_uri(image, **(options or {}))
    except ImageInputError as exc:
        raise _tool_error(exc, "input") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as an MCP tool error
        raise _tool_error(exc, "internal") from exc
    messages = build_messages(uri, prompt, default)
    try:
        text = client.complete(messages, model=model)
    except Exception as exc:  # noqa: BLE001 - surfaced as an MCP tool error
        raise _tool_error(exc, "inference") from exc
    return annotate(text)


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
