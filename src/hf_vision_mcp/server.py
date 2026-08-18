"""FastMCP server exposing the vision tools."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import config as config_mod
from .client import VisionClient
from .images import ImageInputError, to_data_uri
from .prompts import ASK_DEFAULT, DESCRIBE_DEFAULT, SCREENSHOT_DEFAULT, build_messages


def create_server(
    cfg: config_mod.Config | None = None,
    client: VisionClient | None = None,
) -> MCPServer:
    cfg = cfg or config_mod.load_config()
    client = client or VisionClient(cfg)
    mcp = MCPServer(name="hf-vision-mcp")

    @mcp.tool()
    def describe_image(image: str, prompt: str | None = None) -> str:
        """Describe an image in natural language.

        Args:
            image: File path, base64 data URI, or http(s) URL of the image.
            prompt: Optional instruction steering the description.
        """
        return _run(client, image, prompt, DESCRIBE_DEFAULT)

    @mcp.tool()
    def analyze_screenshot(image: str, prompt: str | None = None) -> str:
        """Analyze a screenshot as a UI/state capture for an agent.

        Args:
            image: File path, base64 data URI, or http(s) URL of the screenshot.
            prompt: Optional instruction steering the analysis.
        """
        return _run(client, image, prompt, SCREENSHOT_DEFAULT)

    @mcp.tool()
    def ask_about_image(image: str, question: str) -> str:
        """Ask a question about an image and get an answer.

        Args:
            image: File path, base64 data URI, or http(s) URL of the image.
            question: The question to answer about the image.
        """
        return _run(client, image, question, ASK_DEFAULT)

    return mcp


def _run(
    client: VisionClient,
    image: str,
    prompt: str | None,
    default: str,
) -> str:
    try:
        uri = to_data_uri(image)
    except ImageInputError as exc:
        return f"error: {exc}"
    messages = build_messages(uri, prompt, default)
    try:
        return client.complete(messages)
    except Exception as exc:  # noqa: BLE001 - surfaced as a tool result
        return f"error: {exc}"


def main() -> None:
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
