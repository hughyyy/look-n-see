"""Default prompts and chat-message assembly for the vision tools."""

from __future__ import annotations

from typing import Any

DESCRIBE_DEFAULT = (
    "Describe this image in detail. Cover the main subject, layout, visible "
    "text, and anything notable. Be concise but complete."
)

SCREENSHOT_DEFAULT = (
    "Analyze this screenshot as a UI/state capture for an agent. Report: "
    "1) what application or screen is shown, 2) key visible UI elements and "
    "their state (buttons, inputs, dialogs, errors), 3) any text or values "
    "that matter, 4) notable state or changes. Be precise and structured."
)

ASK_DEFAULT = "Answer the question about this image."


def build_messages(image_data_uri: str, prompt: str | None, default: str) -> list[dict[str, Any]]:
    """Build a single-turn chat message embedding the image plus the prompt.

    The message shape is a pure function of (image, prompt) so that later
    session-state work can inject history/context around it additively.
    """
    text = (prompt or "").strip() or default
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_data_uri}},
        {"type": "text", "text": text},
    ]
    return [{"role": "user", "content": content}]
