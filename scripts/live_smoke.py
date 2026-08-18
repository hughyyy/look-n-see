#!/usr/bin/env python3
"""Live smoke test: one real inference call against HuggingFace.

Requires HF_TOKEN in the environment. Run with the venv python:

    HF_TOKEN=hf__... .venv/bin/python scripts/live_smoke.py [image] [question]

`image` defaults to a small sample image; `question` defaults to a generic
description request. Prints the model's response to stdout.
"""

import os
import sys

from hf_vision_mcp.client import VisionClient
from hf_vision_mcp.config import load_config
from hf_vision_mcp.images import to_data_uri
from hf_vision_mcp.prompts import ASK_DEFAULT, build_messages

SAMPLE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        print("error: HF_TOKEN is not set", file=sys.stderr)
        return 2

    image = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_URI
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."

    cfg = load_config()
    client = VisionClient(cfg)
    uri = to_data_uri(image)
    messages = build_messages(uri, question, ASK_DEFAULT)
    print(f"target: {cfg.target}", file=sys.stderr)
    print(client.complete(messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
