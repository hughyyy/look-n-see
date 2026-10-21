#!/usr/bin/env python3
"""Live smoke test: one real inference call.

Requires the provider token in the environment: OPENROUTER_API_KEY for
openrouter: model ids (the default), HF_TOKEN for HF/hf: ids. Run with the
venv python:

    OPENROUTER_API_KEY=sk-or-... .venv/bin/python scripts/live_smoke.py [image] [question]

`image` defaults to a small sample image; `question` defaults to a generic
description request. Prints the model's response to stdout.
"""

import os
import sys

from look_n_see.client import VisionClient
from look_n_see.config import load_config
from look_n_see.images import to_data_uri
from look_n_see.prompts import ASK_DEFAULT, build_messages

SAMPLE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def main() -> int:
    cfg = load_config()
    needs_openrouter = cfg.provider_for(cfg.model) == "openrouter"
    token_var = "OPENROUTER_API_KEY" if needs_openrouter else "HF_TOKEN"
    if not os.environ.get(token_var):
        print(f"error: {token_var} is not set for model {cfg.model}", file=sys.stderr)
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
