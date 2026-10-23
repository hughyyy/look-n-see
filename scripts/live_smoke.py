#!/usr/bin/env python3
"""Live smoke test: one real inference call.

Requires the provider token in the environment: OPENROUTER_API_KEY for
openrouter: model ids (the default), HF_TOKEN for HF/hf: ids. Run with the
venv python:

    OPENROUTER_API_KEY=sk-or-... .venv/bin/python scripts/live_smoke.py [image] [question]

`image` defaults to a small sample image; `question` defaults to a generic
description request. Prints the model's response to stdout, or a
machine-readable report with `--json` (exit code 1 when the call fails or the
response is flagged as degraded), suitable for CI-able checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from look_n_see.client import VisionClient
from look_n_see.config import load_config
from look_n_see.images import to_data_uri
from look_n_see.prompts import ASK_DEFAULT, build_messages
from look_n_see.sanity import classify

SAMPLE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=SAMPLE_URI)
    parser.add_argument("question", nargs="?", default="Describe this image.")
    parser.add_argument(
        "--json", action="store_true", help="emit a machine-readable report"
    )
    args = parser.parse_args()

    cfg = load_config()
    needs_openrouter = cfg.provider_for(cfg.model) == "openrouter"
    token_var = "OPENROUTER_API_KEY" if needs_openrouter else "HF_TOKEN"
    if not os.environ.get(token_var):
        if args.json:
            print(json.dumps({"ok": False, "error": f"{token_var} is not set"}))
        else:
            print(f"error: {token_var} is not set for model {cfg.model}", file=sys.stderr)
        return 2

    client = VisionClient(cfg)
    try:
        uri = to_data_uri(
            args.image,
            timeout=min(cfg.timeout, cfg.deadline),
            allow_private_urls=cfg.allow_private_urls,
        )
        messages = build_messages(uri, args.question, ASK_DEFAULT)
        completion = client.complete_with_meta(messages)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        if args.json:
            print(json.dumps({"ok": False, "target": cfg.target, "error": str(exc)}))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    flags = classify(completion.text)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "target": cfg.target,
                    "model": completion.model,
                    "provider": completion.provider,
                    "latency_s": completion.latency_s,
                    "retries": completion.retries,
                    "finish_reason": completion.finish_reason,
                    "flags": flags,
                    "text": completion.text,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(f"target: {cfg.target}", file=sys.stderr)
        print(completion.text)
    return 0 if "refusal" not in flags and "empty" not in flags else 1


if __name__ == "__main__":
    sys.exit(main())
