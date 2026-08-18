# hf-vision-mcp — Implementation Plan

## Goal
A stdio MCP server exposing a vision model via HuggingFace serverless
inference, so a vision-less main model can analyze screenshots and explain
images. See `BRIEF.md` for requirements.

## Stack
- Python 3.14 (only toolchain on this workstation)
- `mcp` (MCPServer, mcp 2.x) + `huggingface_hub` (InferenceClient)
- stdlib for image input parsing (urllib, base64, mimetypes) — no extra deps
- `pyproject.toml` (setuptools) with console-script entry point `hf-vision-mcp`
- Tests: stdlib `unittest` (zero dev deps)

## Layout
```
pyproject.toml
src/hf_vision_mcp/
  __init__.py
  __main__.py        # python -m hf_vision_mcp
  config.py          # env parsing + defaults
  images.py          # file path / data URI / URL -> (media_type, bytes) -> data URI
  client.py          # InferenceClient wrapper: chat_completion, retry/backoff
  prompts.py         # default prompts per tool
  server.py          # FastMCP app + tool registrations + main()
tests/
  test_images.py
  test_prompts.py
  test_server.py     # registration smoke + mocked client
```

## Config (env vars)
| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | HF auth (required for inference) |
| `VISION_MODEL` | `google/gemma-4-26B-A4B-it` | model id for serverless inference |
| `HF_ENDPOINT` | — | dedicated endpoint URL; overrides `VISION_MODEL` target |
| `VISION_MAX_TOKENS` | `1024` | max generated tokens |
| `VISION_TIMEOUT` | `60` | per-request timeout (s) |
| `VISION_RETRIES` | `3` | retries on 429/5xx/connection |
| `VISION_RETRY_BACKOFF` | `2.0` | exponential backoff base (s) |

## Tools (all accept optional prompt to steer output)
1. `describe_image(image, prompt=None)` — general description.
2. `analyze_screenshot(image, prompt=None)` — UI/state analysis.
3. `ask_about_image(image, question)` — Q&A over the image.

`image` accepts: file path, base64 data URI, or `http(s)://` URL (auto-detect).

## Design seams (for deferred session state)
- `client.complete(messages)` takes OpenAI-style chat messages (text + image
  content parts) and returns text — a pure function of messages, so history/
  context injection later is additive.
- Image input is normalized to a data URI before building messages, so the
  message shape never depends on input kind.
- Session state (`VISION_SESSION_MODE`, per-session history/context,
  `clear_session`) is spec'd in `BRIEF.md`, NOT implemented now.

## Error handling
- Retry 429/5xx/connection errors with exponential backoff (`VISION_RETRIES`).
- Invalid image input → descriptive tool error.
- HF failures → readable message (model target + status), not a traceback.

## Oh My Pi (OMP) integration
The server is registered as a user-level MCP server so agents can call it from
any working directory.

- Config: `~/.omp/agent/mcp.json` →
  `hf-vision` (stdio, `.venv/bin/hf-vision-mcp`, `HF_TOKEN` copied from the
  launching shell via the `"HF_TOKEN": "HF_TOKEN"` env pattern, `timeout: 120000`).
- Why `timeout: 120000`: OMP's default MCP request timeout is 30s, too short for
  serverless inference on a vision model; 120s gives headroom.
- After editing, run `/mcp reload`, then `/mcp test hf-vision` and `/mcp list`
  in an OMP session. The three tools (`describe_image`, `analyze_screenshot`,
  `ask_about_image`) then appear alongside built-in tools for the main agent
  and subagents.
- To make it project-scoped instead of user-scoped, move the same block to
  `.omp/mcp.json` in this repo.

## Verification
- Offline (this session): `compileall`, server starts and registers 3 tools,
  image parsing unit tests (file/data-uri/URL), prompts unit tests, mocked
  client end-to-end tool test.
- Live (user, later): `HF_TOKEN` set + `scripts/live_smoke.py` — one real
  inference call. Not run by me; user does this.
