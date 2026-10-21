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
| `HF_TOKEN` | — | HF auth (needed for HF/`hf:` models; fallback route) |
| `OPENROUTER_API_KEY` | — | OpenRouter auth (needed for `openrouter:` models — the default route) |
| `VISION_MODEL` | `openrouter:google/gemma-4-26b-a4b-it` | default model id; `openrouter:` prefix routes to OpenRouter, `hf:`/bare ids to HF |
| `VISION_PROVIDER` | `auto` | `auto` (prefix decides), `hf`, or `openrouter` (bare ids) |
| `VISION_FALLBACKS` | — | comma/space-separated model chain tried after `VISION_MODEL` on failure |
| `HF_ENDPOINT` | — | dedicated HF endpoint URL; pins the HF transport target |
| `VISION_MAX_TOKENS` | `2048` | max generated tokens (raised from 1024: reasoning models starve at 1024) |
| `VISION_TIMEOUT` | `60` | per-request timeout (s) |
| `VISION_RETRIES` | `3` | retries on 429/5xx/connection |
| `VISION_RETRY_BACKOFF` | `2.0` | exponential backoff base (s) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | optional `HTTP-Referer` / `X-Title` headers |

## Tools (all accept optional prompt to steer output, and optional model)
1. `describe_image(image, prompt=None, model=None)` — general description.
2. `analyze_screenshot(image, prompt=None, model=None)` — UI/state analysis.
3. `ask_about_image(image, question, model=None)` — Q&A over the image.

`image` accepts: file path, base64 data URI, or `http(s)://` URL (auto-detect).
`model` accepts any id in the same syntax as `VISION_MODEL`; omitted = default.
On failure the call walks the fallback chain (`VISION_FALLBACKS`).

## Benchmark suite (`scripts/benchmark.py` + `hf_vision_mcp.benchmark`)
- Run: models × images × reps, streaming JSONL, resumable; per-call metadata
  (provider, transport, latency, finish_reason, usage, retries).
- Summarize: per-model stats, output-shape flags (refusals, JSON-marker dumps,
  empty/terse — ported from the vision-collapse classifier), repetition rate,
  flagged samples; writes by-model views + `analysis_prompt.md` for an
  evaluation agent.
- Benchmark rows disable fallbacks so attribution is exact.
- Sweep results live under `benchmarks/` (gitignored); verdict: default =
  `openrouter:google/gemma-4-26b-a4b-it` (same model as the old HF default,
  ~2x median / ~3x p90 latency improvement, ~$0.0002/call), HF serverless
  gemma kept as `VISION_FALLBACKS`. Notable: Qwen3-VL-30B-A3B-Instruct had the
  best raw accuracy; gpt-5-nano/nemotron need >1024 max_tokens (reasoning).

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
  `hf-vision` (stdio, `.venv/bin/hf-vision-mcp`,
  `HF_TOKEN` resolved from OMP's own HuggingFace credential store via
  `!/home/bernie/.local/bin/omp token huggingface`,
  `OPENROUTER_API_KEY` via `!.../omp token openrouter`,
  `VISION_FALLBACKS=google/gemma-4-26B-A4B-it`, `timeout: 120000`).
- The `!` env indirection runs `omp token huggingface` (OMP's supported way to
  read the stored `huggingface` api_key in `~/.omp/agent/agent.db`) and injects
  the trimmed output as `HF_TOKEN` — no token duplication, no re-export, and it
  tracks the credential OMP itself uses.
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
