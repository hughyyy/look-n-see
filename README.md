# look-n-see

A lightweight [MCP](https://modelcontextprotocol.io) server that gives agents without
native vision the ability to **see**: three tools — **describe_image,
analyze_screenshot, ask_about_image** — run a vision-capable model on an image and
return text the calling model can act on.

Images are accepted three ways, auto-detected: a file path on disk, a base64
`data:` URI, or an `http(s)://` URL. The default route is
`openrouter:google/gemma-4-26b-a4b-it` via OpenRouter (chosen by a benchmark sweep
for identical quality at ~2x lower median latency and ~$0.0002/call vs the HF
serverless route), with an automatic fallback chain to
`Qwen/Qwen3-VL-30B-A3B-Instruct` then `openrouter:google/gemma-4-31b-it`.

## Tools

Every tool returns free-form text (JSON where the model deems it sensible) and
accepts an optional `prompt` to steer the answer. All config is environment-driven
(see below).

| Tool | Description |
| --- | --- |
| `describe_image` | `image` (required), `prompt`?, `model`? — general natural-language description. |
| `analyze_screenshot` | `image` (required), `prompt`?, `model`? — UI/state analysis of a screenshot for an agent. |
| `ask_about_image` | `image`, `question` (required), `model`? — Q&A over the image. |

Model ids are provider-routed: `openrouter:<slug>` → OpenRouter,
`hf:<id>`/bare → HuggingFace Inference (`VISION_PROVIDER=openrouter` flips bare
ids). On failure the call walks the fallback chain; retries with backoff cover
429/5xx/connection errors.

## Requirements

- Python 3.10+
- An API key for at least one provider: `OPENROUTER_API_KEY` (default route),
  `HF_TOKEN` (fallback route)

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .

# run as a stdio MCP server
.venv/bin/look-n-see
```

### Register with Oh My Pi

Add to `~/.omp/agent/mcp.json` (adjust paths to your checkout; the `!` env
indirection resolves tokens from OMP's credential store):

```json
{
  "mcpServers": {
    "look-n-see": {
      "type": "stdio",
      "command": "/absolute/path/to/look-n-see/.venv/bin/look-n-see",
      "env": {
        "HF_TOKEN": "!/home/<user>/.local/bin/omp token huggingface",
        "OPENROUTER_API_KEY": "!/home/<user>/.local/bin/omp token openrouter"
      },
      "timeout": 120000
    }
  }
}
```

`timeout: 120000` is important — OMP's default MCP request timeout (30s) is too
short for serverless vision inference. For a client with plain env secrets, drop
the `!` lines and export `HF_TOKEN`/`OPENROUTER_API_KEY` before launching.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | OpenRouter auth (needed for `openrouter:` models — the default route) |
| `HF_TOKEN` | — | HF auth (needed for `hf:`/bare ids — the fallback route) |
| `VISION_MODEL` | `openrouter:google/gemma-4-26b-a4b-it` | default model id; `openrouter:` prefix routes to OpenRouter, `hf:`/bare ids to HF |
| `VISION_PROVIDER` | `auto` | `auto` (prefix decides), `hf`, or `openrouter` (bare ids) |
| `VISION_FALLBACKS` | `Qwen/Qwen3-VL-30B-A3B-Instruct openrouter:google/gemma-4-31b-it` | comma/space-separated chain tried after `VISION_MODEL` on failure; `""` disables |
| `HF_ENDPOINT` | — | dedicated HF endpoint URL; pins the HF transport target |
| `VISION_MAX_TOKENS` | `2048` | max generated tokens (1024 starves reasoning models) |
| `VISION_TIMEOUT` | `60` | per-request timeout (s) |
| `VISION_RETRIES` | `3` | retries on 429/5xx/connection errors |
| `VISION_RETRY_BACKOFF` | `2.0` | exponential backoff base (s) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | optional `HTTP-Referer` / `X-Title` headers |

## How it works

The server normalizes every image input to a data URI, builds one chat message
(image + prompt) per tool call, and hands it to `VisionClient` — a pure function of
the message list, which is the seam for future session-state work. The client
routes per model id (`openrouter:` vs `hf:`/bare), retries with exponential
backoff, and walks the fallback chain on hard failure. Stateless by design: no
session history is kept between calls.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -q   # 56 offline tests: parsing, routing,
                                                    # retries, prompts, server behavior (mocked client)
HF_TOKEN=... OPENROUTER_API_KEY=... .venv/bin/python scripts/live_smoke.py [image] [question]
```

Tool registration must not require a live HF call; `tests/test_server.py` covers
the end-to-end path with a mocked client, `scripts/live_smoke.py` does one real
inference call for manual verification. See `BRIEF.md` (requirements),
`IMPLEMENTATION.md` (design, OMP integration, release plan), and
`VISION_SESSION_COLLAPSE.md` (known issue: long-lived server processes can
degrade — restart the process to recover).

## Releases & packaging

**Status: planned, not yet set up** — intended to mirror scrape-n-search's setup
(CI + tag-triggered release to GitHub and PyPI via trusted publishing). The
concrete plan is in `IMPLEMENTATION.md` ("Release & packaging plan"); nothing is
published or configured yet.
