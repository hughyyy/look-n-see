# look-n-see

A lightweight [MCP](https://modelcontextprotocol.io) server that gives agents without
native vision the ability to **see** — and to **read**: three vision tools
(**describe_image, analyze_screenshot, ask_about_image**) run a vision-capable
model on an image and return text the calling model can act on, and an OCR tool
(**extract_text**) pulls text out of images via three engines (VLM, local
RapidOCR, or Mistral OCR).

Images are accepted three ways, auto-detected: a file path on disk, a base64
`data:` URI, or an `http(s)://` URL. The default vision route is
`openrouter:google/gemma-4-26b-a4b-it` via OpenRouter (chosen by a benchmark sweep
for identical quality at ~2x lower median latency and ~$0.0002/call vs the HF
serverless route), with an automatic fallback chain to
`Qwen/Qwen3-VL-30B-A3B-Instruct` then `openrouter:google/gemma-4-31b-it`.

## Release variants

One package, one release — the tool surface is selected at startup by two env
flags (`tools/list` advertises exactly what is registered):

| Variant | Install | Env |
| --- | --- | --- |
| **Both (default)** | `pip install 'look-n-see[ocr]'` (or `uvx --from 'look-n-see[ocr]' look-n-see`) | *(defaults)* |
| Vision only | `pip install look-n-see` | `OCR_ENABLED=0` |
| OCR only | `pip install 'look-n-see[ocr]'` | `VISION_ENABLED=0` |

The `ocr` extra pulls the local RapidOCR engine (ONNX, no torch). Dedicated
`look-n-see-vision` / `look-n-see-ocr` console scripts remain a possible future
addition (see `IMPLEMENTATION.md`).

## Tools

Every tool returns structured output the calling model can act on (JSON where it
matters) and accepts an optional `prompt` to steer the answer. All config is
environment-driven (see below).

| Tool | Description |
| --- | --- |
| `describe_image` | `image` (required), `prompt`?, `model`? — general natural-language description. |
| `analyze_screenshot` | `image` (required), `prompt`?, `model`? — UI/state analysis of a screenshot for an agent. |
| `ask_about_image` | `image`, `question` (required), `model`? — Q&A over the image. |
| `extract_text` | `image` (required), `prompt`?, `model`?, `engine`? — OCR. Returns `{"text", "engine", ...}`. |

Model ids are provider-routed: `openrouter:<slug>` → OpenRouter,
`hf:<id>`/bare → HuggingFace Inference (`VISION_PROVIDER=openrouter` flips bare
ids). On failure the call walks the fallback chain; retries with backoff cover
429/5xx/connection errors.

### OCR engines

| Engine | How | Requires |
| --- | --- | --- |
| `vlm` (default) | image + OCR prompt through the existing vision client; zero extra deps | the default model's provider key |
| `rapidocr` | local ONNX (PP-OCR models), offline, boxes + confidence | `pip install 'look-n-see[ocr]'` |
| `mistral` | dedicated cloud OCR (`api.mistral.ai/v1/ocr`), per-page markdown + bboxes | `MISTRAL_API_KEY` (≈$4/1k pages) |

Select per call with `engine=` or globally with `OCR_ENGINE`.

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
| `VISION_ENABLED` | `1` | set `0` to disable the vision tools (OCR-only variant) |
| `OCR_ENABLED` | `1` | set `0` to disable the OCR tool (vision-only variant) |
| `OCR_ENGINE` | `vlm` | default OCR engine: `vlm` \| `rapidocr` \| `mistral` |
| `MISTRAL_API_KEY` | — | auth for the `mistral` OCR engine |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | optional `HTTP-Referer` / `X-Title` headers |

## How it works

The server normalizes every image input to a data URI, builds one chat message
(image + prompt) per tool call, and hands it to `VisionClient` — a pure function of
the message list, which is the seam for future session-state work. The client
routes per model id (`openrouter:` vs `hf:`/bare), retries with exponential
backoff, and walks the fallback chain on hard failure. The OCR tool routes through
an `OcrDispatcher` with the same image normalization: the `vlm` engine reuses the
vision client, `rapidocr` decodes locally via ONNX, and `mistral` posts the data
URI to Mistral's OCR API. Stateless by design: no session history is kept between
calls, and `VISION_ENABLED`/`OCR_ENABLED` decide at startup which tools
`tools/list` advertises.

## Development

```bash
.venv/bin/pip install -e '.[ocr]'                    # incl. local OCR engine
.venv/bin/python -m unittest discover -s tests -q   # offline tests: parsing, routing,
                                                    # retries, prompts, engines, server behavior
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
