# look-n-see — Project Brief

## Purpose
An MCP (Model Context Protocol) server that exposes vision-capable models
via OpenRouter and HuggingFace Inference, plus an OCR tool with three
engines. It gives models without native vision the ability to "see" and
"read": the main model asks this server to analyze screenshots, explain
images, and extract text, and the results come back as tool results the
main model can act on.

## Refined requirements (from concept Q&A)

### Access
- **OpenRouter** (default): OpenAI-compatible chat completions via httpx,
  authenticated with `OPENROUTER_API_KEY` (from env; OMP credential store).
- **HuggingFace serverless Inference API** via `huggingface_hub`
  `InferenceClient`, authenticated with `HF_TOKEN`; used for `hf:`/bare ids.
- Optional `HF_ENDPOINT` override for a dedicated HF Inference Endpoint
  (pins the HF transport only).
- Model ids are provider-routed: `openrouter:<slug>` → OpenRouter,
  `hf:<id>`/bare → HF; `VISION_PROVIDER=openrouter` flips bare ids.
- Simple retry with backoff on 429/5xx; fallback chain `VISION_FALLBACKS`
  on hard failure; clear errors otherwise.

### Transport
- **stdio** only (local MCP client spawns the server as a subprocess).

### Model
- Configurable via env (`VISION_MODEL`), default
  `openrouter:google/gemma-4-26b-a4b-it` (Gemma 4 26B A4B MoE via OpenRouter)
  — chosen by benchmark sweep (2026-10-22): identical quality to the HF
  serverless route at ~2x lower median latency / ~3x lower p90, ~$0.0002/call.
- Fallback chain (default, `VISION_FALLBACKS`; set "" to disable):
  1. `Qwen/Qwen3-VL-30B-A3B-Instruct` (HF) — precision/second opinion
  2. `openrouter:google/gemma-4-31b-it` — fast last resort
  An explicit `model=` tool argument is how a caller requests a second
  opinion without waiting for a failure.
- Every tool accepts an optional `model` id to override per call.
- Notes from the sweep: Qwen/Qwen3-VL-30B-A3B-Instruct (HF) had the best raw
  accuracy (correct calendar weekday derivation) but slower latency;
  reasoning-heavy models (gpt-5-nano, nemotron) starve at 1024 max_tokens —
  default budget raised to 2048.

### Image input
- Accept all three, auto-detected:
  - file path on disk
  - base64 data URI (`data:image/...;base64,...`)
  - `http(s)://` URL

### Tools (specialized set, prompt-driven)
- `describe_image` — general natural-language description.
- `analyze_screenshot` — UI/state analysis of a screenshot.
- `ask_about_image` — Q&A over an image.
- `extract_text` — OCR: all visible text as `{"text", "engine", ...}` JSON.
  - Every tool accepts an optional `prompt`/`focus` parameter so the main model
    can steer toward targeted answers (features, values, specific regions), and
    an optional `model` parameter for per-call provider/model selection.
  - `extract_text` additionally accepts `engine` (per-call override of
    `OCR_ENGINE`).

### OCR engines (researched 2026-10 — see `research/ocr-options.md`)
- `vlm` (default): image + OCR prompt through the existing vision client;
  zero extra deps, no new secrets.
- `rapidocr`: local ONNX engine (PP-OCR models) — offline, no torch;
  requires the optional `ocr` extra (`rapidocr-onnxruntime`; note 1.4.x is
  Python <3.13, older Pythons get the 1.2 line via env markers).
- `mistral`: dedicated cloud OCR (httpx → `api.mistral.ai/v1/ocr`),
  document-grade accuracy, per-page markdown + bounding boxes, ≈$4/1k pages;
  requires `MISTRAL_API_KEY`.
- Output is **prompt-driven, loosely structured** (JSON where sensible, not a
  rigid schema); `extract_text` returns a JSON envelope with the extracted
  text plus engine-specific metadata.

### Release variants (one package, role-gated registration)
- Both (default): register vision + OCR tools (`VISION_ENABLED=1`,
  `OCR_ENABLED=1`).
- Vision only: `OCR_ENABLED=0`.
- OCR only: `VISION_ENABLED=0`.
- `tools/list` advertises exactly the enabled surface. Dedicated
  `look-n-see-vision`/`look-n-see-ocr` console scripts are a possible later
  addition (tracked in `IMPLEMENTATION.md`, not implemented).

### Benchmarking (implemented)
- `scripts/benchmark.py`: models × images × reps (default 5) with resumable
  JSONL recording (provider, latency, usage, finish_reason, flags), per-model
  summary with output-shape classification, and a generated agent prompt for
  a strengths/weaknesses assessment.
- Run data lives in `benchmarks/` (gitignored). Sweep verdict 2026-10-22:
  keep gemma-4-26B family (A4B) as default via OpenRouter; 4-31B as runner-up
  (fastest, +25% cost, no quality gain); Qwen3-VL-30B most accurate but
  slower; gpt-5-nano / nemotron unusable at 1024 max_tokens.

### Future: session state (NOT in initial implementation)
- Long-term goal: repeated calls share context — the server keeps a
  multi-turn conversation with the vision model so later calls can reference
  earlier images/answers.
- Planned shape (design seam only now), **configurable — default multi-turn**:
  - Optional `session_id` on every tool; omitted = stateless (current behavior).
  - Two modes (env `VISION_SESSION_MODE`):
    - `history` (default): full multi-turn chat history with the vision model —
      later calls reference earlier images/answers.
    - `context`: fetch/accumulate reference — inject a rolling context block
      into each call instead of full raw history.
    - Configurable per deployment; default `history`.
  - Server keeps in-memory state per session (one MCP client per stdio
    process, so per-process dict is natural); appends the turn and sends
    history/context within the window.
  - A `clear_session`/`reset_session` tool; context-budget trimming when
    approaching the window limit.
  - Build the "prompt → model call" path as a pure function of
    (messages, image) in the initial implementation so history/context injection
    is a later, additive change. `InferenceClient.chat_completion` already
    supports multi-turn image content.

### Deployment
- Local now (stdio subprocess, runnable entry point), service-ready later:
  `pyproject.toml` with a console-script entry point; all config via env vars.

### Verification
- Offline unit tests (stdlib `unittest`): server starts, tools register, input
  parsing works, mocked clients. Live check is manual:
  `scripts/live_smoke.py` with the provider token in the environment.
- Known issue (2026-10-21): a long-lived server process can degrade over
  ~tens of calls (canned refusals / non-analyses) — see
  `VISION_SESSION_COLLAPSE.md`; a fresh process answers correctly.

## Stack
- Python 3.14 (only available toolchain on this workstation)
- `mcp` (MCPServer, mcp 2.x), `huggingface_hub`, `httpx`
- Optional `ocr` extra: `rapidocr-onnxruntime` (env-markered — see
  `IMPLEMENTATION.md` "OCR design")
- pip + venv, `pyproject.toml`

## Non-goals (initial implementation)
- No local model hosting / weights management (the local OCR engine is
  prebuilt ONNX models via pip, not weight management).
- No training or fine-tuning.
- No image generation.
- No HTTP/SSE transport, no rigid output schemas, no PDF parsing (Mistral
  accepts PDFs directly, but the tool surface stays image-first).
- No session state / multi-turn context (specified above, deferred).
