# hf-vision-mcp — Project Brief

## Purpose
An MCP (Model Context Protocol) server that exposes a vision-capable model via
HuggingFace Inference. It gives models without native vision the ability to
"see": the main model asks this server to analyze screenshots and explain
images, and the results come back as tool results the main model can act on.

## Refined requirements (from concept Q&A)

### Access
- HuggingFace **serverless Inference API** via `huggingface_hub` `InferenceClient`,
  authenticated with `HF_TOKEN`.
- Optional `HF_ENDPOINT` override for a dedicated Inference Endpoint later.
- Simple retry with backoff on 429/5xx; clear errors otherwise.

### Transport
- **stdio** only (local MCP client spawns the server as a subprocess).

### Image input
- Accept all three, auto-detected:
  - file path on disk
  - base64 data URI (`data:image/...;base64,...`)
  - `http(s)://` URL

### Tools (specialized set, prompt-driven)
- `describe_image` — general natural-language description.
- `analyze_screenshot` — UI/state analysis of a screenshot.
- `ask_about_image` — Q&A over an image.
- Every tool accepts an optional `prompt`/`focus` parameter so the main model
  can steer toward targeted answers (features, values, specific regions).
- Output is **prompt-driven, loosely structured** (JSON where sensible, not a
  rigid schema). No separate OCR tool — prompt-driven Q&A covers text extraction.

### Model
- Configurable via env (`VISION_MODEL`), default `google/gemma-4-26B-A4B-it`
  (Gemma 4 26B A4B MoE, instruct). Apache-2.0, served on HF serverless
  inference today (verified against `router.huggingface.co/v1/models`).
- Note: `google/gemma-4-12B-it` (the originally chosen 12B Unified) is
  `any-to-any` and NOT on the serverless router — it needs a dedicated
  Inference Endpoint (`HF_ENDPOINT`). 26B A4B is the same Gemma 4 family,
  256K context, MoE (3.8B active → fast), MMMU-Pro 73.8%.
- Other serverless vision options via `VISION_MODEL`:
  `google/gemma-4-31B-it`, `Qwen/Qwen3-VL-30B-A3B-Instruct`,
  `Qwen/Qwen2.5-VL-72B-Instruct`.

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
- Offline smoke tests: server starts, tools register, input parsing works,
  mocked HF client. Live HF calls gated behind an env flag; user runs the
  live test themselves.

## Stack
- Python 3.14 (only available toolchain on this workstation)
- `mcp` (MCPServer, mcp 2.x), `huggingface_hub`
- pip + venv, `pyproject.toml`

## Non-goals (initial implementation)
- No local model hosting / weights management.
- No training or fine-tuning.
- No image generation.
- No HTTP/SSE transport, no OCR tool, no rigid output schemas.
- No session state / multi-turn context (specified above, deferred).
