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
- Configurable via env (`VISION_MODEL`), default `Qwen2.5-VL-7B-Instruct`
  (widely available on serverless inference, strong vision/UI understanding).

### Deployment
- Local now (stdio subprocess, runnable entry point), service-ready later:
  `pyproject.toml` with a console-script entry point; all config via env vars.

### Verification
- Offline smoke tests: server starts, tools register, input parsing works,
  mocked HF client. Live HF calls gated behind an env flag; user runs the
  live test themselves.

## Stack
- Python 3.14 (only available toolchain on this workstation)
- `mcp` (FastMCP), `huggingface_hub`
- pip + venv, `pyproject.toml`

## Non-goals
- No local model hosting / weights management.
- No training or fine-tuning.
- No image generation.
- No HTTP/SSE transport, no OCR tool, no rigid output schemas.
