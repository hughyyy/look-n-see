# AGENTS.md — look-n-see

## Project
MCP server exposing a vision model via OpenRouter and HuggingFace Inference.
See `README.md` (user-facing), `BRIEF.md` (requirements), `IMPLEMENTATION.md`
(design + release plan), `HANDOVER.md` (session handover, current state,
known issues).

## Conventions
- Python 3.10+. No Node/Cargo toolchains on this workstation.
- Server dependency-light: MCP SDK + `huggingface_hub` + `httpx` only; the
  local OCR engine (`rapidocr-onnxruntime`) is an optional `ocr` extra.
  Tests use stdlib `unittest` — zero dev deps.
- OCR routing: `OcrDispatcher` in `src/look_n_see/ocr.py` with three engines
  (vlm default, rapidocr local, mistral cloud). Don't add a new OCR engine
  without updating the dispatcher + `research/ocr-options.md` tradeoffs.
- Tool-surface variants: `VISION_ENABLED`/`OCR_ENABLED` gate registration at
  startup — the three release variants come from env, not separate packages.
- Secrets (HF token, OpenRouter key) come from environment, never hardcoded or
  committed.
- Config (model id, endpoint URLs, timeouts) via environment variables with
  sane defaults — see the `Config` dataclass in `src/look_n_see/config.py`.
- Every MCP tool must return structured output the calling model can act on
  (JSON where it matters, not prose-only). Failures raise `ToolError` with
  `{"error": {"type": "input|inference|ocr|internal", "exception", "message"}}`
  so the client sees `isError: true`; never return `"error: ..."` as a result.
- Image inputs are validated in `images.py`: 25 MiB cap, `image/*` only,
  http(s)-only redirects, private/loopback hosts refused unless
  `VISION_ALLOW_PRIVATE_URLS=1`, URL credentials/query strings redacted.
- Retries/fallbacks share one `VISION_DEADLINE` budget; per-request timeouts
  derive from the remaining budget. Keep new network calls inside that budget.
- Output-shape classification lives in `sanity.py` (shared by the server's
  degradation warning and the benchmark harness); do not duplicate patterns.

## Workflow
- Read `BRIEF.md` before starting work.
- Keep the brief and `IMPLEMENTATION.md` in sync with decisions made during
  implementation. README documents the user-facing surface (tools, env vars).
- One commit per logical change; conventional commit messages.
- Release/packaging setup follows the plan in `IMPLEMENTATION.md`
  ("Release & packaging plan") — parity with scrape-n-search. Don't invent a
  different mechanism.

## Verification
- `python -m compileall` / import smoke test after changes.
- `python -m unittest discover -s tests -q` — full offline suite (image input
  parsing/policy, config incl. OCR roles/deadline, prompts, transports,
  retries/fallbacks/deadline, OCR engines (mocked httpx), benchmark + harness
  scripts, sanity classification, server registration/structured errors and
  role gating).
- MCP server must start and register tools without a live HF call
  (tool registration must not require network); OCR engines other than `vlm`
  must degrade to clear errors when their extra/key is missing.
- Live calls are manual: `scripts/live_smoke.py` with the provider token in the
  environment (or via `omp token <provider>`); live OCR via the `extract_text`
  tool (`OCR_ENGINE=rapidocr` local, or default vlm).
- Known issue: long-lived server processes can degrade over many calls
  (canned refusals / non-analyses) — see `VISION_SESSION_COLLAPSE.md`; a fresh
  process recovers immediately.
