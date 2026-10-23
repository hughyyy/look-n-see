# look-n-see — Implementation Plan

## Goal
A stdio MCP server exposing a vision model via HuggingFace serverless
inference, so a vision-less main model can analyze screenshots and explain
images. See `BRIEF.md` for requirements.

## Stack
- Python 3.14 (only toolchain on this workstation)
- `mcp` (MCPServer, mcp 2.x) + `huggingface_hub` (InferenceClient)
- stdlib for image input parsing (urllib, base64, mimetypes) — no extra deps
- `pyproject.toml` (setuptools) with console-script entry point `look-n-see`
- Tests: stdlib `unittest` (zero dev deps)

## Layout
```
pyproject.toml       # metadata, console script, optional `ocr` extra
LICENSE              # MIT
.github/workflows/
  ci.yml             # unittest on push/PR, Python 3.10 + 3.14
  release.yml        # tag -> test -> build -> GitHub release -> PyPI OIDC
README.md            # user-facing: tools, install, config, dev
BRIEF.md             # requirements
IMPLEMENTATION.md    # this file: design, OMP integration, release plan
HANDOVER.md          # session handover: state, known issues, next steps
VISION_SESSION_COLLAPSE.md  # known issue: long-lived process degradation
research/
  ocr-options.md     # OCR engine + distribution-pattern research (2026-10)
src/look_n_see/
  __init__.py
  __main__.py        # python -m look_n_see
  config.py          # env parsing + defaults (incl. OCR roles/engine, deadline)
  images.py          # file path / data URI / URL -> (media_type, bytes) -> data URI
  client.py          # provider clients: retry/backoff, fallback chain, deadline
  openrouter.py      # OpenRouter chat-completions transport (httpx)
  ocr.py             # OCR dispatcher + vlm/rapidocr/mistral engines
  prompts.py         # default prompts per tool
  sanity.py          # output-shape classifier + degradation annotation
  server.py          # MCPServer app + role-gated registration + health + main()
tests/
  test_images.py     # input parsing, caps, URL policy, redirects, media types
  test_prompts.py
  test_openrouter.py
  test_config.py     # env parsing, provider routing, OCR roles, deadline
  test_client.py     # retries, fallback chain, deadline, cleanup, transports
  test_ocr.py        # mistral adapter, dispatcher, engine errors, lifecycle
  test_benchmark.py
  test_sanity.py     # refusal anchoring, severe flags, annotation
  test_scripts.py    # harness scripts: sampling, preflight, summaries
  test_server.py     # registration, schemas, structured errors, stdio handshakes
```

## Config (env vars)
| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | HF auth (needed for HF/`hf:` models; fallback route) |
| `OPENROUTER_API_KEY` | — | OpenRouter auth (needed for `openrouter:` models — the default route) |
| `VISION_MODEL` | `openrouter:google/gemma-4-26b-a4b-it` | default model id; `openrouter:` prefix routes to OpenRouter, `hf:`/bare ids to HF |
| `VISION_PROVIDER` | `auto` | `auto` (prefix decides), `hf`, or `openrouter` (bare ids) |
| `VISION_FALLBACKS` | `Qwen/Qwen3-VL-30B-A3B-Instruct openrouter:google/gemma-4-31b-it` | comma/space-separated model chain tried after `VISION_MODEL` on failure; unset = default chain, `""` = disabled |
| `HF_ENDPOINT` | — | dedicated HF endpoint URL; pins the HF transport target |
| `VISION_MAX_TOKENS` | `2048` | max generated tokens (raised from 1024: reasoning models starve at 1024) |
| `VISION_TIMEOUT` | `60` | per-request timeout (s) |
| `VISION_RETRIES` | `3` | retries on 429/5xx/connection |
| `VISION_RETRY_BACKOFF` | `2.0` | exponential backoff base (s) |
| `VISION_ENABLED` | `1` | `0` → don't register the vision tools (OCR-only variant) |
| `OCR_ENABLED` | `1` | `0` → don't register the OCR tool (vision-only variant) |
| `OCR_ENGINE` | `vlm` | default OCR engine: `vlm` \| `rapidocr` \| `mistral` |
| `MISTRAL_API_KEY` | — | auth for the `mistral` OCR engine |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | optional `HTTP-Referer` / `X-Title` headers |
| `VISION_DEADLINE` | `100` | wall-clock budget for one call across retries + fallbacks |
| `VISION_ALLOW_PRIVATE_URLS` | `0` | allow loopback/private image URLs (SSRF guard; `1` opts in) |

## Tools (all accept optional prompt to steer output, and optional model)
1. `describe_image(image, prompt=None, model=None)` — general description.
2. `analyze_screenshot(image, prompt=None, model=None)` — UI/state analysis.
3. `ask_about_image(image, question, model=None)` — Q&A over the image.
4. `extract_text(image, prompt=None, model=None, engine=None)` — OCR; returns
   `{"text", "engine", ...}` JSON. `engine` overrides `OCR_ENGINE`.
5. `health()` — JSON report of roles, routing, credential presence, deadline
   and URL policy; no network call, registered in every variant.

`image` accepts: file path, base64 data URI, or `http(s)://` URL (auto-detect).
`model` accepts any id in the same syntax as `VISION_MODEL`; omitted = default.
On failure the call walks the fallback chain (`VISION_FALLBACKS`).

## OCR design (implemented 2026-10)

- `ocr.py` exposes `OcrDispatcher.extract(uri, prompt, model, engine)` — the
  OCR counterpart of `client.complete`; engines are cached per process.
- `vlm` engine: `build_messages(uri, prompt, OCR_DEFAULT)` through the same
  `VisionClient` (so fallbacks, retries, and provider routing are shared).
- `rapidocr` engine: lazy import of `rapidocr-onnxruntime` + cv2/numpy — an
  `OcrError` with an install hint when the `ocr` extra is missing. Handles
  both the 1.2.x `(result, elapse)` tuple and the 1.4.x plain-list shapes.
- `mistral` engine: `httpx.Client` → `POST /v1/ocr` with Bearer auth,
  retry/backoff reuse (`_is_retryable`/`_retry_delay` from client), per-page
  markdown joined. Images are sent as the documented
  `{"type": "image_url", "image_url": <data URI>}` document; `document_url`
  is reserved for PDF/PPTX/DOCX. `MISTRAL_API_KEY` required; an `OcrError`
  otherwise.
- Packaging: `[project.optional-dependencies] ocr` pulls rapidocr with env
  markers (`>=1.4` on Python <3.13; `>=1.2` on >=3.13 — 1.4.x declares
  `requires_python <3.13`, verified against PyPI 2026-10-22).
- Variants: `VISION_ENABLED`/`OCR_ENABLED` gate registration at startup, so
  `tools/list` advertises exactly the enabled surface (MCP spec-sanctioned;
  same pattern as scrape-n-search's `ENABLED_TOOLS`). Dedicated
  `look-n-see-vision`/`look-n-see-ocr` console scripts are **not implemented**
  (TBD — only if a role must be unselectable at startup; they'd just set the
  env flag in a wrapper).
- Engine choice research + cloud/local alternatives: `research/ocr-options.md`.

## Benchmark suite (`scripts/benchmark.py` + `look_n_see.benchmark`)
- Run: models × images × reps, streaming JSONL, resumable; per-call metadata
  (provider, transport, latency, finish_reason, usage, retries).
- Summarize: per-model stats, output-shape flags (refusals, JSON-marker dumps,
  empty/terse — ported from the vision-collapse classifier), repetition rate,
  flagged samples; writes by-model views + `analysis_prompt.md` for an
  evaluation agent.
- Benchmark rows disable fallbacks so attribution is exact, record the image
  path (plus basename) and retry/latency metadata on failures, and tolerate
  corrupt/truncated JSONL rows on both resume and summarize. `resume=False`
  truncates an existing results file instead of appending duplicates.
  p90 uses nearest-rank; repetition excludes single-rep pairs.
- Sweep results live under `benchmarks/` (gitignored); verdict: default =
  `openrouter:google/gemma-4-26b-a4b-it` (same model as the old HF default,
  ~2x median / ~3x p90 latency improvement, ~$0.0002/call), with the default
  fallback chain Qwen3-VL-30B-A3B-Instruct (precision/second opinion) then
  openrouter gemma-4-31b (last resort). Notable: gpt-5-nano/nemotron need
  >1024 max_tokens (reasoning), so the budget default is 2048.

## Design seams (for deferred session state)
- `client.complete(messages)` takes OpenAI-style chat messages (text + image
  content parts) and returns text — a pure function of messages, so history/
  context injection later is additive.
- Image input is normalized to a data URI before building messages, so the
  message shape never depends on input kind.
- Session state (`VISION_SESSION_MODE`, per-session history/context,
  `clear_session`) is spec'd in `BRIEF.md`, NOT implemented now.

## Error handling
- Retry 408/425/429/5xx/529 and connection errors with jittered exponential
  backoff (`VISION_RETRIES`), honoring a numeric `Retry-After` up to 60 s.
- One `VISION_DEADLINE` budget spans the retry and fallback chain; each
  request timeout is `min(VISION_TIMEOUT, remaining)` (HF's client-level
  timeout is adjusted under a lock because `InferenceClient` has no per-call
  override).
- Tool failures are structured: `ToolError` with
  `{"error": {"type": input|inference|ocr|internal, "exception", "message"}}`,
  so MCP clients see `isError: true` instead of an ordinary text result.
- Invalid, unreadable, empty, or non-image inputs (file, data URI, URL) →
  descriptive tool error; URL fetches cap the read at `MAX_IMAGE_BYTES + 1`,
  close `HTTPError` bodies, reject non-`image/*` media types, restrict
  redirects to http(s), refuse private/loopback/link-local hosts unless
  `VISION_ALLOW_PRIVATE_URLS=1`, and redact userinfo/query strings in errors.
- Malformed provider responses (non-object JSON, missing `choices`/`pages`,
  non-string content parts) → readable error, not `AttributeError`/`IndexError`.
- Env edge values: empty model/base URL fall back to defaults; negative retries/
  backoff clamp to 0; non-positive max tokens/timeout/deadline and NaN/inf fall
  back; unknown `OCR_ENGINE`/`VISION_PROVIDER` log a warning and fall back.
- Empty/refusal/generation-payload responses get a warning trailer via
  `sanity.annotate` (collapse-incident mitigation; see `VISION_SESSION_COLLAPSE.md`).

## Oh My Pi (OMP) integration
The server is registered as a user-level MCP server so agents can call it from
any working directory.

- Config: `~/.omp/agent/mcp.json` →
  `look-n-see` (stdio, `.venv/bin/look-n-see`,
  `HF_TOKEN` resolved from OMP's own HuggingFace credential store via
  `!/home/<user>/.local/bin/omp token huggingface`,
  `OPENROUTER_API_KEY` via `!.../omp token openrouter`,
  `VISION_FALLBACKS=Qwen/Qwen3-VL-30B-A3B-Instruct openrouter:google/gemma-4-31b-it`,
  `timeout: 120000`).
- The `!` env indirection runs `omp token huggingface` (OMP's supported way to
  read the stored `huggingface` api_key in `~/.omp/agent/agent.db`) and injects
  the trimmed output as `HF_TOKEN` — no token duplication, no re-export, and it
  tracks the credential OMP itself uses.
- Why `timeout: 120000`: OMP's default MCP request timeout is 30s, too short for
  serverless inference on a vision model; 120s gives headroom.
- After editing, run `/mcp reload`, then `/mcp test look-n-see` and `/mcp list`
  in an OMP session. The four tools (`describe_image`, `analyze_screenshot`,
  `ask_about_image`, `extract_text`) then appear alongside built-in tools for
  the main agent and subagents; `VISION_ENABLED`/`OCR_ENABLED` in the server
  env trim the set per variant.
- To make it project-scoped instead of user-scoped, move the same block to
  `.omp/mcp.json` in this repo.

## Release & packaging plan (parity with scrape-n-search)

**Status: implemented in-repo (2026-10-23); external steps remain.** The repo
now has `LICENSE` (MIT), complete `pyproject.toml` metadata (`readme`,
`license = "MIT"` + `license-files`, `[project.urls]`, keywords/classifiers,
setuptools>=77 for PEP 639), `ci.yml`, and `release.yml`.

Facts checked:
- GitHub repo `hughyyy/look-n-see` exists; default branch `main`.
- `look-n-see` is available on PyPI (404 as of 2026-10); `look-n-see-mcp` is
  the fallback name.
- Tests are stdlib `unittest` with zero dev deps — CI runs `unittest`.
- `python -m build` produces `look_n_see-0.1.0.tar.gz` +
  `look_n_see-0.1.0-py3-none-any.whl`; wheel METADATA carries
  `License-Expression: MIT`, `License-File: LICENSE`, both project URLs and the
  markdown README.

Remaining steps:
1. **PyPI one-time admin**: register the trusted publisher for `look-n-see` →
   `hughyyy/look-n-see`, workflow `release.yml`, environment `release`.
2. **First release**: `git tag v0.1.0 && git push origin v0.1.0`; verify
   `uvx --from 'look-n-see[ocr]' look-n-see` on a clean machine.
3. **Optional parity (deferred)**: `install.sh`/`setup.sh` installer like
   scrape-n-search's.

Non-goals for packaging: no Docker image, no npm/JS packaging, no HTTP
transport — stdio only, per BRIEF.

## Verification
- Offline: `compileall`; `python -m unittest discover -s tests -q` (370 tests —
  image parsing incl. cap/empty/media/URL policy, config incl. OCR roles,
  deadline, model normalization and invalid env values, prompts, transports,
  retries/backoff/jitter/Retry-After, deadline budgeting, fallback chains,
  OCR engines with mocked httpx, dispatcher/transport lifecycle, benchmark
  resume/paths/stats, response sanity, harness scripts, server registration +
  role gating + tool schemas + structured errors, and a real stdio handshake
  per release variant).
- Branch coverage (`coverage run --branch --source=look_n_see`) is 100% for
  every module except the two retry-loop exits and the two `__main__` entry
  guards; total 99%. Server startup/registration makes no provider call.
- Live (manual): provider tokens in env (or `omp token <provider>`), then
  `scripts/live_smoke.py` (add `--json` for a machine-readable report/exit code)
  for vision and the `extract_text` tool for OCR (rapidocr engine = local).
  Verified 2026-10-22: default vision route, rapidocr + vlm engines, and
  role-gated registrations over stdio. Mistral remains live-untested (no key).
- A restart of the OMP client re-spawns the server from the editable install,
  so a running session picks up new tool registrations without config changes.
