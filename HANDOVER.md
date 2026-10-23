# HANDOVER — look-n-see

Handover for the next agent session. Project: **look-n-see** — an MCP server
that gives agents without native vision the ability to **see** (describe images,
analyze screenshots, answer image questions) and **read** (OCR), via OpenRouter
and HuggingFace Inference plus optional local/cloud OCR engines.

## What this project is

- **Problem solved**: a vision-less main model asks this stdio MCP server to
  analyze screenshots / explain images / extract text; results come back as
  tool results it can act on.
- **Four tools**: `describe_image`, `analyze_screenshot`, `ask_about_image`
  (vision, via `VisionClient` — OpenRouter default
  `openrouter:google/gemma-4-26b-a4b-it`, HF/`hf:` fallback chain), and
  `extract_text` (OCR, three engines: `vlm` default, `rapidocr` local,
  `mistral` cloud).
- **Three release variants from one package** — `VISION_ENABLED`/`OCR_ENABLED`
  env flags gate tool registration at startup (`tools/list` advertises exactly
  the enabled surface): both (default), vision-only (`OCR_ENABLED=0`),
  OCR-only (`VISION_ENABLED=0`).

## Current state (verified 2026-10-23)

- **370 tests pass** (`python -m unittest discover -s tests -q`, zero dev deps).
  Offline only; 99% branch coverage (only the two retry-loop exits and the two
  `__main__` guards are uncovered).
- Registered with Oh My Pi: `~/.omp/agent/mcp.json` → `look-n-see` (stdio,
  `/mnt/shared/Source/look-n-see/.venv/bin/look-n-see`, editable install of the
  repo, env: `HF_TOKEN`/`OPENROUTER_API_KEY` via `!omp token ...`,
  `VISION_FALLBACKS` override, timeout 120000). **Restart OMP to pick up the
  current source — no config change needed.**
- Live-verified: vision tools on the default OpenRouter route; `extract_text`
  with the rapidocr engine (local, ~3s) and the vlm engine (OpenRouter);
  role-gated registrations over stdio handshakes. Mistral OCR remains
  key-less/untested live.
- GitHub: pushed to `github.com/hughyyy/look-n-see` (public, branch `main`).
  Packaging is now implemented in-repo (MIT LICENSE, full metadata, CI +
  release workflows); nothing published to PyPI yet.
- Git history through the OCR release: initial server → OpenRouter provider +
  fallbacks → benchmark harness → gemma-4-26b default → rename → collapse doc →
  security hardening → OCR research/implementation. The latest review pass
  (2026-10-23) is the commit carrying the fixes listed below.

## Review pass (2026-10-23)

- **Mistral OCR payload fixed** — images are sent as
  `{"type": "image_url", "image_url": <data URI>}`, matching the API docs;
  the previous `document_url` form (and its test) was wrong for images.
- **Per-call deadline** (`VISION_DEADLINE`, default 100 s) now spans retries and
  fallbacks; per-request timeouts derive from the remaining budget, and URL
  fetches use `min(VISION_TIMEOUT, VISION_DEADLINE)` instead of a hardcoded 30 s.
- **Structured tool errors** — failures raise `ToolError` with a JSON
  `{"error": {"type", "exception", "message"}}` body and `isError: true`,
  replacing the old `"error: ..."` prose results.
- **URL hardening** — non-`image/*` media types rejected; private/loopback/
  link-local hosts refused unless `VISION_ALLOW_PRIVATE_URLS=1`; query strings
  redacted alongside userinfo in echoed URLs; redirects are host-checked too.
- **Model-id normalization** — prefixes are matched case-insensitively and
  trimmed for per-call ids, not just env config.
- **Retry policy** — 408/425/529 added, numeric `Retry-After` honored, equal
  jitter added; failures now carry retry/latency metadata into benchmark rows.
- **Benchmark harness** — summarize tolerates corrupt rows, `resume=False`
  truncates instead of doubling, rows/metadata store image paths, p90 is
  nearest-rank, repetition ignores single-rep pairs, per-model slugs
  disambiguate collisions, classifier refusal patterns are sentence-anchored.
- **Lifecycle** — transports/engines close on server shutdown; engine and
  transport construction locks; HF content normalization matches OpenRouter.
- **Diagnostics** — new `health` tool (no network) plus a warning trailer on
  empty/refusal/generation-payload responses; `scripts/live_smoke.py --json`.
- **Tests** — hermetic config tests (ambient `VISION_*`/`OCR_*` no longer leak
  in), script tests, sanity tests, plus coverage for every fix above.


## Since the previous handover (initial session → 2026-10-22)

- **Edge-case hardening + coverage expansion (2026-10-23)** — 164 → 294 tests.
  Image input: exact-cap boundary, empty payloads rejected for file/data
  URI/URL, case-insensitive `data:`/`;BASE64` forms, unreadable files and
  mid-stream read failures wrapped as `ImageInputError`, HTTPError body
  tempfiles closed (no ResourceWarning), redirect loops and
  `file:`/`ftp:`/`data:` redirect targets refused, credential redaction on
  fetch and oversize errors. Config: empty model/site/app/base URL and
  whitespace-only keys treated as unset, credentials stripped, negative
  retries/backoff clamped, non-positive max tokens/timeout and NaN/inf fall
  back; unmapped provider strings route bare ids to HF. Providers:
  non-object JSON bodies and non-string content parts raise `OpenRouterError`;
  missing `finish_reason`/`usage` handled; HF responses without choices and
  non-dict usage handled; `_describe_error` tolerates non-dict error bodies;
  `VISION_PROVIDER=openrouter` routing and HF-endpoint/OpenRouter-fallback
  isolation tested. OCR: Mistral non-object bodies/non-list or non-dict pages
  rejected, connection errors retried, exponential backoff tested; dispatcher
  forwards prompt/model, propagates engine errors and does not cache failed
  engines; the real local ONNX engine runs on a blank PNG when the `ocr` extra
  is installed (skipped otherwise). Benchmark: resume tolerates partial rows;
  symlinked output files refused; metadata/summary fields (latency, finish
  reasons, retries, repetition) and the flagged-sample cap asserted.
  Server: stdio handshake covers all four env combinations (including no
  tools), lists *and invokes* a tool, and asserts MCP argument validation
  (`ToolError` for missing/wrong-typed args) and the error-result contract.
- **OCR** — `extract_text` tool + `OcrDispatcher` (`src/look_n_see/ocr.py`):
  `vlm` (reuses VisionClient; zero deps), `rapidocr` (local ONNX, `ocr` extra;
  handles both the 1.2.x `(result, elapse)` tuple and 1.4.x list shapes),
  `mistral` (httpx → `api.mistral.ai/v1/ocr`, `MISTRAL_API_KEY`). Research and
  tradeoffs: `research/ocr-options.md`.
- **Variant gating** — `VISION_ENABLED`/`OCR_ENABLED`/`OCR_ENGINE` (
  `vlm|rapidocr|mistral`) + `MISTRAL_API_KEY` config; env-markered
  `[project.optional-dependencies] ocr`
  (`rapidocr>=1.4` on Python <3.13; `>=1.2` on >=3.13).
- **Security hardening** (from review) — 25 MiB image input cap; http(s)-only
  redirects (stdlib allows ftp); URL-userinfo redaction in errors;
  O_NOFOLLOW/O_EXCL benchmark writes; `mcp>=2` floor; OpenRouter response
  type validation; PII redaction of machine paths in docs.
- **Docs** — README (variants table, OCR engines, config), BRIEF, IMPLEMENTATION
  (OCR design + release plan), AGENTS, this handover.
- **Model/mode comparison runs** (2026-10-22) — `scripts/compare_modes.py`
  (vision vs vlm-ocr vs rapidocr over a sampled set) and
  `scripts/compare_models.py` (multi-model × vision/vlm-ocr, reference-free
  consensus metrics, resumable JSONL). Two live runs over `/tmp/omp-sshots*`:
  15-image three-mode sweep; 5 text-dense images × 5 models × 2 modes
  (55/55 calls ok). Headline findings (detail in gitignored
  `benchmarks/compare-models-*/summary.md`): gemma-4-31b-it == gemma-4-26b
  OCR accuracy at 1.6–3× lower latency; qwen3-vl-30b fabricated ~15% of its
  OCR tokens on one dense page (pixel-verified); gpt-4o-mini diverges by
  reading different real text rather than fabricating; every VLM-OCR model
  beat rapidocr on prose at equal latency.

## How to run

```bash
cd /mnt/shared/Source/look-n-see
.venv/bin/python -m unittest discover -s tests -q   # full offline suite
.venv/bin/look-n-see                                 # stdio MCP server

# live smoke (tokens via OMP store or env):
HF_TOKEN=$(omp token huggingface) OPENROUTER_API_KEY=$(omp token openrouter) \
  .venv/bin/python scripts/live_smoke.py [image] [question]
# live OCR through the server (any engine):
OCR_ENGINE=rapidocr .venv/bin/look-n-see   # spawn, then call extract_text
```

Keys come from env only (`HF_TOKEN`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`);
the OMP registration resolves the first two via `omp token <provider>`.

## Key files

- `src/look_n_see/server.py` — MCPServer + role-gated tool registration
- `src/look_n_see/client.py` — provider clients, retry/backoff, fallback chain
- `src/look_n_see/openrouter.py` — OpenRouter transport (httpx)
- `src/look_n_see/ocr.py` — OCR dispatcher + vlm/rapidocr/mistral engines
- `src/look_n_see/images.py` — file/data-URI/URL → data URI (bounded, http(s)-only redirects)
- `src/look_n_see/config.py` — env parsing + defaults (incl. OCR roles/engine)
- `src/look_n_see/sanity.py` — output-shape classifier + degradation annotation
- `src/look_n_see/benchmark.py` + `scripts/benchmark.py` — benchmark harness
- `scripts/compare_modes.py`, `scripts/compare_models.py` — vision/OCR
  three-mode and multi-model comparison harnesses (resumable JSONL +
  consensus/local-ref metrics)
- `research/ocr-options.md` — OCR engine + distribution-pattern research

## Known issues / notes

- **Vision session collapse** (2026-10-21, `VISION_SESSION_COLLAPSE.md`): a
  long-lived server process can degrade over ~tens of calls (canned refusals /
  non-analyses). Fresh process recovers immediately — restart OMP (or the
  stdio process) to recover. Mitigations now shipped: the `health` tool and a
  warning trailer on empty/refusal/generation-payload responses; automatic
  client recreation/restart is still not implemented.
- **`rapidocr-onnxruntime` 1.4.x requires Python <3.13** (PyPI metadata) —
  Python 3.13/3.14 get 1.2.3 via env markers. 1.2.x returns
  `(result, elapse)`; handled in `ocr.py`.
- **`MISTRAL_API_KEY` is not registered in mcp.json** — `extract_text` with
  `engine=mistral` errors until the key is added. The payload now matches the
  documented `image_url` form, but the engine is still live-untested.
- **OMP default MCP timeout is 30s** — registration uses 120s; keep it. The
  server-side `VISION_DEADLINE` (100s) bounds a call across retries/fallbacks
  so the client timeout is no longer exceeded by design.
- `min(VISION_TIMEOUT, VISION_DEADLINE)` bounds URL fetches; the HF transport
  adjusts its client-level timeout under a lock because `InferenceClient` has
  no per-call timeout argument.
- Benchmarks/ outputs (`benchmarks/`) are gitignored, local-only; the data
  contains model outputs over private-project screenshots — don't commit.
- The MCP client "about"/description: set via `gh repo edit`; README + about
  must stay in sync.

## Open items / next steps

- **Release**: packaging is implemented in-repo (LICENSE, metadata, CI +
  release workflows; see `IMPLEMENTATION.md`). Remaining external steps:
  register the PyPI trusted publisher (one-time manual), then
  `git tag v0.1.0 && git push origin v0.1.0`, and verify
  `uvx --from 'look-n-see[ocr]' look-n-see` on a clean machine.
- **Optional B-scripts** (TBD): `look-n-see-vision`/`look-n-see-ocr` console
  scripts that preset the env flags — only if a role must be hard-excluded at
  startup.
- **Session state** — spec'd in `BRIEF.md` ("Future: session state"), not
  implemented; `client.complete` is the seam.
- **Vision-collapse follow-ups** — `health` + response flagging are done;
  still open: raw-response logging and automatic client recreation/restart
  after N non-conforming responses.
- **Model-chain reorder (recommended, pending wider validation)** — the
  multi-model comparison measured gemma-4-31b-it at accuracy equal to the
  26b default with 1.6–3× lower latency (vision + vlm-ocr). Candidates after
  a 15–20 image confirmation run: `VISION_MODEL=openrouter:google/gemma-4-31b-it`
  with 26b as first fallback. qwen3-vl-30b not recommended for the chain
  (fabrication); gpt-4o-mini not recommended for OCR (coverage).
- **Extended live testing (partially done)** — vlm/rapidocr exercised live;
  mistral still untested key-wise. Remaining: drive the mounted tools through
  the registered server after an OMP restart, and optionally add
  `MISTRAL_API_KEY` to mcp.json.

The collapse-incident reproduction and the benchmark sweep are documented
outputs of previous sessions — keep the JSONL/data local (gitignored), and
treat vendor-reported benchmark numbers as dated claims, not facts.
