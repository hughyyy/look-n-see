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

## Current state (all verified 2026-10-22)

- **86 tests pass** (`python -m unittest discover -s tests -q`, zero dev deps).
- Registered with Oh My Pi: `~/.omp/agent/mcp.json` → `look-n-see` (stdio,
  `/mnt/shared/Source/look-n-see/.venv/bin/look-n-see`, editable install of the
  repo, env: `HF_TOKEN`/`OPENROUTER_API_KEY` via `!omp token ...`,
  `VISION_FALLBACKS` override, timeout 120000). **Restart OMP to pick up the
  current source — no config change needed.**
- Live-verified: vision tools on the default OpenRouter route; `extract_text`
  with the rapidocr engine (local, ~3s) and the vlm engine (OpenRouter);
  role-gated registrations over stdio handshakes.
- GitHub: pushed to `github.com/hughyyy/look-n-see` (public, branch `main`,
  descriptive about field set). Nothing published to PyPI yet.
- Git history (current HEAD `2f8b862`): `54222ff` (initial server) →
  `888c164`/`597ca3a`/`9f2e6f1` (OpenRouter provider + fallbacks + error
  handling) → `8334112` (benchmark harness) → `71fd588`/`f3edca2`
  (gemma-4-26b default + Qwen/gemma-4-31b fallback chain + HF routing fix) →
  `90647c4` (rename to look-n-see) → `605e4a0` (vision-collapse incident doc) →
  `c3b9ac9` (README + brief/implementation/agents sync) → `a153151` (security
  hardening: input caps, redirect restriction, O_NOFOLLOW writes, mcp>=2) →
  `36de452` (OCR research memo) → `2f8b862` (OCR implementation).

## Since the previous handover (initial session → 2026-10-22)

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
- `src/look_n_see/benchmark.py` + `scripts/benchmark.py` — benchmark harness
- `scripts/compare_modes.py`, `scripts/compare_models.py` — vision/OCR
  three-mode and multi-model comparison harnesses (resumable JSONL +
  consensus/local-ref metrics)
- `research/ocr-options.md` — OCR engine + distribution-pattern research
- `README.md` (user-facing), `BRIEF.md` (requirements), `IMPLEMENTATION.md` (design + release plan)

## Known issues / notes

- **Vision session collapse** (2026-10-21, `VISION_SESSION_COLLAPSE.md`): a
  long-lived server process can degrade over ~tens of calls (canned refusals /
  non-analyses). Fresh process recovers immediately — restart OMP (or the
  stdio process) to recover. Hypotheses in the doc; no fix implemented.
- **`rapidocr-onnxruntime` 1.4.x requires Python <3.13** (PyPI metadata) —
  Python 3.13/3.14 get 1.2.3 via env markers. 1.2.x returns
  `(result, elapse)`; handled in `ocr.py`.
- **`MISTRAL_API_KEY` is not registered in mcp.json** — `extract_text` with
  `engine=mistral` errors until the key is added. Default (`vlm`) and
  `rapidocr` work out of the box.
- **OMP default MCP timeout is 30s** — registration uses 120s; keep it (the
  vision route can exceed 30s; rapidocr init needs ~3s first call).
- Benchmarks/ outputs (`benchmarks/`) are gitignored, local-only; the data
  contains model outputs over private-project screenshots — don't commit.
- The MCP client "about"/description: set via `gh repo edit`; README + about
  must stay in sync.

## Open items / next steps

- **Release & packaging** (full plan in `IMPLEMENTATION.md`; parity with
  scrape-n-search): add LICENCE (MIT, pending approval) → pyproject metadata
  (`readme`, `license`, `[project.urls]`) → `.github/workflows/ci.yml`
  (unittest, Python 3.10/3.14) → `.github/workflows/release.yml` (test →
  build → GitHub release → PyPI trusted publishing) → register the PyPI trusted
  publisher (manual, one-time) → `git tag v0.1.0 && git push origin v0.1.0`,
  verify `uvx --from 'look-n-see[ocr]' look-n-see`.
- **Optional B-scripts** (TBD): `look-n-see-vision`/`look-n-see-ocr` console
  scripts that preset the env flags — only if a role must be hard-excluded at
  startup.
- **Session state** — spec'd in `BRIEF.md` ("Future: session state"), not
  implemented; `client.complete` is the seam.
- **Vision-collapse follow-ups** (from `VISION_SESSION_COLLAPSE.md`): raw
  response logging, self-check/auto-restart after N non-conforming responses,
  a `health` tool.
- **Model-chain reorder (recommended, pending wider validation)** — the
  multi-model comparison measured gemma-4-31b-it at accuracy equal to the
  26b default with 1.6–3× lower latency (vision + vlm-ocr). Candidates after
  a 15–20 image confirmation run: `VISION_MODEL=openrouter:google/gemma-4-31b-it`
  with 26b as first fallback. qwen3-vl-30b not recommended for the chain
  (fabrication); gpt-4o-mini not recommended for OCR (coverage).
- **Extended live testing (partially done)** — comparison runs exercised all
  three OCR engines live (vlm/rapidocr via scripts; mistral still untested
  key-wise). Remaining: drive the mounted tools through the registered server
  after an OMP restart, and optionally add `MISTRAL_API_KEY` to mcp.json.

## Note

The collapse-incident reproduction and the benchmark sweep are documented
outputs of previous sessions — keep the JSONL/data local (gitignored), and
treat vendor-reported benchmark numbers as dated claims, not facts.
