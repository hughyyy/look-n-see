# AGENTS.md — look-n-see

## Project
MCP server exposing a vision model via OpenRouter and HuggingFace Inference.
See `README.md` (user-facing), `BRIEF.md` (requirements), `IMPLEMENTATION.md`
(design + release plan).

## Conventions
- Python 3.10+. No Node/Cargo toolchains on this workstation.
- Server dependency-light: MCP SDK + `huggingface_hub` + `httpx` only. Tests use
  stdlib `unittest` — zero dev deps.
- Every MCP tool must return structured output the calling model can act on
  (JSON where it matters, not prose-only).
- Secrets (HF token, OpenRouter key) come from environment, never hardcoded or
  committed.
- Config (model id, endpoint URLs, timeouts) via environment variables with
  sane defaults — see the `Config` dataclass in `src/look_n_see/config.py`.

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
  parsing, config, prompts, OpenRouter transport, retries/fallbacks, server
  registration + mocked-client tool behavior).
- MCP server must start and register tools without a live HF call
  (tool registration must not require network).
- Live calls are manual: `scripts/live_smoke.py` with the provider token in the
  environment (or via `omp token <provider>`).
- Known issue: long-lived server processes can degrade over many calls
  (canned refusals / non-analyses) — see `VISION_SESSION_COLLAPSE.md`; a fresh
  process recovers immediately.
