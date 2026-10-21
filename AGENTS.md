# AGENTS.md — look-n-see

## Project
MCP server exposing a vision model via HuggingFace Inference. See `BRIEF.md`.

## Conventions
- Python 3.14. No Node/Cargo toolchains on this workstation.
- Keep the server dependency-light: MCP SDK + `huggingface_hub` only.
- Every MCP tool must return structured output the calling model can act on
  (JSON where it matters, not prose-only).
- Secrets (HF token) come from environment, never hardcoded or committed.
- Config (model id, endpoint URL, timeouts) via environment variables with
  sane defaults.

## Workflow
- Read `BRIEF.md` before starting work.
- Keep the brief in sync with decisions made during implementation.
- One commit per logical change; conventional commit messages.

## Verification
- `python -m compileall` / import smoke test after changes.
- MCP server must start and register tools without a live HF call
  (tool registration must not require network).
- Live HF calls are integration tests, gated behind an env flag.
