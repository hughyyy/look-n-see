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
pyproject.toml
README.md            # user-facing: tools, install, config, dev
BRIEF.md             # requirements
IMPLEMENTATION.md    # this file: design, OMP integration, release plan
VISION_SESSION_COLLAPSE.md  # known issue: long-lived process degradation
src/look_n_see/
  __init__.py
  __main__.py        # python -m look_n_see
  config.py          # env parsing + defaults
  images.py          # file path / data URI / URL -> (media_type, bytes) -> data URI
  client.py          # provider clients: retry/backoff, fallback chain
  openrouter.py      # OpenRouter chat-completions transport (httpx)
  prompts.py         # default prompts per tool
  server.py          # MCPServer app + tool registrations + main()
tests/
  test_images.py
  test_prompts.py
  test_openrouter.py
  test_config.py     # env parsing, provider routing, fallback parsing
  test_client.py     # retries, fallback chain, transports (mocked)
  test_server.py     # registration smoke + mocked client
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
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | — | optional `HTTP-Referer` / `X-Title` headers |

## Tools (all accept optional prompt to steer output, and optional model)
1. `describe_image(image, prompt=None, model=None)` — general description.
2. `analyze_screenshot(image, prompt=None, model=None)` — UI/state analysis.
3. `ask_about_image(image, question, model=None)` — Q&A over the image.

`image` accepts: file path, base64 data URI, or `http(s)://` URL (auto-detect).
`model` accepts any id in the same syntax as `VISION_MODEL`; omitted = default.
On failure the call walks the fallback chain (`VISION_FALLBACKS`).

## Benchmark suite (`scripts/benchmark.py` + `look_n_see.benchmark`)
- Run: models × images × reps, streaming JSONL, resumable; per-call metadata
  (provider, transport, latency, finish_reason, usage, retries).
- Summarize: per-model stats, output-shape flags (refusals, JSON-marker dumps,
  empty/terse — ported from the vision-collapse classifier), repetition rate,
  flagged samples; writes by-model views + `analysis_prompt.md` for an
  evaluation agent.
- Benchmark rows disable fallbacks so attribution is exact.
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
- Retry 429/5xx/connection errors with exponential backoff (`VISION_RETRIES`).
- Invalid image input → descriptive tool error.
- HF failures → readable message (model target + status), not a traceback.

## Oh My Pi (OMP) integration
The server is registered as a user-level MCP server so agents can call it from
any working directory.

- Config: `~/.omp/agent/mcp.json` →
  `look-n-see` (stdio, `.venv/bin/look-n-see`,
  `HF_TOKEN` resolved from OMP's own HuggingFace credential store via
  `!/home/bernie/.local/bin/omp token huggingface`,
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
  in an OMP session. The three tools (`describe_image`, `analyze_screenshot`,
  `ask_about_image`) then appear alongside built-in tools for the main agent
  and subagents.
- To make it project-scoped instead of user-scoped, move the same block to
  `.omp/mcp.json` in this repo.

## Release & packaging plan (parity with scrape-n-search)

**Status: planned — nothing published or configured yet.** Target: mirror
scrape-n-search's setup — `ci.yml` + `release.yml` GitHub Actions, distribution
via PyPI with trusted publishing (OIDC, no tokens), and `uvx`/`pipx`
installability, with release flow fully automated from a tag.

Facts already checked:
- GitHub repo `hughyyy/look-n-see` exists (created empty; default branch
  `main` — local branch renamed from `master` to match).
- `look-n-see` is available on PyPI (404 today); `look-n-see-mcp` also free as
  an alternative if the plain name is taken by the time we publish.
- Tests are stdlib `unittest` with zero dev deps — CI runs `unittest`, not
  pytest.

Planned steps, in order:

1. **Add `LICENSE` (MIT)** — same as scrape-n-search. Pending user approval
   (legal choice).
2. **pyproject metadata**: add `readme = "README.md"`, `license = "MIT"`,
   `[project.urls]` (repository → `https://github.com/hughyyy/look-n-see`),
   keywords/classifiers. Name stays `look-n-see`; `requires-python >=3.10`.
3. **`.github/workflows/ci.yml`**: tests on push to `main` + PRs, Python
   matrix `["3.10", "3.14"]`, steps: checkout → setup-python → `pip install -e .`
   → `python -m unittest discover -s tests -q`.
4. **`.github/workflows/release.yml`**: on tag push `v*` — job 1 tests
   (same as CI), job 2 builds sdist + wheel (`pip install build` → `python -m
   build`), uploads `dist/` artifact; job 3 creates the GitHub Release with
   auto-generated notes and attaches artifacts
   (`gh release create "$GITHUB_REF_NAME" dist/* --generate-notes`);
   job 4 publishes to PyPI via `pypa/gh-action-pypi-publish@release/v1`
   (trusted publishing).
5. **PyPI one-time admin step** (manual, outside the repo): register the
   trusted publisher for project `look-n-see` → `hughyyy/look-n-see` (workflow
   `release.yml`, environment `release` if used). Until then the publish job
   fails with a clear OIDC error, as in scrape-n-search.
6. **First release**: `git tag v0.1.0 && git push origin v0.1.0`; verify on a
   clean machine with `uvx look-n-see` (registers as `look-n-see` console
   script), then point MCP client config at `uvx look-n-see`.
7. **Optional parity (deferred)**: one-line `install.sh` + `setup.sh`
   installer like scrape-n-search's (source tarball → user venv → harness
   registration), and a PyPI classifiers pass.

Non-goals for packaging: no Docker image, no npm/JS packaging, no HTTP
transport — stdio only, per BRIEF.

## Verification
- Offline (this session): `compileall`, server starts and registers 3 tools,
  image parsing unit tests (file/data-uri/URL), prompts unit tests, mocked
  client end-to-end tool test.
- Live (user, later): `HF_TOKEN` set + `scripts/live_smoke.py` — one real
  inference call. Not run by me; user does this.
