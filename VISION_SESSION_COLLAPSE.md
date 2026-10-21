# Issue: vision MCP server collapsed mid-session (canned refusals / non-analyses)

Date: 2026-10-21/22
Reporter: agent session running the per-component visual audit of a Vue demo app
Severity: medium (diagnostic tool degraded; workaround = restart process)

## Summary

During a long agent session performing ~60 image analyses through this MCP
server's `analyze_screenshot` tool, the server's outputs **degenerated over
time**: coherent (if sometimes confabulating) analyses gave way to JSON
point-marker dumps, an invented Vue/Tailwind component reconstruction, a
`dalle.text2im` tool-call payload, and finally **canned text-only refusals**
("I am a text-based AI and do not have the capability to generate, render, or
export image files"). A fresh process with the same client, model, and image
answered correctly — so the **model and the image are not the problem; the
long-lived server process is the suspect**.

## Server identity (facts)

- Launcher (from `~/.omp/agent/mcp.json`):
  `type: stdio`, `command: <checkout>/.venv/bin/look-n-see`,
  `env: HF_TOKEN = omp token huggingface`, `timeout: 120000`
- Stack (per `IMPLEMENTATION.md`): Python 3.14, `mcp` 2.x, `huggingface_hub`
  `InferenceClient` (serverless inference), stdlib image parsing.
- Model: `google/gemma-4-26B-A4B-it` (env `VISION_MODEL`); `HF_ENDPOINT` can
  override; `VISION_MAX_TOKENS=1024`, `VISION_TIMEOUT=60`, `VISION_RETRIES=3`
  (backoff 2s). Tools: `describe_image` / `analyze_screenshot` /
  `ask_about_image`, each with an optional steering prompt.
- **Stateless by design**: `client.complete(messages)` is a pure function of
  messages (per `IMPLEMENTATION.md`); no session history is implemented.

## Observed degradation pattern (chronological, all in one session)

| Phase | Behaviour | Example |
| --- | --- | --- |
| Early | Structured analyses; occasional single-item confabulation on busy/dim images | "no drawer visible" when DOM proved it rendered; "continuous line" on a chart with a proven 2-segment gap |
| Mid | Output-format imitation: terse `NONE`, then `{"point":[x,y], "label": ...}` markers — some fabricated (geometry disproved them) | stepper "clipped labels" at pixel coords with zero measured overflow |
| Late | Task confusion: image analysis treated as code generation / image generation | calendar toolbar card → fake Vue/Tailwind component source; scheduler card → `dalle.text2im` action payload |
| Terminal | Canned refusals from a text-only path | `overlays-06-blockui.png` → "I cannot fulfill this request… largely obscured or clipped", then "I am a text-based AI and do not have the capability to generate, render, or export image files." (This same image analysed correctly by a fresh process.) |

## Evidence

1. **Fresh-process control (strongest evidence):**
   `HF_TOKEN=$(omp token huggingface) ./.venv/bin/python scripts/live_smoke.py /tmp/vdaudit/overlays-06-blockui.png "Describe only what is visible. Is there a dark mask over a pane with a Wait message chip?"`
   → `Yes, there is a dark mask over a pane with a "Wait..." message chip.`
   The exact image the session server refused; correct answer from a fresh
   process, same model, same client code.
2. **Endpoint healthy:** `scripts/live_smoke.py` (default sample) → correct
   description of the sample.
3. **Environment checks passed:** `HF_TOKEN` resolves via
   `~/.local/bin/omp token huggingface`; server launched with no
   `VISION_MODEL`/`HF_ENDPOINT` overrides (defaults apply).
4. **Deterministic counterpart:** the DOM/geometry layer (Playwright
   `getBoundingClientRect`, `elementFromPoint`, computed styles) adjudicated
   every vision claim; all fixed defects were independently confirmed by it,
   including the image above (mask present, `blocked: true`, "Wait…" text).
5. Images referenced are archived in
   `<app checkout>/docs/assets/visual-audit-2026-10-21.tar.gz`
   (extracted `vdaudit/`), live copies in `/tmp/vdaudit/` (volatile).

## Hypotheses (unverified)

- **H1 (leading): the long-lived server process's InferenceClient connection
  or the serverless session it pins degrades** (serverless backends can pin
  per-connection workers; a degraded worker would explain fallback
  generations and text-only refusals while a fresh process gets a healthy
  worker). Retry/backoff logic exists but only fires on 429/5xx/connection
  errors — a 200-with-garbage response is not detected.
- **H2: provider-side routing/safety layer** intermittently intercepts and
  answers with text-only boilerplate; fresh connections happen to route
  differently.
- **H3: prompt/context drift** — the repetition of near-identical prompts
  across ~60 calls nudged the model into template outputs. (Does not explain
  the canned refusals, which are unlike the model's normal failures.)

## Instructions for the investigating agent

1. **Reproduce cleanly:**
   - Start a fresh `look-n-see` process and, through `analyze_screenshot`
     (or a raw call via `scripts/live_smoke.py`), analyse
     `/tmp/vdaudit/overlays-06-blockui.png` (or the tarball copy) with the
     prompt from the Evidence section. Expect the correct mask description.
   - Using the SAME long-lived session path (the mcp.json stdio process),
     repeat the call ~20–40 times on varied images and record when/if output
     transitions to refusals or non-analyses. Note the call count and any
     response-size/format change.
2. **Capture the raw layer:** `client.complete()` currently returns text
   only. Add temporary logging (or a wrapper) that dumps the **full raw
   response JSON** (choices, finish_reason, logprobs, usage) for every call —
   distinguish model-generated refusal text from client/provider errors.
   Check `finish_reason` and HTTP status on the degrading calls.
3. **Connection lifecycle:** inspect `huggingface_hub` `InferenceClient`
   session reuse (`client.py`) — does one `InferenceClient` live for the
   process lifetime? Test recreating the client per request, or per N
   requests, and compare longevity. Check whether `close()` is ever called.
4. **Restart behaviour:** restart the stdio server mid-session (via the
   MCP supervisor) and confirm `analyze_screenshot` recovers immediately on
   the previously-refused image — this validates H1 vs H2.
5. **Provider routing:** run with `VISION_MODEL` pointed at the same
   architecture/size from a different org (or an `HF_ENDPOINT` dedicated
   endpoint if one is configured) and repeat the 20–40-call sequence.
6. **Defensive fixes to propose (do not implement without approval):**
   - Startup/periodic self-check: on each tool call, optionally sanity-check
     the response (non-empty, matches expected shape) and surface a warning.
   - Auto-restart or client-recreation after N consecutive non-conforming
     responses (or after `VISION_RETRIES`-style thresholds).
   - Raw-response logging (see 2) shipped by default for diagnosis.
   - A `health` tool or `scripts/live_smoke.py --json` for CI-able checks.
7. **Report back with:** call-count-to-degradation, first non-conforming
   output verbatim + its raw JSON, whether restart recovered it, and one
   paragraph recommending the fix with the evidence for it.
