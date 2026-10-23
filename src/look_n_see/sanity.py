"""Output-shape classification shared by the tools and the benchmark harness.

Ported from the vision-collapse investigation (`VISION_SESSION_COLLAPSE.md`).
Refusal/gen-payload detection is anchored to sentence starts and self-reference
so an ordinary analysis containing "unable to read the small label" is not
mistaken for a refusal; the tool layer surfaces a warning for the severe flags.
"""

from __future__ import annotations

import re

# Sentence-initial or clearly self-referential refusal language. Anchoring
# avoids flagging descriptive text such as "the dialog is unable to load".
REFUSAL_RE = re.compile(
    r"(?:^|[.!?]\s+)("
    r"as an ai\b"
    r"|i am (?:a )?(?:text-based|ai|language model)\b"
    r"|i (?:cannot|can't|am unable to|don't have the ability|do not have the ability)\b"
    r"|i do not have the capability\b"
    r"|unable to (?:fulfill|assist|process|comply|generate|render|export)\b"
    r")",
    re.IGNORECASE,
)
GEN_PAYLOAD_PATTERNS = [
    "dalle",
    "text2im",
    '"model": "dall',
    "generate an image",
    "cannot generate, render, or export",
]
JSON_MARKER_RE = re.compile(r'"point"\s*:|"label"\s*:|"x"\s*:\s*["\d-]|"y"\s*:\s*["\d-]')
# Flags that indicate provider/session degradation worth telling the caller
# about. `terse` and `json_markers` are quality signals, not failures.
SEVERE_FLAGS = frozenset({"empty", "refusal", "gen_payload"})


def classify(text: str) -> list[str]:
    low = (text or "").lower()
    flags: list[str] = []
    if not text or not text.strip():
        flags.append("empty")
    if REFUSAL_RE.search(text or ""):
        flags.append("refusal")
    if any(p in low for p in GEN_PAYLOAD_PATTERNS):
        flags.append("gen_payload")
    if JSON_MARKER_RE.search(text or "") or (text or "").lstrip().startswith(("{", "[")):
        flags.append("json_markers")
    if len((text or "").strip()) < 60 and not flags:
        flags.append("terse")
    if not flags:
        flags.append("normal")
    return flags


def severe_flags(text: str) -> list[str]:
    """Subset of `classify()` flags that warrant a caller-visible warning."""
    return [flag for flag in classify(text) if flag in SEVERE_FLAGS]


def annotate(text: str) -> str:
    """Append a warning line when a response looks like a degraded generation."""
    flags = severe_flags(text)
    if not flags:
        return text
    note = (
        "[look-n-see warning: response flagged as "
        + ", ".join(flags)
        + "; possible provider/session degradation - retry or restart the server]"
    )
    return f"{text}\n\n{note}" if text else note
