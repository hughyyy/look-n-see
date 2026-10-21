"""Benchmark harness: many models x many images x N reps, recorded for analysis.

Layout of a run directory (default `benchmarks/run-<timestamp>/`):

    run.json            metadata: models, images, prompt, config, versions
    results.jsonl       one JSON object per call, streaming-appended
    by-model/<slug>.jsonl  per-model views of the same rows
    summary.json        per-model stats + flagged-call samples (summarize())
    analysis_prompt.md  ready-to-hand prompt for an evaluation agent

`run_benchmark` is resumable: existing (model, image, rep) keys are skipped.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .client import Completion, InferenceError, VisionClient
from .images import ImageInputError, to_data_uri
from .prompts import SCREENSHOT_DEFAULT, build_messages

# --- output-shape classifiers (port of the vision-collapse investigation) -----

REFUSAL_PATTERNS = [
    "text-based ai",
    "do not have the capability",
    "cannot fulfill",
    "unable to",
    "as an ai",
    "i cannot",
    "i don't have the ability",
    "largely obscured or clipped",
    "cannot generate, render, or export",
    "i am an ai language model",
]
GEN_PAYLOAD_PATTERNS = ["dalle", "text2im", '"model": "dall', "generate an image", "generate, render, or export"]
JSON_MARKER_RE = re.compile(r'"point"\s*:|"label"\s*:|"x"\s*:\s*["\d-]|"y"\s*:\s*["\d-]')


def classify(text: str) -> list[str]:
    low = (text or "").lower()
    flags: list[str] = []
    if not text or not text.strip():
        flags.append("empty")
    if any(p in low for p in REFUSAL_PATTERNS):
        flags.append("refusal")
    if any(p in low for p in GEN_PAYLOAD_PATTERNS):
        flags.append("gen_payload")
    if JSON_MARKER_RE.search(text) or text.lstrip().startswith(("{", "[")):
        flags.append("json_markers")
    if len(text.strip()) < 60 and not flags:
        flags.append("terse")
    if not flags:
        flags.append("normal")
    return flags


@dataclass(frozen=True)
class BenchmarkSpec:
    models: tuple[str, ...]
    images: tuple[str, ...]
    reps: int = 5
    prompt: str = SCREENSHOT_DEFAULT
    out_dir: str = "benchmarks"


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", ".", model_id)


def _write_row(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def run_benchmark(
    spec: BenchmarkSpec,
    client: VisionClient,
    *,
    resume: bool = True,
    progress: bool = True,
) -> Path:
    """Run models x images x reps through the client and record every call.

    Returns the run directory. `client` must expose `complete_with_meta(
    messages, model=...)`; benchmark calls pass `fallbacks=()` so every row
    is attributed to the requested model, never a silent fallback.
    """
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "models": list(spec.models),
        "images": [os.path.basename(p) for p in spec.images],
        "reps": spec.reps,
        "prompt": spec.prompt,
        "python": sys.version.split()[0],
        "hf_hub_version": _version("huggingface_hub"),
        "httpx_version": _version("httpx"),
        "config": {
            "model": client.config.model,
            "provider": client.config.provider,
            "endpoint": client.config.endpoint,
            "fallbacks": list(client.config.fallbacks),
            "max_tokens": client.config.max_tokens,
            "timeout": client.config.timeout,
            "retries": client.config.retries,
        },
    }
    run_file = out_dir / "run.json"
    if not run_file.exists():
        run_file.write_text(json.dumps(run_meta, indent=2) + "\n")

    results_file = out_dir / "results.jsonl"
    done: set[tuple[str, str, int]] = set()
    if resume and results_file.exists():
        for line in results_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["model"], row["image"], row["rep"]))

    if progress:
        print(f"run: {out_dir}  models={len(spec.models)} images={len(spec.images)} reps={spec.reps}",
              file=sys.stderr)

    total = 0
    for model_id in spec.models:
        for image_path in spec.images:
            image_name = os.path.basename(image_path)
            try:
                uri = to_data_uri(image_path)
            except ImageInputError as exc:
                row = {
                    "model": model_id,
                    "image": image_name,
                    "rep": 0,
                    "prompt": spec.prompt,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "ok": False,
                    "error": f"image read failed: {exc}",
                    "flags": ["image_error"],
                }
                _write_row(results_file, row)
                continue
            messages = build_messages(uri, spec.prompt, spec.prompt)
            for rep in range(1, spec.reps + 1):
                if (model_id, image_name, rep) in done:
                    continue
                row = _call_row(
                    model_id, image_name, rep, spec.prompt, messages, client
                )
                _write_row(results_file, row)
                total += 1
                if progress and total % 10 == 0:
                    print(f"  {total} new calls recorded", file=sys.stderr)
    if progress:
        print(f"done: {total} new rows in {results_file}", file=sys.stderr)
    return out_dir


def _version(pkg: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(pkg)
    except Exception:
        return "?"


def _call_row(
    model_id: str,
    image_name: str,
    rep: int,
    prompt: str,
    messages: list[dict],
    client: VisionClient,
) -> dict:
    row: dict = {
        "model": model_id,
        "image": image_name,
        "rep": rep,
        "prompt": prompt,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        completion: Completion = client.complete_with_meta(
            messages, model=model_id, fallbacks=()
        )
        row.update(
            ok=True,
            text=completion.text,
            transport=completion.transport,
            provider=completion.provider,
            retries=completion.retries,
            latency_s=completion.latency_s,
            finish_reason=completion.finish_reason,
            usage=completion.usage,
        )
    except InferenceError as exc:
        row.update(ok=False, error=str(exc), retries=0)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        row.update(ok=False, error=f"{exc.__class__.__name__}: {exc}")
    row["flags"] = classify(row.get("text") or "") if row.get("ok") else [
        "error"
    ]
    return row


def summarize(out_dir: str | Path, *, write_files: bool = True) -> dict:
    """Per-model stats + flagged samples; writes summary.json and by-model/*.jsonl."""
    out_dir = Path(out_dir)
    run_meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    summary: dict = {"run": run_meta, "models": {}}
    for model_id, model_rows in by_model.items():
        ok = [r for r in model_rows if r.get("ok")]
        failed = [r for r in model_rows if not r.get("ok")]
        latencies = [r["latency_s"] for r in ok]
        texts = [r["text"] for r in ok]
        usage = [r.get("usage") or {} for r in ok]
        flags: dict[str, int] = {}
        for r in model_rows:
            for f in r.get("flags") or []:
                flags[f] = flags.get(f, 0) + 1
        errors: dict[str, int] = {}
        for r in failed:
            errors[r.get("error") or "unknown"] = errors.get(r.get("error") or "unknown", 0) + 1
        flagged = [
            {
                "image": r["image"],
                "rep": r["rep"],
                "flags": r["flags"],
                "snippet": (r.get("text") or r.get("error") or "")[:200],
            }
            for r in model_rows
            if set(r.get("flags") or []) - {"normal"}
        ][:20]
        summary["models"][model_id] = {
            "calls": len(model_rows),
            "ok": len(ok),
            "failed": len(failed),
            "errors": errors,
            "flag_counts": flags,
            "flagged_samples": flagged,
            "latency_s": _quantiles(latencies) if latencies else None,
            "text_len_median": _median([len(t) for t in texts]) if texts else None,
            "usage_totals": {
                "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            },
            "finish_reasons": _counts(r.get("finish_reason", "—") for r in ok),
            "retries_used": _counts(r.get("retries", 0) for r in ok),
            "repetition": _repetition_rate(model_rows),
        }

    if write_files:
        by_model_dir = out_dir / "by-model"
        by_model_dir.mkdir(exist_ok=True)
        for model_id, model_rows in by_model.items():
            with open(by_model_dir / f"{_slug(model_id)}.jsonl", "w", encoding="utf-8") as fh:
                for r in model_rows:
                    fh.write(json.dumps(r) + "\n")
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "analysis_prompt.md").write_text(
            _analysis_prompt(out_dir, list(by_model)), encoding="utf-8"
        )
    return summary


def _repetition_rate(model_rows: list[dict]) -> float | None:
    """Share of (model, image) pairs whose N reps are byte-identical."""
    pairs: dict[tuple, list[str]] = {}
    for r in model_rows:
        if r.get("ok"):
            pairs.setdefault((r["model"], r["image"]), []).append(r["text"])
    if not pairs:
        return None
    identical = sum(1 for texts in pairs.values() if len(texts) > 1 and len(set(texts)) == 1)
    return identical / len(pairs)


def _quantiles(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    n = len(values)
    if n == 1:
        return {"min": values[0], "median": values[0], "p90": values[0], "max": values[0]}
    def pct(p: float) -> float:
        return values[min(n - 1, int(p * n))]
    return {
        "min": round(values[0], 2),
        "median": round(statistics.median(values), 2),
        "p90": round(pct(0.90), 2),
        "max": round(values[-1], 2),
    }


def _median(values: list) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _counts(values) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _analysis_prompt(out_dir: Path, models: list[str]) -> str:
    template = """# Vision model benchmark analysis

You are evaluating vision models benchmarked on identical UI-screenshot
analyses. Data is in the run directory:

## Data layout

- `__OUT__/run.json` — metadata: models, images, prompts, configs, versions.
- `__OUT__/summary.json` — per-model stats: call success, latency, token
  usage, output-quality flags (refusal / json-marker dumps / gen-payloads /
  terse / empty), flagged samples, and repetition rate across reps.
- `__OUT__/by-model/<slug>.jsonl` — full response text for every
  (model, image, rep). Models in this run: __MODELS__.
- Images are UI screenshots of PrimeVue-style components ("vuedemond") whose
  filenames indicate the component (e.g. `data-00-datatable-paginator.png`).

Images may be inspected directly at /tmp/vdaudit/ (or the image paths in
run.json) if you have image-reading tools.

## Ground rules

- Assessment must be grounded in the recorded outputs and stats. Quote
  concrete evidence; do not speculate about model internals.
- Compare models only on this benchmark (same images, same prompt, N reps
  each). A claim is *consistent* if it appears in most reps and *suspect* if
  a single rep contradicts the others with no basis. Note disagreement
  between models on the same image when it shows a real difference in
  perception vs. formatting.
- Prefer models that: answer every image, produce structured complete
  UI-state analyses (elements, states, text/values), avoid refusing or
  confabulating coordinates/code, and are consistent across reps.
- Cost and latency matter: note per-1k-token pricing (OpenRouter: see
  openrouter.ai) and measured latency percentiles.

## Output format

Return JSON:
{
  "model_assessments": [
    {"model": "<model id>",
      "strengths": ["..."], "weaknesses": ["..."],
      "quality_score": <1-10>, "latency_penalty": "none|mild|heavy",
      "verdict": "strong candidate|acceptable|weak"}
  ],
  "recommended_default": "<model id, with openrouter: prefix if applicable>",
  "recommendation_rationale": "<one paragraph, evidence-based>",
  "runner_up": "<model id>"
}
"""
    return (
        template.replace("__OUT__", str(out_dir))
        .replace("__MODELS__", ", ".join(models))
    )
