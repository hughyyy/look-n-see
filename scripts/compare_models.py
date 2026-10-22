#!/usr/bin/env python3
"""Multi-model x multi-mode comparison over text-dense screenshots.

Modes (server tool paths):
  vision    - analyze_screenshot prompt (SCREENSHOT_DEFAULT)
  vlm-ocr   - extract_text with engine=vlm (OCR_DEFAULT prompt)
  rapidocr  - extract_text with engine=rapidocr (local reference signal)

For each image each model runs vision + vlm-ocr; rapidocr runs once as an
independent local reference. Metrics without ground truth are consensus
based (see summarize): recall of the multi-model consensus tokens, and
coverage of the local-OCR token set.

Usage:

    OPENROUTER_API_KEY=... \
        .venv/bin/python scripts/compare_models.py \
        --images /tmp/omp-sshots-15af255cc7366d43.webp,... \
        --models 'openrouter:google/gemma-4-26b-a4b-it openrouter:google/gemma-4-31b-it ...'

Writes results.jsonl (resumable) + summary.md under benchmarks/compare-models-<ts>/.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from look_n_see.client import VisionClient
from look_n_see.config import load_config
from look_n_see.images import to_data_uri
from look_n_see.ocr import OcrDispatcher
from look_n_see.prompts import SCREENSHOT_DEFAULT, build_messages

from compare_modes import open_nofollow_append, tokens, word_count

DEFAULT_MODELS = (
    "openrouter:google/gemma-4-26b-a4b-it",
    "openrouter:google/gemma-4-31b-it",
    "openrouter:google/gemma-3-27b-it",
    "openrouter:qwen/qwen3-vl-30b-a3b-instruct",
    "openrouter:openai/gpt-4o-mini",
)


def run_call(client, dispatcher, path: str, mode: str, model: str) -> dict:
    uri = to_data_uri(path)
    row: dict = {"path": path, "mode": mode, "model": model}
    t0 = datetime.now(timezone.utc)
    try:
        if mode == "vision":
            meta = client.complete_with_meta(
                build_messages(uri, None, SCREENSHOT_DEFAULT), model=model, fallbacks=()
            )
            row.update(ok=True, text=meta.text, model=model)
        elif mode == "vlm-ocr":
            raw = dispatcher.extract(uri, engine="vlm", model=model)
            row.update(ok=True, text=json.loads(raw).get("text", ""))
        else:  # rapidocr: independent local reference, run once per image
            raw = dispatcher.extract(uri, engine="rapidocr")
            parsed = json.loads(raw)
            row.update(ok=True, text=parsed.get("text", ""))
            items = parsed.get("items") or []
            row["engine_meta"] = {
                "items": len(items),
                "mean_conf": round(statistics.mean(float(i["confidence"]) for i in items), 3)
                if items
                else None,
            }
    except Exception as exc:  # noqa: BLE001 - record and continue
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}", text="")
    row["latency_s"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
    row["ts"] = t0.isoformat()
    row["chars"] = len(row.get("text") or "")
    row["words"] = word_count(row.get("text") or "")
    return row


def _load_existing(run_dir: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    f = run_dir / "results.jsonl"
    if not f.exists():
        return done
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            done.add((row["path"], row["mode"], row.get("model") or ""))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def summarize(run_dir: Path, images: list[str], models: list[str]) -> dict:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cells: dict[tuple[str, str, str], dict] = {(r["path"], r["mode"], r.get("model", "")): r for r in rows}

    def cell(path, mode, model):
        return cells.get((path, mode, model))

    # --- vlm-ocr consensus analysis (per image) ---
    per_image: dict[str, dict] = {}
    for path in images:
        ocr_models = [m for m in models if cell(path, "vlm-ocr", m) and cell(path, "vlm-ocr", m)["ok"]]
        rapid = cell(path, "rapidocr", "")
        rapid_toks = tokens(rapid["text"]) if rapid and rapid["ok"] else set()

        # consensus = tokens shared by >= 2 models (single-model noise excluded)
        model_toks = {m: tokens(cell(path, "vlm-ocr", m)["text"]) for m in ocr_models}
        counts: dict[str, int] = {}
        for mt in model_toks.values():
            for t in mt:
                counts[t] = counts.get(t, 0) + 1
        consensus = {t for t, c in counts.items() if c >= 2 and len(ocr_models) > 1}
        unique_any = {t for t, c in counts.items() if c == 1}

        model_stats = {}
        for m in ocr_models:
            mt = model_toks[m]
            inter = []
            for m2 in ocr_models:
                if m2 != m:
                    other = model_toks[m2]
                    inter.append(
                        len(mt & other) / len(mt | other) if (mt | other) else 1.0
                    )
            cons_rec = len(mt & consensus) / len(consensus) if consensus else None
            loc_rec = len(mt & rapid_toks) / len(rapid_toks) if rapid_toks else None
            uniq = mt & unique_any
            # unique tokens that local OCR also saw -> likely real text others missed
            uniq_real = (uniq & rapid_toks) if rapid_toks else set()
            model_stats[m] = {
                "words": len(mt),
                "mean_pairwise_jaccard": round(statistics.mean(inter), 3),
                "consensus_recall": round(cons_rec, 3) if cons_rec is not None else None,
                "local_recall": round(loc_rec, 3) if loc_rec is not None else None,
                "unique_tokens": sorted(uniq),
                "unique_real_checked": sorted(uniq_real),
            }
        per_image[path] = {
            "consensus_size": len(consensus),
            "models": model_stats,
            "rapid": {
                "words": len(rapid_toks),
                "items": (rapid or {}).get("engine_meta", {}).get("items"),
                "mean_conf": (rapid or {}).get("engine_meta", {}).get("mean_conf"),
            },
        }

    # --- vision mode: local-text surfacing + latency ---
    vision_stats: dict[str, dict] = {}
    lat: dict[str, list[float]] = {m: [] for m in models}
    for model in models:
        cover = []
        ok = 0
        for path in images:
            v = cell(path, "vision", model)
            rapid = cell(path, "rapidocr", "")
            if v and v["ok"]:
                ok += 1
                lat[model].append(v["latency_s"])
                if rapid and rapid["ok"]:
                    rt = tokens(rapid["text"])
                    if rt:
                        cover.append(len(tokens(v["text"]) & rt) / len(rt))
        vision_stats[model] = {
            "ok": ok,
            "local_text_surfacing": round(statistics.mean(cover), 3) if cover else None,
            "latency_median_s": round(statistics.median(lat[model]), 2) if lat[model] else None,
        }
    for model in models:
        rows_m = [r for r in rows if r["mode"] == "vlm-ocr" and r["model"] == model and r["ok"]]
        lat[model] = [r["latency_s"] for r in rows_m]
        vision_stats[model]["vlm_ocr_latency_median_s"] = (
            round(statistics.median(lat[model]), 2) if lat[model] else None
        )
    return {"per_image": per_image, "vision": vision_stats, "rows": rows}


def write_summary(run_dir: Path, summary: dict, models: list[str]) -> None:
    out = ["# Multi-model x mode comparison (text-dense screenshots)\n"]
    out.append("## Per-model overview (5 images x 2 modes)")
    out.append("| model | vision ok | vision surfacing (loc-text) | vision med s | vlm-ocr med s |")
    out.append("|---|---|---|---|---|")
    for m, s in summary["vision"].items():
        out.append(
            f"| {m} | {s['ok']}/5 | {s['local_text_surfacing']} | {s['latency_median_s']} | "
            f"{s['vlm_ocr_latency_median_s']} |"
        )
    out.append("")
    for path, pi in summary["per_image"].items():
        name = Path(path).name
        out.append(f"## {name}")
        out.append(f"- consensus OCR tokens: {pi['consensus_size']} | rapidocr: {pi['rapid']['words']} words, "
                   f"{pi['rapid']['items']} items, conf {pi['rapid']['mean_conf']}")
        out.append("| model | words | pairwise-j | consensus-rec | local-rec | unique txt |")
        out.append("|---|---|---|---|---|---|")
        for m, st in pi["models"].items():
            uniq = " ".join(st["unique_tokens"][:12])
            if len(st["unique_tokens"]) > 12:
                uniq += "..."
            out.append(f"| {m} | {st['words']} | {st['mean_pairwise_jaccard']} | "
                       f"{st['consensus_recall']} | {st['local_recall']} | {uniq[:100]} |")
        out.append("")
    # vision excerpts for the write-up
    rows = summary["rows"]
    out.append("## Vision-mode first lines (app identification)")
    for path in sorted({r["path"] for r in rows}):
        out.append(f"### {Path(path).name}")
        for model in models:
            r = next((x for x in rows if x["path"] == path and x["mode"] == "vision" and x["model"] == model), None)
            if r and r["ok"]:
                first = next((ln for ln in r["text"].splitlines() if ln.strip()), "")
                out.append(f"- {model}: {first[:110]}")
        out.append("")
    (run_dir / "summary.md").write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="comma-separated image paths")
    ap.add_argument("--models", help="space/comma separated model ids (default: repo chain + 3 more)")
    ap.add_argument("--out", default="benchmarks", help="output root (default: benchmarks)")
    ap.add_argument("--skip-rapidocr", action="store_true", help="skip local reference rows")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    images = [p.strip() for p in args.images.split(",") if p.strip()]
    missing = [p for p in images if not os.path.isfile(p)]
    if missing:
        print(f"error: missing images: {missing}", file=sys.stderr)
        return 2
    models = tuple(m for m in (args.models or " ").replace(",", " ").split() if m.strip())
    if not models:
        models = DEFAULT_MODELS

    cfg = load_config()
    if not cfg.openrouter_api_key:
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / f"compare-models-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    client = VisionClient(cfg)
    dispatcher = OcrDispatcher(cfg, vision_client=client)

    jobs: list[tuple[str, str, str]] = []
    for path in images:
        for model in models:
            jobs.append((path, "vision", model))
            jobs.append((path, "vlm-ocr", model))
        if not args.skip_rapidocr:
            jobs.append((path, "rapidocr", ""))
    done = set() if args.no_resume else _load_existing(run_dir)
    todo = [j for j in jobs if j not in done]
    print(f"images: {len(images)} models: {len(models)} pending: {len(todo)} (done: {len(done)})")

    for i, (path, mode, model) in enumerate(todo, 1):
        row = run_call(client, dispatcher, path, mode, model)
        with open_nofollow_append(run_dir / "results.jsonl") as fh:
            fh.write(json.dumps(row) + "\n")
        tag = model or "rapidocr"
        status = "ok" if row["ok"] else f"ERR {row.get('error', '')[:60]}"
        print(f"[{i}/{len(todo)}] {Path(path).name} {mode} {tag}: {status} ({row['latency_s']}s, {row['words']}w)")

    summary = summarize(run_dir, images, models)
    write_summary(run_dir, summary, models)
    print(f"\ncomplete: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
