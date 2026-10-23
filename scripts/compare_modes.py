#!/usr/bin/env python3
"""Three-mode comparison over screenshots: vision mode vs VLM-as-OCR vs local RapidOCR.

Modes mirror the server tools exactly:
  vision    - analyze_screenshot path (SCREENSHOT_DEFAULT via VisionClient)
  vlm-ocr   - extract_text path (OcrDispatcher engine=vlm)
  rapidocr  - extract_text path (OcrDispatcher engine=rapidocr, local ONNX)

Usage (provider tokens in the environment):

    OPENROUTER_API_KEY=... HF_TOKEN=... \
        .venv/bin/python scripts/compare_modes.py \
        --images-dir /tmp --glob 'omp-sshots-*.webp' --limit 15

  --images     explicit comma-separated paths (overrides dir/glob/limit)
  --limit N    evenly spaced sample of N (0 = all)
  --out DIR    output root; run dir created under it (default: benchmarks)

Writes results.jsonl (one row per image x mode, resumable) and summary.md.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from look_n_see.client import VisionClient
from look_n_see.config import Config, load_config
from look_n_see.images import to_data_uri
from look_n_see.ocr import OcrDispatcher
from look_n_see.prompts import SCREENSHOT_DEFAULT, build_messages

MODES = ("vision", "vlm-ocr", "rapidocr")
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def open_nofollow_append(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW
    return os.fdopen(os.open(path, flags, 0o644), "a", encoding="utf-8")


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def select_images(dir_: str, pattern: str, limit: int) -> list[str]:
    paths = sorted(globlib.glob(str(Path(dir_) / pattern)))
    if not paths:
        return []
    if limit <= 0 or limit >= len(paths):
        return paths
    if limit == 1:
        return [paths[0]]
    # Evenly spaced sample that always includes the final image.
    indexes = [(i * (len(paths) - 1)) // (limit - 1) for i in range(limit)]
    return [paths[i] for i in sorted(set(indexes))]


def run_one(dispatcher: OcrDispatcher, client: VisionClient, path: str, mode: str) -> dict:
    uri = to_data_uri(path)
    row: dict = {"path": path, "mode": mode}
    t0 = datetime.now(timezone.utc)
    try:
        if mode == "vision":
            meta = client.complete_with_meta(build_messages(uri, None, SCREENSHOT_DEFAULT))
            row.update(
                ok=True,
                text=meta.text,
                model=meta.model,
                provider=meta.provider,
            )
        else:
            engine = mode.removesuffix("-ocr")
            raw = dispatcher.extract(uri, engine=engine)
            parsed = json.loads(raw)
            row.update(ok=True, text=parsed.get("text", ""))
            if engine == "rapidocr":
                items = parsed.get("items") or []
                row["engine_meta"] = {
                    "items": len(items),
                    "mean_conf": round(
                        statistics.mean(float(i["confidence"]) for i in items), 3
                    )
                    if items
                    else None,
                }
            else:
                row["model"] = parsed.get("model")
    except Exception as exc:  # noqa: BLE001 - record and continue
        row.update(ok=False, error=f"{type(exc).__name__}: {exc}", text="")
    row["latency_s"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 2)
    row["ts"] = t0.isoformat()
    row["chars"] = len(row.get("text") or "")
    row["words"] = word_count(row.get("text") or "")
    return row


def _load_existing(run_dir: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    f = run_dir / "results.jsonl"
    if not f.exists():
        return done
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            done.add((row["path"], row["mode"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def summarize(run_dir: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_image: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_image.setdefault(row["path"], {})[row["mode"]] = row

    agg: dict[str, Any] = {"rows": len(rows), "images": len(by_image)}
    lat: dict[str, list[float]] = {m: [] for m in MODES}
    recall_local, extra_vlm, vision_sees, confs = [], [], [], []
    for path, modes in sorted(by_image.items()):
        for m in MODES:
            if m in modes and modes[m]["ok"]:
                lat[m].append(modes[m]["latency_s"])
        rapid, vlm, vision = modes.get("rapidocr"), modes.get("vlm-ocr"), modes.get("vision")
        if rapid and vlm and rapid["ok"] and vlm["ok"]:
            rt, vt = tokens(rapid["text"]), tokens(vlm["text"])
            if rt:
                recall_local.append(len(rt & vt) / len(rt))
            if vt:
                extra_vlm.append(1 - len(vt & rt) / len(vt))
        if rapid and vision and rapid["ok"] and vision["ok"]:
            rt, vt = tokens(rapid["text"]), tokens(vision["text"])
            if rt:
                vision_sees.append(len(rt & vt) / len(rt))
        meta = (rapid or {}).get("engine_meta") or {}
        if meta.get("mean_conf") is not None:
            confs.append(meta["mean_conf"])
    agg["latency"] = {m: (round(statistics.median(v), 2), round(statistics.mean(v), 2))
                      for m, v in lat.items() if v}
    agg["recall_local_in_vlm"] = round(statistics.median(recall_local), 3) if recall_local else None
    agg["extra_vlm_not_local"] = round(statistics.median(extra_vlm), 3) if extra_vlm else None
    agg["vision_sees_local_text"] = round(statistics.median(vision_sees), 3) if vision_sees else None
    agg["rapidocr_mean_conf"] = round(statistics.mean(confs), 3) if confs else None
    return {"agg": agg, "by_image": by_image, "errors": [r for r in rows if not r["ok"]]}


def write_summary(run_dir: Path, summary: dict) -> None:
    agg, by_image, errors = summary["agg"], summary["by_image"], summary["errors"]
    out = ["# Three-mode comparison over /tmp/omp-sshots\n",
           f"- images: {agg['images']}, rows: {agg['rows']}", ""]
    lat = agg["latency"]
    out.append("## Latency (s): median, mean per mode")
    for m in MODES:
        if m in lat:
            out.append(f"- {m}: {lat[m]}")
    out.append("")
    out.append("## Text-agreement metrics (token overlap, median across images)")
    out.append(f"- local->VLM recall (share of RapidOCR tokens the VLM also extracted): **{agg['recall_local_in_vlm']}**")
    out.append(f"- VLM extra (share of VLM tokens absent from local OCR): **{agg['extra_vlm_not_local']}**")
    out.append(f"- vision analysis surfaces local-OCR text: **{agg['vision_sees_local_text']}**")
    out.append(f"- RapidOCR mean confidence: **{agg['rapidocr_mean_conf']}**")
    out.append("")
    out.append("## Per image")
    for path, modes in by_image.items():
        name = Path(path).name
        out.append(f"### {name}")
        for m in MODES:
            row = modes.get(m)
            if not row:
                out.append(f"- {m}: MISSING")
                continue
            if not row["ok"]:
                out.append(f"- {m}: ERROR {row['error']}")
                continue
            extra = ""
            if m == "vision" and row.get("model"):
                extra = f" model={row['model']}"
            if m == "rapidocr" and row.get("engine_meta"):
                meta = row["engine_meta"]
                extra = f" items={meta['items']} mean_conf={meta['mean_conf']}"
            if m == "vlm-ocr" and row.get("model"):
                extra = f" model={row['model']}"
            text = (row["text"] or "(no text)").strip()
            shown = text if len(text) <= 400 else text[:400] + "..."
            out.append(f"- **{m}** ({row['latency_s']}s, {row['words']} words{extra}): {shown}")
        out.append("")
    if errors:
        out.append("## Errors")
        for e in errors:
            out.append(f"- {Path(e['path']).name} [{e['mode']}]: {e.get('error')}")
        out.append("")
    (run_dir / "summary.md").write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", default="/tmp", help="directory scanned for screenshots (default: /tmp)")
    ap.add_argument("--glob", default="omp-sshots-*.webp", help="file pattern inside --images-dir")
    ap.add_argument("--images", help="comma-separated explicit paths (overrides dir/glob/limit)")
    ap.add_argument("--limit", type=int, default=0, help="evenly spaced sample of N (0 = all)")
    ap.add_argument("--out", default="benchmarks", help="output root (default: benchmarks)")
    ap.add_argument("--no-resume", action="store_true", help="redo everything")
    args = ap.parse_args()

    if args.images:
        images = [p.strip() for p in args.images.split(",") if p.strip()]
        missing = [p for p in images if not os.path.isfile(p)]
        if missing:
            print(f"error: missing images: {missing}", file=sys.stderr)
            return 2
    else:
        images = select_images(args.images_dir, args.glob, args.limit)
        if not images:
            print("error: no images matched", file=sys.stderr)
            return 2

    cfg = load_config()
    if not cfg.openrouter_api_key:
        print("error: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN not set; hf fallback models will fail", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / f"compare-modes-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = VisionClient(cfg)
    dispatcher = OcrDispatcher(cfg, vision_client=client)
    done = set() if args.no_resume else _load_existing(run_dir)
    todo = [(p, m) for p in images for m in MODES if (p, m) not in done]
    print(f"images: {len(images)}, pending calls: {len(todo)} (done: {len(done)})")

    for i, (path, mode) in enumerate(todo, 1):
        row = run_one(dispatcher, client, path, mode)
        with open_nofollow_append(run_dir / "results.jsonl") as fh:
            fh.write(json.dumps(row) + "\n")
        status = "ok" if row["ok"] else f"ERR {row.get('error', '')[:80]}"
        print(f"[{i}/{len(todo)}] {Path(path).name} {mode}: {status} ({row['latency_s']}s, {row['words']}w)")

    summary = summarize(run_dir)
    write_summary(run_dir, summary)
    print(f"\ncomplete: {run_dir}")
    agg = summary["agg"]
    print(f"recall local->VLM: {agg['recall_local_in_vlm']} | VLM extra: {agg['extra_vlm_not_local']} | "
          f"vision sees local: {agg['vision_sees_local_text']} | conf: {agg['rapidocr_mean_conf']}")
    for m, stats in agg["latency"].items():
        print(f"latency {m}: median {stats[0]}s mean {stats[1]}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
