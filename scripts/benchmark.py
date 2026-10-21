#!/usr/bin/env python3
"""Benchmark CLI: run models x images x reps and summarize.

Run with the venv python and provider tokens in the environment:

    HF_TOKEN=hf__... OPENROUTER_API_KEY=sk-or-... \
        .venv/bin/python scripts/benchmark.py \
        --models google/gemma-4-26B-A4B-it,openrouter:google/gemma-4-26b-a4b-it:free \
        --images-dir /tmp/vdaudit --limit 10 --reps 5

Modes:
  run        execute the benchmark (default)
  summarize  only (re)build summary.json by-model/ analysis_prompt.md
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from look_n_see.benchmark import BenchmarkSpec, run_benchmark, summarize
from look_n_see.client import VisionClient
from look_n_see.config import load_config


def _models(default: str) -> tuple[str, ...]:
    return tuple(m.strip() for m in default.replace(",", " ").split() if m.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma/space separated model ids (default: VISION_MODEL)")
    ap.add_argument("--images-dir", default="/tmp/vdaudit", help="directory of PNG screenshots")
    ap.add_argument("--images", help="comma-separated explicit image paths (overrides --images-dir/--limit)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sorted images (0 = all)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prompt", default="default", help="analysis prompt text or 'default'")
    ap.add_argument("--out", default="benchmarks", help="output root; run dir is created under it")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing rows, redo everything")
    ap.add_argument("--summarize-only", metavar="DIR", help="only rebuild summary files for DIR and exit")
    args = ap.parse_args()

    if args.summarize_only:
        summarize(args.summarize_only)
        print(f"summarized {args.summarize_only}")
        return 0

    cfg = load_config()
    from look_n_see.prompts import SCREENSHOT_DEFAULT

    prompt = SCREENSHOT_DEFAULT if args.prompt == "default" else args.prompt
    models = _models(args.models or cfg.model)
    if "openrouter" in cfg.provider or any(m.startswith("openrouter:") for m in models):
        if not cfg.openrouter_api_key:
            missing_env = "OPENROUTER_API_KEY"
            if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_TOKEN"):
                missing_env = "OPENROUTER_API_KEY (set OPENROUTER_API_KEY, not OPENROUTER_TOKEN)"
            print(f"error: openrouter models requested but {missing_env} is not set", file=sys.stderr)
            return 2
    if any(not m.startswith("openrouter:") for m in models) and not os.environ.get("HF_TOKEN"):
        print("error: HF models requested but HF_TOKEN is not set", file=sys.stderr)
        return 2

    if args.images:
        images = []
        for entry in args.images.replace(",", " ").split():
            entry = entry.strip()
            if not entry:
                continue
            path = Path(entry)
            if not path.is_absolute():
                candidate = Path(args.images_dir) / entry
                if candidate.exists():
                    path = candidate
            images.append(str(path))
        missing = [p for p in images if not Path(p).exists()]
        if missing:
            print(f"error: image(s) not found: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        images = sorted(
            str(p) for p in Path(args.images_dir).iterdir() if p.suffix.lower() == ".png"
        )
        if args.limit:
            images = images[: args.limit]
    if not images:
        print("error: no images selected (check --images-directory/--images)", file=sys.stderr)
        return 2

    spec = BenchmarkSpec(
        models=models,
        images=tuple(images),
        reps=args.reps,
        prompt=prompt,
        out_dir=str(Path(args.out) / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
    )
    client = VisionClient(cfg)
    run_dir = run_benchmark(spec, client, resume=not args.no_resume)
    summary = summarize(run_dir)
    print(f"benchmark complete: {run_dir}")
    for model_id, stats in summary["models"].items():
        lat = stats.get("latency_s") or {}
        median = lat.get("median", "n/a")
        print(
            f"  {model_id}: {stats['ok']}/{stats['calls']} ok, "
            f"median {median}s, flags={stats['flag_counts']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
