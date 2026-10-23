"""Tests for the standalone harness scripts (previously untested).

The scripts are imported from `scripts/` rather than installed, so this module
puts that directory on `sys.path` before importing them as top-level modules.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark as benchmark_cli  # noqa: E402
import compare_modes  # noqa: E402
import compare_models  # noqa: E402

from look_n_see.config import Config  # noqa: E402


def _config(**kw) -> Config:
    base = dict(
        model="m",
        endpoint=None,
        max_tokens=64,
        timeout=5.0,
        retries=0,
        retry_backoff=1.0,
    )
    base.update(kw)
    return Config(**base)


class SelectImagesTests(unittest.TestCase):
    def _images(self, tmp: str, count: int) -> list[str]:
        paths = []
        for i in range(count):
            path = Path(tmp) / f"img{i:02d}.png"
            path.write_bytes(b"x")
            paths.append(str(path))
        return paths

    def test_sample_includes_first_and_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._images(tmp, 10)
            sample = compare_modes.select_images(tmp, "img*.png", 3)
            self.assertEqual(len(sample), 3)
            self.assertEqual(sample[0], paths[0])
            self.assertEqual(sample[-1], paths[-1])

    def test_limit_one_returns_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._images(tmp, 5)
            self.assertEqual(
                compare_modes.select_images(tmp, "img*.png", 1), [paths[0]]
            )

    def test_zero_and_oversized_limits_return_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._images(tmp, 4)
            self.assertEqual(
                compare_modes.select_images(tmp, "img*.png", 0), paths
            )
            self.assertEqual(
                compare_modes.select_images(tmp, "img*.png", 99), paths
            )

    def test_no_matches_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(compare_modes.select_images(tmp, "*.png", 3), [])


class CredentialPreflightTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_openrouter_routing_needs_openrouter_key(self):
        cfg = _config(provider="openrouter")
        self.assertEqual(
            benchmark_cli.missing_credentials(("q/q",), cfg),
            "OPENROUTER_API_KEY",
        )

    def test_openrouter_token_only_hint(self):
        cfg = _config(provider="openrouter")
        with mock.patch.dict(os.environ, {"OPENROUTER_TOKEN": "x"}):
            self.assertIn(
                "not OPENROUTER_TOKEN",
                benchmark_cli.missing_credentials(("q/q",), cfg),
            )

    def test_hf_routing_needs_hf_token(self):
        cfg = _config()
        self.assertEqual(benchmark_cli.missing_credentials(("q/q",), cfg), "HF_TOKEN")

    def test_openrouter_provider_with_bare_ids_needs_no_hf_token(self):
        cfg = _config(provider="openrouter", openrouter_api_key="sk-test")
        self.assertIsNone(benchmark_cli.missing_credentials(("q/q",), cfg))

    def test_mixed_models_need_both_keys(self):
        cfg = _config(openrouter_api_key="sk-test")
        self.assertEqual(
            benchmark_cli.missing_credentials(("hf:a/b", "openrouter:c/d"), cfg),
            "HF_TOKEN",
        )

    def test_models_parser_accepts_commas_and_spaces(self):
        self.assertEqual(
            benchmark_cli._models("a/b, c/d  e/f"), ("a/b", "c/d", "e/f")
        )


class CompareModesSummaryTests(unittest.TestCase):
    def _row(self, path, mode, text="", ok=True, **kw):
        return {
            "path": path,
            "mode": mode,
            "ok": ok,
            "text": text,
            "latency_s": 1.0,
            "words": compare_modes.word_count(text),
            **kw,
        }

    def test_metrics_and_errors(self):
        rows = [
            self._row("/img/a.png", "vision", "alpha beta gamma"),
            self._row(
                "/img/a.png",
                "vlm-ocr",
                "alpha beta",
            ),
            self._row(
                "/img/a.png",
                "rapidocr",
                "alpha",
                engine_meta={"items": 1, "mean_conf": 0.9},
            ),
            self._row("/img/b.png", "vision", ok=False, error="boom"),
            self._row("/img/b.png", "vlm-ocr", ok=False, error="boom"),
            self._row("/img/b.png", "rapidocr", ok=False, error="boom"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            summary = compare_modes.summarize(run_dir)
        agg = summary["agg"]
        self.assertEqual(agg["rows"], 6)
        self.assertEqual(agg["images"], 2)
        self.assertEqual(agg["recall_local_in_vlm"], 1.0)
        self.assertEqual(agg["extra_vlm_not_local"], 0.5)
        self.assertEqual(agg["vision_sees_local_text"], 1.0)
        self.assertEqual(agg["rapidocr_mean_conf"], 0.9)
        self.assertEqual(len(summary["errors"]), 3)

    def test_summary_markdown_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "results.jsonl").write_text(
                json.dumps(self._row("/img/a.png", "vision", "alpha")) + "\n"
            )
            summary = compare_modes.summarize(run_dir)
            compare_modes.write_summary(run_dir, summary)
            text = (run_dir / "summary.md").read_text()
        self.assertIn("Three-mode comparison", text)
        self.assertIn("alpha", text)


class CompareModelsSummaryTests(unittest.TestCase):
    def _row(self, path, mode, model, text="", ok=True):
        return {
            "path": path,
            "mode": mode,
            "model": model,
            "ok": ok,
            "text": text,
            "latency_s": 1.0,
        }

    def test_consensus_and_vision_stats(self):
        path = "/img/a.png"
        rows = [
            self._row(path, "vision", "m1", "alpha beta"),
            self._row(path, "vision", "m2", "alpha beta"),
            self._row(path, "vlm-ocr", "m1", "alpha beta"),
            self._row(path, "vlm-ocr", "m2", "alpha beta"),
            self._row(path, "rapidocr", "", "alpha"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            summary = compare_models.summarize(
                run_dir, [path], ["m1", "m2"]
            )
        per_image = summary["per_image"][path]
        self.assertEqual(per_image["consensus_size"], 2)
        for model in ("m1", "m2"):
            stats = per_image["models"][model]
            self.assertEqual(stats["consensus_recall"], 1.0)
            self.assertEqual(stats["local_recall"], 1.0)
            self.assertEqual(stats["mean_pairwise_jaccard"], 1.0)
        self.assertEqual(summary["vision"]["m1"]["ok"], 1)


if __name__ == "__main__":
    unittest.main()
