"""Unit tests for the benchmark harness: runner, resume, summarize, classifier."""

import json
import tempfile
import unittest
from pathlib import Path

from look_n_see.benchmark import BenchmarkSpec, classify, run_benchmark, summarize
from look_n_see.client import Completion, InferenceError
from look_n_see.config import Config


class FakeBenchClient:
    """Deterministic client: text = '<model>|<image>|<rep>', config like VisionClient."""

    def __init__(self, cfg=None, fail_models=()):
        self.config = cfg or Config(
            model="m", endpoint=None, max_tokens=64, timeout=5.0, retries=0, retry_backoff=1.0
        )
        self.fail_models = fail_models

    def complete_with_meta(self, messages, *, model, fallbacks=None, **kw):
        image_uri = messages[0]["content"][0]["image_url"]["url"]
        image = image_uri.rsplit("/", 1)[-1]
        if model in self.fail_models:
            raise InferenceError(f"boom {model}")
        return Completion(
            text=f"{model}|{image}|{kw.get('rep', '?')}",
            model=model,
            provider="hf",
            transport="FakeBenchClient",
            retries=0,
            latency_s=0.5,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )


def _write_images(tmp: Path, names: list[str]) -> list[str]:
    out = []
    for name in names:
        path = tmp / name
        path.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
        out.append(str(path))
    return out


class RunAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = _write_images(self.root, ["a.png", "b.png", "c.png"])
        self.spec = BenchmarkSpec(
            models=("m1", "m2"),
            images=tuple(self.images),
            reps=2,
            out_dir=str(self.root / "run"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_full_matrix(self):
        run_dir = run_benchmark(self.spec, FakeBenchClient(), progress=False)
        rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2 * 3 * 2)
        self.assertTrue(all(r["ok"] for r in rows))
        self.assertTrue(all(not r.get("error") for r in rows))
        self.assertTrue((run_dir / "run.json").exists())

    def test_resume_skips_existing(self):
        run_dir = run_benchmark(self.spec, FakeBenchClient(), progress=False)
        count = len((run_dir / "results.jsonl").read_text().splitlines())
        run_benchmark(self.spec, FakeBenchClient(), progress=False)
        count2 = len((run_dir / "results.jsonl").read_text().splitlines())
        self.assertEqual(count, count2)

    def test_failures_recorded_with_flags(self):
        spec = BenchmarkSpec(
            models=("bad",), images=tuple(self.images), reps=1,
            out_dir=str(self.root / "run"),
        )
        run_dir = run_benchmark(spec, FakeBenchClient(fail_models=("bad",)), progress=False)
        rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
        self.assertTrue(all(not r["ok"] for r in rows))
        self.assertTrue(all(r["flags"] == ["error"] for r in rows))

    def test_fallback_disabled_in_rows(self):
        # assert the client receives fallbacks=() so rows attribute the model
        class RecordingClient(FakeBenchClient):
            def __init__(self):
                super().__init__()
                self.fallbacks_seen = []

            def complete_with_meta(self, messages, *, model, fallbacks=None, **kw):
                self.fallbacks_seen.append(fallbacks)
                return super().complete_with_meta(messages, model=model, fallbacks=fallbacks, **kw)

        client = RecordingClient()
        spec = BenchmarkSpec(
            models=("m1",), images=tuple(self.images[:1]), reps=1,
            out_dir=str(self.root / "run"),
        )
        run_benchmark(spec, client, progress=False)
        self.assertTrue(all(f == () for f in client.fallbacks_seen))


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        images = _write_images(self.root, ["a.png"])
        spec = BenchmarkSpec(
            models=("m1", "m2"), images=tuple(images), reps=2,
            out_dir=str(self.root / "run"),
        )
        self.run_dir = run_benchmark(spec, FakeBenchClient(), progress=False)
        self.summary = summarize(self.run_dir, write_files=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_summary_shape(self):
        self.assertEqual(set(self.summary["models"]), {"m1", "m2"})
        stats = self.summary["models"]["m1"]
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["ok"], 2)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["latency_s"]["median"], 0.5)
        self.assertEqual(stats["usage_totals"]["completion_tokens"], 40)
        self.assertIn("flag_counts", stats)

    def test_writes_by_model_and_prompt(self):
        summarize(self.run_dir, write_files=True)
        by_model = self.run_dir / "by-model"
        self.assertTrue((by_model / "m1.jsonl").exists())
        self.assertTrue((by_model / "m2.jsonl").exists())
        prompt = (self.run_dir / "analysis_prompt.md").read_text()
        self.assertIn("m1", prompt)
        self.assertIn("m2", prompt)


class ClassifierTests(unittest.TestCase):
    def test_flags(self):
        self.assertIn("refusal", classify("I am a text-based AI and cannot fulfill this"))
        self.assertIn("gen_payload", classify('{"model": "dalle.text2im"}'))
        self.assertIn("json_markers", classify('{"point": [1, 2], "label": "x"}'))
        self.assertIn("terse", classify("NONE"))
        self.assertIn("empty", classify(""))
        self.assertEqual(
            classify("A dark mask over a pane with a Wait message chip, visible in the screenshot."),
            ["normal"],
        )


if __name__ == "__main__":
    unittest.main()
