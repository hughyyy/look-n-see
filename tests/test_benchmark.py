"""Unit tests for the benchmark harness: runner, resume, summarize, classifier."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from look_n_see.benchmark import (
    BenchmarkSpec,
    _call_row,
    _quantiles,
    _repetition_rate,
    _slug,
    _version,
    classify,
    run_benchmark,
    summarize,
)
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
        meta = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(meta["models"], ["m1", "m2"])
        self.assertEqual(meta["images"], ["a.png", "b.png", "c.png"])
        self.assertEqual(meta["reps"], 2)
        self.assertEqual(meta["config"]["model"], "m")
        self.assertIn("prompt", meta)

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

    def test_resume_tolerates_corrupt_lines(self):
        run_dir = run_benchmark(self.spec, FakeBenchClient(), progress=False)
        count = len((run_dir / "results.jsonl").read_text().splitlines())
        with open(run_dir / "results.jsonl", "a") as fh:
            fh.write("this is not json\n")
        # Corrupt line must be skipped, never crash the resume.
        run_benchmark(self.spec, FakeBenchClient(), progress=False)
        count2 = len((run_dir / "results.jsonl").read_text().splitlines())
        self.assertEqual(count + 1, count2)

    def test_unreadable_image_skipped_with_warning(self):
        spec = BenchmarkSpec(
            models=("m1",),
            images=tuple([str(self.root / "gone.png")] + self.images[:1]),
            reps=1,
            out_dir=str(self.root / "run"),
        )
        err = io.StringIO()
        with redirect_stderr(err):
            run_dir = run_benchmark(spec, FakeBenchClient(), progress=True)
        log = err.getvalue()
        self.assertIn("warning: skipping unreadable image", log)
        self.assertIn("gone.png", log)
        rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)  # only the readable image ran
        self.assertIn("run:", log)
        self.assertIn("done:", log)

    def test_progress_logs_run_summary(self):
        spec = BenchmarkSpec(
            models=("m1",), images=tuple(self.images[:1]), reps=1,
            out_dir=str(self.root / "run"),
        )
        err = io.StringIO()
        with redirect_stderr(err):
            run_benchmark(spec, FakeBenchClient(), progress=True)
        log = err.getvalue()
        self.assertIn("run:", log)
        self.assertIn("done: 1 new rows", log)

    def test_progress_reports_chunked_counts(self):
        # 3 images x 4 reps = 12 rows -> the every-10-calls progress line fires.
        spec = BenchmarkSpec(
            models=("m1",), images=tuple(self.images), reps=4,
            out_dir=str(self.root / "run"),
        )
        err = io.StringIO()
        with redirect_stderr(err):
            run_benchmark(spec, FakeBenchClient(), progress=True)
        log = err.getvalue()
        self.assertIn("10 new calls recorded", log)
        self.assertIn("done: 12 new rows", log)

    def test_progress_disabled_writes_no_stderr(self):
        err = io.StringIO()
        with redirect_stderr(err):
            run_benchmark(
                BenchmarkSpec(
                    models=("m1",),
                    images=tuple(self.images[:1]),
                    reps=1,
                    out_dir=str(self.root / "quiet"),
                ),
                FakeBenchClient(),
                progress=False,
            )
        self.assertEqual(err.getvalue(), "")

    def test_partial_rows_are_skipped_on_resume(self):
        run_dir = self.root / "partial"
        run_dir.mkdir()
        (run_dir / "results.jsonl").write_text(
            json.dumps({"model": "m1"}) + "\n" + "not json\n"
        )
        # Valid-but-incomplete rows must not crash the resume pass.
        out = run_benchmark(
            BenchmarkSpec(
                models=("m1",),
                images=tuple(self.images[:1]),
                reps=1,
                out_dir=str(run_dir),
            ),
            FakeBenchClient(),
            progress=False,
        )

        lines = (out / "results.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 3)  # partial + garbage + one new row

    def test_zero_reps_writes_metadata_only(self):
        out = run_benchmark(
            BenchmarkSpec(
                models=("m1",),
                images=tuple(self.images[:1]),
                reps=0,
                out_dir=str(self.root / "zero"),
            ),
            FakeBenchClient(),
            progress=False,
        )
        self.assertTrue((out / "run.json").exists())
        self.assertFalse((out / "results.jsonl").exists())

    def test_existing_run_metadata_is_preserved(self):
        run_dir = self.root / "keep"
        run_dir.mkdir()
        original = {"created": "original", "models": ["old"]}
        (run_dir / "run.json").write_text(json.dumps(original))
        run_benchmark(
            BenchmarkSpec(
                models=("m1",),
                images=tuple(self.images[:1]),
                reps=1,
                out_dir=str(run_dir),
            ),
            FakeBenchClient(),
            progress=False,
        )
        self.assertEqual(json.loads((run_dir / "run.json").read_text()), original)

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    def test_symlinked_results_file_is_refused(self):
        run_dir = self.root / "link-results"
        run_dir.mkdir()
        victim = self.root / "victim.txt"
        victim.write_text("keep\n")
        (run_dir / "results.jsonl").symlink_to(victim)
        with self.assertRaises(OSError):
            run_benchmark(
                BenchmarkSpec(
                    models=("m1",),
                    images=tuple(self.images[:1]),
                    reps=1,
                    out_dir=str(run_dir),
                ),
                FakeBenchClient(),
                progress=False,
            )
        self.assertEqual(victim.read_text(), "keep\n")

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    def test_symlinked_run_json_is_not_followed(self):
        run_dir = self.root / "link-meta"
        run_dir.mkdir()
        victim = self.root / "victim.json"
        victim.write_text('{"keep": true}\n')
        (run_dir / "run.json").symlink_to(victim)
        # O_EXCL sees the symlink as an existing run and skips the write; the
        # link itself is never followed, so the target stays untouched.
        run_benchmark(
            BenchmarkSpec(
                models=("m1",),
                images=tuple(self.images[:1]),
                reps=1,
                out_dir=str(run_dir),
            ),
            FakeBenchClient(),
            progress=False,
        )
        self.assertEqual(json.loads(victim.read_text()), {"keep": True})

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    def test_symlinked_by_model_output_is_refused(self):
        run_dir = run_benchmark(self.spec, FakeBenchClient(), progress=False)
        victim = self.root / "victim-model.jsonl"
        victim.write_text("keep\n")
        by_model = run_dir / "by-model"
        by_model.mkdir(exist_ok=True)
        (by_model / "m1.jsonl").symlink_to(victim)
        with self.assertRaises(OSError):
            summarize(run_dir, write_files=True)
        self.assertEqual(victim.read_text(), "keep\n")


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
        self.assertEqual(stats["finish_reasons"], {"stop": 2})
        self.assertEqual(stats["retries_used"], {"0": 2})
        self.assertEqual(stats["repetition"], 1.0)

    def test_writes_by_model_and_prompt(self):
        summarize(self.run_dir, write_files=True)
        by_model = self.run_dir / "by-model"
        self.assertTrue((by_model / "m1.jsonl").exists())
        self.assertTrue((by_model / "m2.jsonl").exists())
        prompt = (self.run_dir / "analysis_prompt.md").read_text()
        self.assertIn("m1", prompt)
        written = json.loads((self.run_dir / "summary.json").read_text())
        self.assertEqual(set(written["models"]), {"m1", "m2"})

    def test_write_files_false_creates_no_outputs(self):
        self.assertFalse((self.run_dir / "by-model").exists())
        self.assertFalse((self.run_dir / "summary.json").exists())
        self.assertFalse((self.run_dir / "analysis_prompt.md").exists())


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

    def test_terse_boundary_is_exclusive(self):
        # terse = non-empty, unflagged, < 60 chars; exactly 60 is normal.
        self.assertIn("terse", classify("x" * 59))
        self.assertEqual(classify("x" * 60), ["normal"])

    def test_combined_flags_are_reported_in_order(self):
        self.assertEqual(
            classify('{"model": "dalle"}'), ["gen_payload", "json_markers"]
        )
        self.assertEqual(classify(""), ["empty"])


class HelperEdgeTests(unittest.TestCase):
    def test_call_row_records_non_inference_errors(self):
        class RaisingClient:
            config = Config(
                model="m", endpoint=None, max_tokens=64, timeout=5.0,
                retries=0, retry_backoff=1.0,
            )

            def complete_with_meta(self, messages, *, model, fallbacks=None, **kw):
                raise ValueError("boom")

        row = _call_row("m1", "a.png", 1, "p", [{"role": "user"}], RaisingClient())
        self.assertFalse(row["ok"])
        self.assertEqual(row["error"], "ValueError: boom")
        self.assertEqual(row["flags"], ["error"])
        self.assertEqual(row["retries"], 0)

    def test_call_row_records_completion_retries(self):
        class RetryingClient:
            config = Config(
                model="m", endpoint=None, max_tokens=64, timeout=5.0,
                retries=2, retry_backoff=0.0,
            )

            def complete_with_meta(self, messages, *, model, fallbacks=None, **kw):
                return Completion(
                    text="ok",
                    model=model,
                    provider="hf",
                    transport="fake",
                    retries=2,
                    latency_s=0.1,
                    finish_reason="stop",
                )

        row = _call_row("m1", "a.png", 1, "p", [], RetryingClient())
        self.assertTrue(row["ok"])
        self.assertEqual(row["retries"], 2)

    def test_repetition_rate_none_without_pairs(self):
        rows = [
            {"model": "m", "image": "a.png", "ok": False, "text": "x"},
            {"model": "m", "image": "a.png", "ok": False, "text": "y"},
        ]
        self.assertIsNone(_repetition_rate(rows))

    def test_repetition_rate_identical_and_mixed(self):
        identical = [
            {"model": "m", "image": "a.png", "ok": True, "text": "same"},
            {"model": "m", "image": "a.png", "ok": True, "text": "same"},
        ]
        self.assertEqual(_repetition_rate(identical), 1.0)
        mixed = [
            {"model": "m", "image": "a.png", "ok": True, "text": "one"},
            {"model": "m", "image": "a.png", "ok": True, "text": "two"},
        ]
        self.assertEqual(_repetition_rate(mixed), 0.0)
        single_rep = [{"model": "m", "image": "a.png", "ok": True, "text": "x"}]
        self.assertEqual(_repetition_rate(single_rep), 0.0)
        self.assertIsNone(_repetition_rate([]))

    def test_classify_refusal_wins_over_terse(self):
        # "I cannot" is both short and a refusal; only the refusal is reported.
        self.assertEqual(classify("I cannot"), ["refusal"])

    def test_repetition_rate_ignores_failed_rows(self):
        rows = [
            {"model": "m", "image": "a.png", "ok": True, "text": "same"},
            {"model": "m", "image": "a.png", "ok": True, "text": "same"},
            {"model": "m", "image": "a.png", "ok": False, "text": "boom"},
        ]
        self.assertEqual(_repetition_rate(rows), 1.0)

    def test_quantiles_single_value(self):
        self.assertEqual(
            _quantiles([7.0]),
            {"min": 7.0, "median": 7.0, "p90": 7.0, "max": 7.0},
        )

    def test_quantiles_two_values(self):
        self.assertEqual(
            _quantiles([9.0, 1.0]),
            {"min": 1.0, "median": 5.0, "p90": 9.0, "max": 9.0},
        )

    def test_version_unknown_package(self):
        self.assertEqual(
            _version("definitely-not-a-real-package-xyz-123"), "?"
        )

    def test_version_for_installed_package(self):
        self.assertNotEqual(_version("httpx"), "?")

    def test_quantiles_multiple_values(self):
        self.assertEqual(
            _quantiles([5.0, 1.0, 3.0]),
            {"min": 1.0, "median": 3.0, "p90": 5.0, "max": 5.0},
        )

    def test_slug_replaces_unsafe_characters(self):
        self.assertEqual(
            _slug("openrouter:google/gemma-4-26b"), "openrouter.google.gemma-4-26b"
        )

    def test_classify_whitespace_only_is_empty(self):
        self.assertEqual(classify("  \n\t"), ["empty"])

    def test_classify_leading_bracket_is_json_markers(self):
        self.assertIn("json_markers", classify("[1, 2, 3]"))


class SummarizeErrorTests(unittest.TestCase):
    def test_failed_row_without_error_bucketed_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"created": "now", "models": ["m"], "images": ["a.png"], "reps": 1})
            )
            (run_dir / "results.jsonl").write_text(
                json.dumps({"model": "m", "image": "a.png", "rep": 1, "ok": False}) + "\n"
            )
            summary = summarize(run_dir, write_files=False)
        self.assertEqual(summary["models"]["m"]["failed"], 1)
        self.assertEqual(summary["models"]["m"]["errors"], {"unknown": 1})
        # Failed-only model: no latency percentiles, no median text length.
        self.assertIsNone(summary["models"]["m"]["latency_s"])
        self.assertIsNone(summary["models"]["m"]["text_len_median"])

    def test_empty_results_file_produces_no_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"models": [], "images": [], "reps": 0})
            )
            (run_dir / "results.jsonl").write_text("")
            summary = summarize(run_dir, write_files=False)
        self.assertEqual(summary["models"], {})


class SummarizeFlaggedTests(unittest.TestCase):
    def test_flagged_samples_collected_and_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"created": "now", "models": ["m"], "images": ["a.png"], "reps": 1})
            )
            long_text = "x" * 500
            rows = [
                json.dumps(
                    {
                        "model": "m",
                        "image": "a.png",
                        "rep": 1,
                        "ok": True,
                        "text": long_text,
                        "latency_s": 1.0,
                        "flags": ["terse"],
                    }
                ),
                json.dumps(
                    {
                        "model": "m",
                        "image": "a.png",
                        "rep": 2,
                        "ok": True,
                        "text": "fine",
                        "latency_s": 1.0,
                    }
                ),
            ]
            (run_dir / "results.jsonl").write_text("\n".join(rows) + "\n")
            summary = summarize(run_dir, write_files=False)
        stats = summary["models"]["m"]
        self.assertEqual(stats["flag_counts"], {"terse": 1})
        (sample,) = stats["flagged_samples"]
        self.assertEqual(sample["image"], "a.png")
        self.assertEqual(sample["flags"], ["terse"])
        self.assertEqual(len(sample["snippet"]), 200)  # truncated
        self.assertNotIn("terse", stats["flag_counts"].get("normal", {}))

    def test_flagged_samples_are_capped_at_twenty(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"created": "now", "models": ["m"], "images": ["a.png"], "reps": 21})
            )
            rows = [
                json.dumps(
                    {
                        "model": "m",
                        "image": "a.png",
                        "rep": rep,
                        "ok": True,
                        "text": "x",
                        "latency_s": 0.1,
                        "flags": ["terse"],
                    }
                )
                for rep in range(1, 22)
            ]
            (run_dir / "results.jsonl").write_text("\n".join(rows) + "\n")
            summary = summarize(run_dir, write_files=False)
        self.assertEqual(len(summary["models"]["m"]["flagged_samples"]), 20)


if __name__ == "__main__":
    unittest.main()
