"""OCR engine tests: Mistral adapter over mocked httpx, dispatcher routing."""

import base64
import httpx
import json
import unittest
from unittest import mock

from look_n_see import ocr as ocr_mod
from look_n_see.config import Config
from look_n_see.images import ImageInputError
from look_n_see.ocr import (
    MistralOcr,
    OcrDispatcher,
    OcrError,
    RapidOcrEngine,
    VlmOcr,
)

CFG = Config(
    model="test-model",
    endpoint=None,
    max_tokens=64,
    timeout=5.0,
    retries=1,
    retry_backoff=0.1,
    mistral_api_key="test-key",
)

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_DATA_URI = f"data:image/png;base64,{PNG_1PX}"


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {
            "pages": [{"markdown": "line one"}, {"markdown": "line two"}],
            "model": "mistral-ocr-latest",
        }
        self.request = mock.MagicMock()

    def json(self):
        return self._json


class FakeClient:
    def __init__(self, result="TEXT", error=None):
        self.result = result
        self.error = error

    def complete(self, messages, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


class MistralOcrTests(unittest.TestCase):
    def test_requires_api_key(self):
        cfg = Config(
            model="m", endpoint=None, max_tokens=64, timeout=5.0, retries=0, retry_backoff=1.0
        )
        with self.assertRaises(OcrError):
            MistralOcr(cfg)

    def test_posts_data_uri_and_returns_markdown(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(engine._http, "post", return_value=_FakeResponse()) as post:
            result = engine.extract(PNG_DATA_URI)
        post.assert_called_once_with(
            "/ocr",
            json={
                "model": "mistral-ocr-latest",
                "document": {"type": "document_url", "document_url": PNG_DATA_URI},
            },
        )
        self.assertEqual(result["text"], "line one\n\nline two")
        self.assertEqual(result["engine"], "mistral")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(
            engine._http.headers["Authorization"], "Bearer test-key"
        )

    def test_missing_pages_returns_empty_text(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http, "post", return_value=_FakeResponse(json_data={"model": "m"})
        ):
            result = engine.extract(PNG_DATA_URI)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["pages"], 0)

    def test_model_override_and_reported_model(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(engine._http, "post", return_value=_FakeResponse()) as post:
            result = engine.extract(PNG_DATA_URI, model="mistral-ocr-custom")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "mistral-ocr-custom")
        self.assertEqual(result["model"], "mistral-ocr-latest")  # echoed by API

    def test_prompt_parameter_is_ignored(self):
        # The prompt parameter only steers the vlm engine; Mistral's payload
        # must stay stable so a prompt can never leak into the request.
        engine = MistralOcr(CFG)
        with mock.patch.object(engine._http, "post", return_value=_FakeResponse()) as post:
            engine.extract(PNG_DATA_URI, prompt="just the numbers")
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("prompt", payload)

    def test_retries_on_429_then_succeeds(self):
        engine = MistralOcr(CFG)  # retries=1
        error = _FakeResponse(status_code=429, json_data={"error": {"message": "slow down"}})
        with mock.patch.object(engine._http, "post", side_effect=[error, _FakeResponse()]) as post:
            result = engine.extract(PNG_DATA_URI)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["pages"], 2)

    def test_exhausts_retries_and_raises(self):
        engine = MistralOcr(CFG)  # retries=1
        error = _FakeResponse(status_code=503, json_data={"error": {"message": "down"}})
        with mock.patch.object(engine._http, "post", side_effect=[error, error]):
            with self.assertRaises(OcrError) as ctx:
                engine.extract(PNG_DATA_URI)
        self.assertIn("503", str(ctx.exception))

    def test_non_retryable_status_stops_immediately(self):
        # 401 is not in the retryable set: one attempt, then raise.
        engine = MistralOcr(CFG)  # retries=1
        error = _FakeResponse(status_code=401, json_data={"error": {"message": "bad key"}})
        with mock.patch.object(engine._http, "post", return_value=error) as post:
            with self.assertRaises(OcrError) as ctx:
                engine.extract(PNG_DATA_URI)
        self.assertEqual(post.call_count, 1)
        self.assertIn("bad key", str(ctx.exception))

    def test_non_object_body_raises_clear_error(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http, "post", return_value=_FakeResponse(json_data=[1, 2])
        ):
            with self.assertRaisesRegex(OcrError, "expected a JSON object"):
                engine.extract(PNG_DATA_URI)

    def test_pages_not_a_list_raises(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http,
            "post",
            return_value=_FakeResponse(json_data={"pages": "oops"}),
        ):
            with self.assertRaisesRegex(OcrError, "'pages' is not a list"):
                engine.extract(PNG_DATA_URI)

    def test_page_entry_not_object_raises(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http,
            "post",
            return_value=_FakeResponse(
                json_data={"pages": [{"markdown": "ok"}, "junk"]}
            ),
        ):
            with self.assertRaisesRegex(OcrError, "page entry is not an object"):
                engine.extract(PNG_DATA_URI)

    def test_page_without_markdown_is_empty(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http,
            "post",
            return_value=_FakeResponse(
                json_data={"pages": [{"index": 0}], "model": "m"}
            ),
        ):
            result = engine.extract(PNG_DATA_URI)
        self.assertEqual(result["text"], "")

    def test_retry_backoff_is_exponential(self):
        cfg = Config(
            model="m",
            endpoint=None,
            max_tokens=64,
            timeout=5.0,
            retries=2,
            retry_backoff=0.25,
            mistral_api_key="k",
        )
        engine = MistralOcr(cfg)
        error = _FakeResponse(status_code=429, json_data={"error": {"message": "slow"}})
        with mock.patch.object(engine._http, "post", side_effect=[error, error, error]):
            with mock.patch("look_n_see.ocr.time.sleep") as sleep:
                with self.assertRaises(OcrError):
                    engine.extract(PNG_DATA_URI)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.25, 0.5])

    def test_negative_retries_still_attempts_once(self):
        cfg = Config(
            model="m",
            endpoint=None,
            max_tokens=64,
            timeout=5.0,
            retries=-1,
            retry_backoff=0.0,
            mistral_api_key="k",
        )
        engine = MistralOcr(cfg)
        with mock.patch.object(engine._http, "post", return_value=_FakeResponse()) as post:
            result = engine.extract(PNG_DATA_URI)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["pages"], 2)

    def test_model_falls_back_when_api_omits_it(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http,
            "post",
            return_value=_FakeResponse(json_data={"pages": []}),
        ):
            result = engine.extract(PNG_DATA_URI, model="mistral-ocr-custom")
        self.assertEqual(result["model"], "mistral-ocr-custom")

    def test_connection_error_retried_then_succeeds(self):
        engine = MistralOcr(CFG)  # retries=1
        with mock.patch.object(
            engine._http,
            "post",
            side_effect=[httpx.ConnectError("down"), _FakeResponse()],
        ) as post:
            result = engine.extract(PNG_DATA_URI)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["pages"], 2)

    def test_null_pages_is_empty(self):
        engine = MistralOcr(CFG)
        with mock.patch.object(
            engine._http,
            "post",
            return_value=_FakeResponse(json_data={"pages": None}),
        ):
            result = engine.extract(PNG_DATA_URI)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["pages"], 0)


class VlmOcrTests(unittest.TestCase):
    def test_uses_ocr_prompt(self):
        class FakeClient:
            def __init__(self):
                self.seen = None

            def complete(self, messages, **kw):
                self.seen = (messages, kw)
                return "TEXT"

        client = FakeClient()
        result = VlmOcr(client).extract(PNG_DATA_URI)
        self.assertEqual(result, {"text": "TEXT", "engine": "vlm"})
        messages, kwargs = client.seen
        self.assertIn("Extract all visible text", messages[0]["content"][1]["text"])
        self.assertEqual(kwargs, {"model": None})

    def test_custom_prompt_and_model_override(self):
        class FakeClient:
            def __init__(self):
                self.seen = None

            def complete(self, messages, **kw):
                self.seen = (messages, kw)
                return "TEXT"

        client = FakeClient()
        VlmOcr(client).extract(
            PNG_DATA_URI, prompt="Just the price", model="openrouter:qwen/qwen3-vl:free"
        )
        messages, kwargs = client.seen
        self.assertEqual(messages[0]["content"][1]["text"], "Just the price")
        self.assertEqual(kwargs, {"model": "openrouter:qwen/qwen3-vl:free"})

    def test_empty_text_still_returns_envelope(self):
        class EmptyClient:
            def complete(self, messages, **kw):
                return ""

        self.assertEqual(
            VlmOcr(EmptyClient()).extract(PNG_DATA_URI),
            {"text": "", "engine": "vlm"},
        )


class RapidOcrEngineTests(unittest.TestCase):
    def test_unavailable_raises_install_hint(self):
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", False):
            with self.assertRaisesRegex(OcrError, "ocr' extra"):
                RapidOcrEngine()

    def _run_engine(self, raw_result):
        """Run RapidOcrEngine.extract with cv2/np/RapidOCR fully mocked."""
        engine_instance = mock.Mock(return_value=raw_result)  # callable result
        factory = mock.Mock(return_value=engine_instance)
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", True), mock.patch(
            "look_n_see.ocr.RapidOCR", factory
        ), mock.patch("look_n_see.ocr.cv2") as cv2, mock.patch(
            "look_n_see.ocr.np"
        ) as np:
            cv2.imdecode.return_value = "decoded-array"
            engine = RapidOcrEngine()
            result = engine.extract(PNG_DATA_URI)
        factory.assert_called_once_with()
        engine_instance.assert_called_once_with("decoded-array")
        np.frombuffer.assert_called_once()
        cv2.imdecode.assert_called_once()
        return result

    def test_tuple_shape_result(self):
        # rapidocr-onnxruntime 1.2.x returns (result, elapse).
        raw = ([("box1", "hello world", 0.99), ("box2", "second line", 0.5)], 0.35)
        result = self._run_engine(raw)
        self.assertEqual(result["engine"], "rapidocr")
        self.assertEqual(result["text"], "hello world\nsecond line")
        self.assertEqual(result["items"][0], {"text": "hello world", "confidence": 0.99, "box": "box1"})
        self.assertEqual(result["items"][1]["confidence"], 0.5)

    def test_list_shape_result(self):
        # 1.4.x returns the bare list.
        raw = [("box", "only line", 0.9)]
        result = self._run_engine(raw)
        self.assertEqual(result["text"], "only line")
        self.assertEqual(len(result["items"]), 1)

    def test_list_of_lists_result(self):
        result = self._run_engine([["box", "line", 0.8]])
        self.assertEqual(result["text"], "line")
        self.assertEqual(result["items"][0]["confidence"], 0.8)

    def test_empty_result(self):
        result = self._run_engine(None)
        self.assertEqual(result["text"], "")
        self.assertEqual(result["items"], [])

    def test_undecodable_image_raises(self):
        engine_instance = mock.Mock(return_value=[])
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", True), mock.patch(
            "look_n_see.ocr.RapidOCR", mock.Mock(return_value=engine_instance)
        ), mock.patch("look_n_see.ocr.cv2") as cv2, mock.patch("look_n_see.ocr.np"):
            cv2.imdecode.return_value = None  # corrupted image -> decode failure
            with self.assertRaisesRegex(OcrError, "could not be decoded"):
                RapidOcrEngine().extract(PNG_DATA_URI)
        engine_instance.assert_not_called()

    def test_data_uri_helper(self):
        # _data_uri_to_bytes must resolve through the shared image loader.
        self.assertEqual(
            ocr_mod._data_uri_to_bytes(PNG_DATA_URI),
            base64.b64decode(PNG_1PX),
        )

    def test_data_uri_helper_invalid_raises_image_input_error(self):
        with self.assertRaises(ImageInputError):
            ocr_mod._data_uri_to_bytes("data:image/png;base64,@@@")

    def test_import_fallback_degrades_gracefully(self):
        # Simulate a venv without the 'ocr' extra in an isolated subprocess:
        # the rapidocr import fails and the module degrades to
        # _RAPIDOCR_AVAILABLE=False. (The branch is pragma: no cover because
        # CI-without-extra exists as an environment; this exercises it
        # everywhere without mutating the shared module in-process.)
        import subprocess
        import sys

        code = (
            "import builtins\n"
            "real = builtins.__import__\n"
            "def blocked(name, *args, **kwargs):\n"
            "    if name == 'rapidocr_onnxruntime':\n"
            "        raise ImportError('simulated missing ocr extra')\n"
            "    return real(name, *args, **kwargs)\n"
            "builtins.__import__ = blocked\n"
            "from look_n_see import ocr\n"
            "assert ocr._RAPIDOCR_AVAILABLE is False\n"
            "assert ocr.RapidOCR is None and ocr.cv2 is None and ocr.np is None\n"
            "print('fallback-ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("fallback-ok", proc.stdout)

    def test_tuple_with_none_result_is_empty(self):
        result = self._run_engine((None, 0.35))
        self.assertEqual(result["text"], "")
        self.assertEqual(result["items"], [])


@unittest.skipUnless(ocr_mod._RAPIDOCR_AVAILABLE, "ocr extra not installed")
class LocalRapidOcrIntegrationTests(unittest.TestCase):
    """Exercise the real ONNX engine (no mocks) on a valid blank PNG.

    Skipped when the optional `ocr` extra is absent, so the default zero-dep
    test run is unaffected; when present this covers cv2 decoding and the
    real rapidocr result shapes end-to-end.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = RapidOcrEngine()

    def test_blank_image_yields_empty_result(self):
        result = self.engine.extract(PNG_DATA_URI)
        self.assertEqual(result["engine"], "rapidocr")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["items"], [])

    def test_dispatcher_routes_to_real_engine(self):
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        payload = json.loads(dispatcher.extract(PNG_DATA_URI, engine="rapidocr"))
        self.assertEqual(payload["engine"], "rapidocr")
        self.assertEqual(payload["text"], "")


class DispatcherTests(unittest.TestCase):
    def test_default_engine_from_config(self):
        class FakeClient:
            def complete(self, messages, **kw):
                return "TEXT"

        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        result = dispatcher.extract(PNG_DATA_URI)
        self.assertIn('"engine": "vlm"', result)
        self.assertIn("TEXT", result)

    def test_engine_override(self):
        class FakeClient:
            def complete(self, messages, **kw):
                return "TEXT"

        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        result = dispatcher.extract(PNG_DATA_URI, engine="VLM")
        self.assertIn('"engine": "vlm"', result)  # normalized lowercase

    def test_unknown_engine_raises(self):
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        with self.assertRaisesRegex(OcrError, "unknown OCR engine"):
            dispatcher.extract(PNG_DATA_URI, engine="bogus")

    def test_engines_cached(self):
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        dispatcher._engine("vlm")
        dispatcher._engine("vlm")
        self.assertEqual(len(dispatcher._engines), 1)

    def test_routes_to_rapidocr_and_caches(self):
        class FakeClient:
            def complete(self, messages, **kw):
                return "TEXT"

        engine_instance = mock.Mock()
        engine_instance.extract.return_value = {"text": "LOCAL", "engine": "rapidocr"}
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        with mock.patch("look_n_see.ocr.RapidOcrEngine", return_value=engine_instance):
            first = dispatcher._engine("rapidocr")
            second = dispatcher._engine("RAPIDOCR")
        self.assertIs(first, second)
        self.assertIs(dispatcher._engines["rapidocr"], engine_instance)

    def test_routes_to_mistral_with_key_and_caches(self):
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())  # CFG has a key
        engine = dispatcher._engine("mistral")
        self.assertIsInstance(engine, MistralOcr)
        self.assertIs(dispatcher._engine("mistral"), engine)

    def test_mistral_without_key_raises_via_dispatcher(self):
        cfg = Config(
            model="m", endpoint=None, max_tokens=64, timeout=5.0,
            retries=0, retry_backoff=1.0, mistral_api_key=None,
        )
        dispatcher = OcrDispatcher(cfg, vision_client=FakeClient())
        with self.assertRaisesRegex(OcrError, "MISTRAL_API_KEY"):
            dispatcher._engine("mistral")

    def test_engine_name_whitespace_and_case(self):
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        self.assertIsInstance(dispatcher._engine("  MiStRaL "), MistralOcr)

    def test_non_ascii_text_is_not_escaped(self):
        text = "héllo ☃"
        result = OcrDispatcher(CFG, vision_client=FakeClient(result=text)).extract(
            PNG_DATA_URI
        )
        self.assertIn(text, result)
        self.assertNotIn("\\u", result)

    def test_default_vision_client_constructed(self):
        with mock.patch("look_n_see.ocr.VisionClient") as vision_client:
            OcrDispatcher(CFG)
        vision_client.assert_called_once_with(CFG)

    def test_unknown_engine_lists_choices(self):
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        with self.assertRaisesRegex(OcrError, "vlm, rapidocr, mistral"):
            dispatcher.extract(PNG_DATA_URI, engine="bogus")

    def test_extract_returns_json_envelope(self):
        result = OcrDispatcher(CFG, vision_client=FakeClient()).extract(PNG_DATA_URI)
        self.assertEqual(json.loads(result), {"text": "TEXT", "engine": "vlm"})

    def test_config_engine_used_when_override_absent(self):
        rapid_cfg = Config(**{**CFG.__dict__, "ocr_engine": "rapidocr"})
        rapid = OcrDispatcher(rapid_cfg, vision_client=FakeClient())
        with mock.patch("look_n_see.ocr.RapidOcrEngine") as rapid_engine:
            rapid._engine(None)
        rapid_engine.assert_called_once_with()

        mistral_cfg = Config(**{**CFG.__dict__, "ocr_engine": "mistral"})
        mistral = OcrDispatcher(mistral_cfg, vision_client=FakeClient())
        with mock.patch("look_n_see.ocr.MistralOcr") as mistral_engine:
            mistral._engine(None)
        mistral_engine.assert_called_once_with(mistral_cfg)

    def test_extract_forwards_prompt_and_model_to_engine(self):
        engine = mock.Mock()
        engine.extract.return_value = {"text": "T", "engine": "vlm"}
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        with mock.patch.object(dispatcher, "_engine", return_value=engine):
            dispatcher.extract(PNG_DATA_URI, prompt="p", model="m", engine="vlm")
        engine.extract.assert_called_once_with(PNG_DATA_URI, prompt="p", model="m")

    def test_failed_engine_construction_is_not_cached(self):
        cfg = Config(
            model="m",
            endpoint=None,
            max_tokens=64,
            timeout=5.0,
            retries=0,
            retry_backoff=1.0,
            mistral_api_key=None,
        )
        dispatcher = OcrDispatcher(cfg, vision_client=FakeClient())
        for _ in range(2):
            with self.assertRaises(OcrError):
                dispatcher._engine("mistral")
        self.assertEqual(dispatcher._engines, {})

    def test_empty_engine_override_uses_config_engine(self):
        cfg = Config(**{**CFG.__dict__, "ocr_engine": "mistral"})
        dispatcher = OcrDispatcher(cfg, vision_client=FakeClient())
        with mock.patch("look_n_see.ocr.MistralOcr") as mistral_engine:
            dispatcher._engine("")
        mistral_engine.assert_called_once_with(cfg)

    def test_engine_error_propagates_unchanged(self):
        engine = mock.Mock()
        engine.extract.side_effect = OcrError("engine exploded")
        dispatcher = OcrDispatcher(CFG, vision_client=FakeClient())
        with mock.patch.object(dispatcher, "_engine", return_value=engine):
            with self.assertRaisesRegex(OcrError, "engine exploded"):
                dispatcher.extract(PNG_DATA_URI)
