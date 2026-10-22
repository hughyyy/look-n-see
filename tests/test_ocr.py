"""OCR engine tests: Mistral adapter over mocked httpx, dispatcher routing."""

import unittest
from unittest import mock

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


class RapidOcrEngineTests(unittest.TestCase):
    def test_unavailable_raises_install_hint(self):
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", False):
            with self.assertRaisesRegex(OcrError, "ocr' extra"):
                RapidOcrEngine()


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
