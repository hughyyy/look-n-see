"""Server tests: tool registration (incl. role gating), end-to-end with a
mocked client, OCR tool behavior, error paths."""

import asyncio
import unittest

from look_n_see.config import Config
from look_n_see.server import create_server

PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_DATA_URI = f"data:image/png;base64,{PNG_1PX}"

CFG = Config(
    model="test-model",
    endpoint=None,
    max_tokens=64,
    timeout=5.0,
    retries=0,
    retry_backoff=1.0,
)


class FakeClient:
    def __init__(self, result="ok", error=None):
        self.result = result
        self.error = error
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def call(mcp, name, arguments):
    result = asyncio.run(mcp.call_tool(name, arguments))
    return result.content[0].text


def tool_names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


VISION_TOOLS = {"describe_image", "analyze_screenshot", "ask_about_image"}


class RegistrationTests(unittest.TestCase):
    def test_registers_all_tools_by_default(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        self.assertEqual(tool_names(mcp), VISION_TOOLS | {"extract_text"})

    def test_ocr_disabled_registers_vision_only(self):
        mcp = create_server(cfg=Config(**{**CFG.__dict__, "ocr_enabled": False}), client=FakeClient())
        self.assertEqual(tool_names(mcp), VISION_TOOLS)

    def test_vision_disabled_registers_ocr_only(self):
        mcp = create_server(
            cfg=Config(**{**CFG.__dict__, "vision_enabled": False}), client=FakeClient()
        )
        self.assertEqual(tool_names(mcp), {"extract_text"})

    def test_no_tools_when_everything_disabled(self):
        mcp = create_server(
            cfg=Config(**{**CFG.__dict__, "vision_enabled": False, "ocr_enabled": False}),
            client=FakeClient(),
        )
        self.assertEqual(tool_names(mcp), set())


class ToolBehaviorTests(unittest.TestCase):
    def test_describe_image_passes_image_and_default_prompt(self):
        fake = FakeClient(result="A photo of a cat.")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "describe_image", {"image": PNG_DATA_URI})
        self.assertEqual(result, "A photo of a cat.")
        (messages,) = fake.calls
        self.assertEqual(messages[0]["content"][0]["image_url"]["url"], PNG_DATA_URI)
        self.assertIn("Describe this image", messages[0]["content"][1]["text"])

    def test_custom_prompt_steers(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        call(mcp, "analyze_screenshot", {"image": PNG_DATA_URI, "prompt": "Is there a modal?"})
        (messages,) = fake.calls
        self.assertEqual(messages[0]["content"][1]["text"], "Is there a modal?")

    def test_ask_about_image_passes_question(self):
        fake = FakeClient(result="42")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "ask_about_image", {"image": PNG_DATA_URI, "question": "What value?"})
        self.assertEqual(result, "42")
        (messages,) = fake.calls
        self.assertEqual(messages[0]["content"][1]["text"], "What value?")

    def test_model_param_passed_to_client(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        call(
            mcp,
            "analyze_screenshot",
            {
                "image": PNG_DATA_URI,
                "prompt": "What state?",
                "model": "openrouter:google/gemma-4-26b-a4b-it:free",
            },
        )
        self.assertEqual(fake.kwargs[0]["model"], "openrouter:google/gemma-4-26b-a4b-it:free")

    def test_invalid_image_returns_error_text(self):
        fake = FakeClient()
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "describe_image", {"image": "/no/such/file.png"})
        self.assertTrue(result.startswith("error:"))
        self.assertEqual(fake.calls, [])

    def test_inference_failure_returns_error_text(self):
        fake = FakeClient(error=RuntimeError("boom"))
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "describe_image", {"image": PNG_DATA_URI})
        self.assertTrue(result.startswith("error:"))


class OcrToolTests(unittest.TestCase):
    def test_extract_text_uses_vlm_engine_by_default(self):
        fake = FakeClient(result="HELLO WORLD")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "extract_text", {"image": PNG_DATA_URI})
        self.assertEqual(result, '{"text": "HELLO WORLD", "engine": "vlm"}')
        (messages,) = fake.calls
        text = messages[0]["content"][1]["text"]
        self.assertIn("Extract all visible text", text)

    def test_extract_text_custom_prompt_and_model(self):
        fake = FakeClient(result='"42"')
        mcp = create_server(cfg=CFG, client=fake)
        call(
            mcp,
            "extract_text",
            {
                "image": PNG_DATA_URI,
                "prompt": "Just the number",
                "model": "openrouter:qwen/qwen3-vl:free",
            },
        )
        (messages,) = fake.calls
        self.assertEqual(messages[0]["content"][1]["text"], "Just the number")
        self.assertEqual(fake.kwargs[0]["model"], "openrouter:qwen/qwen3-vl:free")

    def test_extract_text_engine_override_unknown(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "bogus"})
        self.assertIn("unknown OCR engine", result)

    def test_extract_text_engine_override_rapidocr_error_when_unavailable(self):
        from unittest import mock

        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", False):
            result = call(mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "rapidocr"})
        self.assertIn("requires the 'ocr' extra", result)

    def test_extract_text_mistral_requires_key(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "mistral"})
        self.assertIn("MISTRAL_API_KEY", result)

    def test_extract_text_invalid_image(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "extract_text", {"image": "/no/such/file.png"})
        self.assertTrue(result.startswith("error:"))


if __name__ == "__main__":
    unittest.main()
