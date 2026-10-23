"""Server tests: tool registration (incl. role gating), end-to-end with a
mocked client, OCR tool behavior, error paths."""

import asyncio
import http.server
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.mcpserver.exceptions import ToolError

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
    # Tool-level URL tests use a loopback image server.
    allow_private_urls=True,
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


def call_failure(mcp, name, arguments):
    """Invoke a tool expected to fail and return its structured ToolError."""
    try:
        asyncio.run(mcp.call_tool(name, arguments))
    except ToolError as exc:
        return exc
    raise AssertionError(f"{name} was expected to raise ToolError")


def error_payload(exc: ToolError) -> dict:
    """Parse the JSON envelope out of a ToolError message.

    MCPServer prefixes the tool name ("Error executing tool X: {...}"), so the
    envelope starts at the first brace.
    """
    text = str(exc)
    return json.loads(text[text.index("{"):])


def tool_names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


VISION_TOOLS = {"describe_image", "analyze_screenshot", "ask_about_image"}
HEALTH_TOOL = {"health"}


REPO_ROOT = Path(__file__).resolve().parents[1]


async def _stdio_probe(overrides: dict[str, str]) -> tuple[set[str], bool, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VISION_", "OCR_", "OPENROUTER_", "HF_", "MISTRAL_"))
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "look_n_see"],
        env={
            **env,
            "VISION_ENABLED": "1",
            "OCR_ENABLED": "1",
            **overrides,
        },
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            # Exercise one tool end-to-end; an unreadable path fails offline
            # before any provider call. The health-only variant needs no args.
            if "describe_image" in names:
                tool, arguments = "describe_image", {"image": "/no/such/file.png"}
            elif "extract_text" in names:
                tool, arguments = "extract_text", {"image": "/no/such/file.png"}
            else:
                tool, arguments = "health", {}
            result = await session.call_tool(tool, arguments)
            return names, result.is_error, result.content[0].text


def stdio_surfaces() -> dict[str, tuple[set[str], bool, str]]:
    """Spawn one server per release variant (concurrently) over stdio."""

    async def _run() -> dict[str, tuple[set[str], bool, str]]:
        both, vision, ocr, none = await asyncio.gather(
            _stdio_probe({}),
            _stdio_probe({"OCR_ENABLED": "0"}),
            _stdio_probe({"VISION_ENABLED": "0"}),
            _stdio_probe({"VISION_ENABLED": "0", "OCR_ENABLED": "0"}),
        )
        return {"both": both, "vision": vision, "ocr": ocr, "none": none}

    return asyncio.run(_run())


class RegistrationTests(unittest.TestCase):
    def test_registers_all_tools_by_default(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        self.assertEqual(tool_names(mcp), VISION_TOOLS | {"extract_text", "health"})

    def test_ocr_disabled_registers_vision_only(self):
        mcp = create_server(cfg=Config(**{**CFG.__dict__, "ocr_enabled": False}), client=FakeClient())
        self.assertEqual(tool_names(mcp), VISION_TOOLS | HEALTH_TOOL)

    def test_vision_disabled_registers_ocr_only(self):
        mcp = create_server(
            cfg=Config(**{**CFG.__dict__, "vision_enabled": False}), client=FakeClient()
        )
        self.assertEqual(tool_names(mcp), {"extract_text"} | HEALTH_TOOL)

    def test_health_only_when_everything_disabled(self):
        mcp = create_server(
            cfg=Config(**{**CFG.__dict__, "vision_enabled": False, "ocr_enabled": False}),
            client=FakeClient(),
        )
        self.assertEqual(tool_names(mcp), HEALTH_TOOL)

    def test_defaults_construct_from_env_and_shared_client(self):
        # The no-argument path (as used by main()) loads env config and
        # builds its own VisionClient — one client shared by vision + OCR.
        with mock.patch("look_n_see.server.config_mod.load_config", return_value=CFG), mock.patch(
            "look_n_see.server.VisionClient", return_value=FakeClient(result="ok")
        ) as vc:
            mcp = create_server()
        self.assertEqual(tool_names(mcp), VISION_TOOLS | {"extract_text", "health"})
        vc.assert_called_once_with(CFG)

    def test_main_runs_stdio_server(self):
        mcp = mock.MagicMock()
        with mock.patch("look_n_see.server.create_server", return_value=mcp) as create:
            from look_n_see import server as server_mod
            server_mod.main()
        create.assert_called_once_with()
        mcp.run.assert_called_once_with(transport="stdio")

    def test_disabled_tool_is_unknown_to_the_server(self):
        mcp = create_server(
            cfg=Config(**{**CFG.__dict__, "vision_enabled": False}), client=FakeClient()
        )
        with self.assertRaises(ToolError) as ctx:
            asyncio.run(mcp.call_tool("describe_image", {"image": PNG_DATA_URI}))
        self.assertIn("Unknown tool", str(ctx.exception))

    def test_missing_arguments_are_rejected(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        cases = {
            "describe_image": {},
            "ask_about_image": {"image": PNG_DATA_URI},
            "extract_text": {},
        }
        for name, arguments in cases.items():
            with self.subTest(tool=name):
                with self.assertRaises(ToolError):
                    asyncio.run(mcp.call_tool(name, arguments))

    def test_wrong_argument_type_is_rejected(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        with self.assertRaises(ToolError):
            asyncio.run(mcp.call_tool("describe_image", {"image": 123}))

    def test_client_built_when_not_injected(self):
        with mock.patch("look_n_see.server.VisionClient") as vision_client:
            create_server(cfg=CFG)
        vision_client.assert_called_once_with(CFG)


class LifecycleTests(unittest.TestCase):
    def test_lifespan_closes_owned_resources(self):
        from look_n_see import server as server_mod

        ocr = mock.Mock()
        client = mock.Mock()
        lifespan = server_mod._make_lifespan(ocr, client, owns_client=True)

        async def run():
            async with lifespan(None):
                pass

        asyncio.run(run())
        ocr.close.assert_called_once()
        client.close.assert_called_once()

    def test_lifespan_leaves_injected_client_open(self):
        from look_n_see import server as server_mod

        ocr = mock.Mock()
        client = mock.Mock()
        lifespan = server_mod._make_lifespan(ocr, client, owns_client=False)

        async def run():
            async with lifespan(None):
                pass

        asyncio.run(run())
        ocr.close.assert_called_once()
        client.close.assert_not_called()


class StdioHandshakeTests(unittest.TestCase):
    """End-to-end registration over the real stdio transport: a spawned
    server process must advertise exactly the variant's tool surface."""

    def test_variants_advertise_surface_and_serve_tools(self):
        expected = {
            "both": VISION_TOOLS | {"extract_text", "health"},
            "vision": VISION_TOOLS | HEALTH_TOOL,
            "ocr": {"extract_text", "health"},
            "none": HEALTH_TOOL,
        }
        for variant, (names, is_error, text) in stdio_surfaces().items():
            with self.subTest(variant=variant):
                self.assertEqual(names, expected[variant])
                if names & (VISION_TOOLS | {"extract_text"}):
                    # Unreadable input fails offline, before any provider call,
                    # and must be reported as a structured MCP tool error.
                    self.assertTrue(is_error, text)
                    payload = json.loads(text[text.index("{"):])
                    self.assertEqual(payload["error"]["type"], "input")
                    self.assertIn("image input", payload["error"]["message"])
                else:
                    self.assertFalse(is_error)
                    payload = json.loads(text)
                    self.assertTrue(payload["ok"])
                    self.assertFalse(payload["vision_enabled"])
                    self.assertFalse(payload["ocr_enabled"])


class ToolSchemaTests(unittest.TestCase):
    """The advertised input schemas are the calling model's contract."""

    def setUp(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        self.tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

    def test_required_arguments(self):
        self.assertEqual(self.tools["describe_image"].input_schema["required"], ["image"])
        self.assertEqual(
            set(self.tools["ask_about_image"].input_schema["required"]),
            {"image", "question"},
        )
        self.assertEqual(self.tools["extract_text"].input_schema["required"], ["image"])

    def test_optional_arguments(self):
        self.assertEqual(
            set(self.tools["describe_image"].input_schema["properties"]),
            {"image", "prompt", "model"},
        )
        self.assertEqual(
            set(self.tools["extract_text"].input_schema["properties"]),
            {"image", "prompt", "model", "engine"},
        )
        self.assertNotIn(
            "question", self.tools["describe_image"].input_schema["properties"]
        )

    def test_every_tool_returns_a_string_result(self):
        for name, tool in self.tools.items():
            with self.subTest(name=name):
                self.assertEqual(
                    tool.output_schema["properties"]["result"]["type"], "string"
                )


class DefaultPromptTests(unittest.TestCase):
    @staticmethod
    def _prompt_for(tool: str, arguments: dict) -> str:
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        call(mcp, tool, {"image": PNG_DATA_URI, **arguments})
        return fake.calls[0][0]["content"][1]["text"]

    def test_analyze_screenshot_default_prompt(self):
        self.assertIn(
            "Analyze this screenshot", self._prompt_for("analyze_screenshot", {})
        )

    def test_blank_prompt_uses_tool_default(self):
        self.assertIn(
            "Describe this image", self._prompt_for("describe_image", {"prompt": "   "})
        )


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

    def test_invalid_image_returns_structured_error(self):
        fake = FakeClient()
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(mcp, "describe_image", {"image": "/no/such/file.png"})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "input")
        self.assertEqual(payload["error"]["exception"], "ImageInputError")
        self.assertIn("image input", payload["error"]["message"])
        self.assertEqual(fake.calls, [])

    def test_inference_failure_returns_structured_error(self):
        fake = FakeClient(error=RuntimeError("boom"))
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(mcp, "describe_image", {"image": PNG_DATA_URI})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "inference")
        self.assertEqual(payload["error"]["message"], "boom")

    def test_empty_image_string_returns_structured_error(self):
        fake = FakeClient()
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(mcp, "describe_image", {"image": "  "})
        self.assertEqual(error_payload(error)["error"]["type"], "input")
        self.assertEqual(fake.calls, [])

    def test_unexpected_input_error_surfaces_as_structured_error(self):
        # A non-ImageInputError from conversion is still a structured failure,
        # tagged `internal`, not a bare "error:" string.
        fake = FakeClient()
        mcp = create_server(cfg=CFG, client=fake)
        with mock.patch("look_n_see.server.to_data_uri", side_effect=ValueError("boom")):
            error = call_failure(mcp, "describe_image", {"image": PNG_DATA_URI})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "internal")
        self.assertIn("boom", payload["error"]["message"])
        self.assertEqual(fake.calls, [])

    def test_refusal_response_is_annotated(self):
        fake = FakeClient(result="I am a text-based AI and cannot fulfill this request.")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "describe_image", {"image": PNG_DATA_URI})
        self.assertIn("I am a text-based AI", result)
        self.assertIn("[look-n-see warning: response flagged as refusal", result)

    def test_empty_response_is_annotated(self):
        fake = FakeClient(result="")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "describe_image", {"image": PNG_DATA_URI})
        self.assertIn("[look-n-see warning: response flagged as empty", result)

    def test_normal_response_is_not_annotated(self):
        fake = FakeClient(result="A dark mask covers the pane with a Wait message chip.")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(mcp, "describe_image", {"image": PNG_DATA_URI})
        self.assertEqual(result, "A dark mask covers the pane with a Wait message chip.")

    def test_health_reports_roles_without_network(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        payload = json.loads(call(mcp, "health", {}))
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["vision_enabled"])
        self.assertTrue(payload["ocr_enabled"])
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["ocr_engine"], "vlm")
        self.assertIn("credentials", payload)


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
        error = call_failure(mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "bogus"})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "ocr")
        self.assertIn("unknown OCR engine", payload["error"]["message"])

    def test_extract_text_engine_override_rapidocr_error_when_unavailable(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", False):
            error = call_failure(
                mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "rapidocr"}
            )
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "ocr")
        self.assertIn("requires the 'ocr' extra", payload["error"]["message"])

    def test_extract_text_rapidocr_success(self):
        engine = mock.Mock()
        engine.extract.return_value = {"text": "LOCAL", "engine": "rapidocr", "items": []}
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", True), mock.patch(
            "look_n_see.ocr.RapidOcrEngine", return_value=engine
        ):
            result = call(
                mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "rapidocr"}
            )
        self.assertEqual(json.loads(result)["text"], "LOCAL")
        engine.extract.assert_called_once_with(PNG_DATA_URI, prompt=None, model=None)
        self.assertEqual(fake.calls, [])  # local OCR must not hit the vision client

    def test_engine_instance_cached_across_calls(self):
        engine = mock.Mock()
        engine.extract.return_value = {"text": "T", "engine": "rapidocr", "items": []}
        mcp = create_server(cfg=CFG, client=FakeClient())
        with mock.patch("look_n_see.ocr._RAPIDOCR_AVAILABLE", True), mock.patch(
            "look_n_see.ocr.RapidOcrEngine", return_value=engine
        ) as factory:
            for _ in range(2):
                call(mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "rapidocr"})
        factory.assert_called_once_with()  # local model loaded once per process
        self.assertEqual(engine.extract.call_count, 2)

    def test_extract_text_mistral_requires_key(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(
            mcp, "extract_text", {"image": PNG_DATA_URI, "engine": "mistral"}
        )
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "ocr")
        self.assertIn("MISTRAL_API_KEY", payload["error"]["message"])

    def test_extract_text_invalid_image(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(mcp, "extract_text", {"image": "/no/such/file.png"})
        self.assertEqual(error_payload(error)["error"]["type"], "input")

    def test_extract_text_unexpected_error_returns_structured_error(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        with mock.patch("look_n_see.server.to_data_uri", side_effect=ValueError("boom")):
            error = call_failure(mcp, "extract_text", {"image": PNG_DATA_URI})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "internal")
        self.assertIn("boom", payload["error"]["message"])

    def test_extract_text_inference_failure_is_inference_error(self):
        from look_n_see.client import InferenceError

        fake = FakeClient(error=InferenceError("provider down"))
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(mcp, "extract_text", {"image": PNG_DATA_URI})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "inference")
        self.assertIn("provider down", payload["error"]["message"])

    def test_extract_text_unexpected_engine_error_is_internal(self):
        mcp = create_server(cfg=CFG, client=FakeClient())
        with mock.patch(
            "look_n_see.server.OcrDispatcher.extract", side_effect=ValueError("boom")
        ):
            error = call_failure(mcp, "extract_text", {"image": PNG_DATA_URI})
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "internal")
        self.assertIn("boom", payload["error"]["message"])


class _UrlToolHandler(http.server.BaseHTTPRequestHandler):
    """Minimal loopback image server for tool-level URL input tests."""

    PNG_BYTES = bytes.fromhex("89504e470d0a1a0a")

    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path == "/shot.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.PNG_BYTES)))
            self.end_headers()
            self.wfile.write(self.PNG_BYTES)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):
        pass


class UrlInputToolTests(unittest.TestCase):
    """End-to-end: an http(s) image URL given to a tool is fetched, converted
    to a data URI, and embedded in the chat message."""

    @classmethod
    def setUpClass(cls):
        cls._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _UrlToolHandler)
        threading.Thread(target=cls._server.serve_forever, daemon=True).start()
        cls.port = cls._server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()
        cls._server.server_close()

    def test_vision_tool_fetches_url_input(self):
        fake = FakeClient(result="ok")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(
            mcp, "describe_image", {"image": f"http://127.0.0.1:{self.port}/shot.png"}
        )
        self.assertEqual(result, "ok")
        (messages,) = fake.calls
        embedded = messages[0]["content"][0]["image_url"]["url"]
        self.assertTrue(embedded.startswith("data:image/png;base64,"))

    def test_extract_text_fetches_url_input(self):
        fake = FakeClient(result="HELLO")
        mcp = create_server(cfg=CFG, client=fake)
        result = call(
            mcp, "extract_text", {"image": f"http://127.0.0.1:{self.port}/shot.png"}
        )
        self.assertIn('"engine": "vlm"', result)
        self.assertIn("HELLO", result)

    def test_unreachable_url_returns_structured_error(self):
        fake = FakeClient()
        mcp = create_server(cfg=CFG, client=fake)
        # Port 1 (tcpmux) has nothing listening; connection refused.
        error = call_failure(
            mcp, "describe_image", {"image": "http://127.0.0.1:1/nope.png"}
        )
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "input")
        self.assertIn("failed to fetch image URL", payload["error"]["message"])

    def test_url_404_returns_structured_error(self):
        fake = FakeClient()
        mcp = create_server(cfg=CFG, client=fake)
        error = call_failure(
            mcp, "describe_image", {"image": f"http://127.0.0.1:{self.port}/gone.png"}
        )
        self.assertEqual(error_payload(error)["error"]["type"], "input")

    def test_private_url_refused_without_opt_in(self):
        cfg = Config(**{**CFG.__dict__, "allow_private_urls": False})
        fake = FakeClient()
        mcp = create_server(cfg=cfg, client=fake)
        error = call_failure(
            mcp, "describe_image", {"image": f"http://127.0.0.1:{self.port}/shot.png"}
        )
        payload = error_payload(error)
        self.assertEqual(payload["error"]["type"], "input")
        self.assertIn("private address", payload["error"]["message"])
        self.assertEqual(fake.calls, [])


class ModuleEntryTests(unittest.TestCase):
    def test_python_m_starts_and_exits_cleanly_on_eof(self):
        # The packaging entry point: `python -m look_n_see` boots the stdio
        # server without any network call and exits 0 when stdin closes.
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "look_n_see"],
            input=b"",
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
