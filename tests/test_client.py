"""Unit tests for VisionClient: per-call model, fallback chain, retries, metadata."""

import time
import unittest
from unittest import mock

import httpx

from hf_vision_mcp import client as client_mod
from hf_vision_mcp.client import Completion, InferenceError, VisionClient, _describe_error, _is_retryable
from hf_vision_mcp.config import Config
from hf_vision_mcp.openrouter import RawResult


def make_config(**kw) -> Config:
    base = dict(
        model="prim",
        endpoint=None,
        max_tokens=64,
        timeout=5.0,
        retries=0,
        retry_backoff=1.0,
    )
    base.update(kw)
    return Config(**base)


class FakeTransport:
    """Records calls; returns a scripted result per call or raises."""

    def __init__(self, behavior):
        self.behavior = behavior  # list of RawResult | Exception, replayed per call
        self.calls: list[tuple[list, str, int]] = []

    def chat_completion(self, messages, model, max_tokens):
        self.calls.append((messages, model, max_tokens))
        if self.behavior:
            item = self.behavior.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return RawResult(text="ok", finish_reason="stop", usage={"completion_tokens": 1})


def attach(fake: FakeTransport, client: VisionClient) -> None:
    client._transport_for = lambda provider: fake  # noqa: E731 - test seam


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://x/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


class FallbackChainTests(unittest.TestCase):
    def test_requested_model_used_and_stripped(self):
        cfg = make_config()
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        result = client.complete_with_meta([{"role": "user"}], model="openrouter:q/q:free")
        self.assertEqual(fake.calls[0][1], "q/q:free")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.model, "openrouter:q/q:free")

    def test_fallback_used_after_primary_failure(self):
        cfg = make_config(fallbacks=("fb",))
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom")])
        attach(fake, client)
        # first failure is non-retryable -> immediate fallback; transport returns ok
        result = client.complete([{"role": "user"}], model="prim")
        self.assertEqual(result, "ok")
        models_called = [c[1] for c in fake.calls]
        self.assertEqual(models_called, ["prim", "fb"])

    def test_all_fail_reports_chain(self):
        cfg = make_config(fallbacks=("fb",))
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom-a"), RuntimeError("boom-b")])
        attach(fake, client)
        with self.assertRaises(InferenceError) as ctx:
            client.complete([{"role": "user"}], model="prim")
        message = str(ctx.exception)
        self.assertIn("prim", message)
        self.assertIn("fb", message)
        self.assertIn("boom-a", message)

    def test_dedup_repeated_chain_entries(self):
        cfg = make_config(fallbacks=("prim", "prim"))
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        client.complete([{"role": "user"}], model="prim")
        self.assertEqual(len(fake.calls), 1)

    def test_empty_fallbacks_disables_chain(self):
        cfg = make_config(fallbacks=("fb",))
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom")])
        attach(fake, client)
        with self.assertRaises(InferenceError):
            client.complete([{"role": "user"}], model="prim", fallbacks=())
        self.assertEqual([c[1] for c in fake.calls], ["prim"])

    def test_single_model_error_matches_old_shape(self):
        cfg = make_config(fallbacks=())
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom")])
        attach(fake, client)
        with self.assertRaises(InferenceError) as ctx:
            client.complete([{"role": "user"}], model="prim")
        self.assertEqual(str(ctx.exception), "vision model call failed (prim): boom")


class RetryTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        cfg = make_config(retries=2, retry_backoff=1.0)
        client = VisionClient(cfg)
        fake = FakeTransport([_http_error(429), _http_error(503)])
        attach(fake, client)
        with mock.patch.object(time, "sleep") as sleep:
            result = client.complete_with_meta([{"role": "user"}], model="prim")
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.retries, 2)
        self.assertEqual(sleep.call_count, 2)

    def test_retries_exhausted_raises(self):
        cfg = make_config(retries=1)
        client = VisionClient(cfg)
        fake = FakeTransport([_http_error(429), _http_error(429), _http_error(429)])
        attach(fake, client)
        with mock.patch.object(time, "sleep"):
            with self.assertRaises(InferenceError):
                client.complete([{"role": "user"}], model="prim")
        self.assertEqual(len(fake.calls), 2)  # retries+1

    def test_non_retryable_stops_immediately(self):
        cfg = make_config(retries=3)
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom")])
        attach(fake, client)
        with self.assertRaises(InferenceError):
            client.complete([{"role": "user"}], model="prim")
        self.assertEqual(len(fake.calls), 1)

    def test_openrouter_missing_key_fails_fast(self):
        cfg = make_config()
        client = VisionClient(cfg)
        with self.assertRaises(InferenceError) as ctx:
            client.complete([{"role": "user"}], model="openrouter:q/q")
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_openrouter_success(self):
        cfg = make_config(openrouter_api_key="sk-test")
        client = VisionClient(cfg)
        payload = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 2},
        }
        with mock.patch(
            "hf_vision_mcp.openrouter.httpx.Client.post",
            return_value=httpx.Response(
                200,
                json=payload,
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
        ) as post:
            result = client.complete_with_meta([{"role": "user"}], model="openrouter:q/q")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.transport, "OpenRouterTransport")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "q/q")

    def test_hf_transport_constructor_model(self):
        # huggingface_hub maps the InferenceClient constructor model to the HF
        # router; an openrouter:-prefixed default must never reach it, and a
        # bare default would leak lowercase slugs. With no endpoint the
        # constructor carries no model — per-call ids drive routing.
        cfg = make_config(model="openrouter:google/gemma-4-26b-a4b-it")
        with mock.patch("hf_vision_mcp.client.InferenceClient") as ic:
            VisionClient(cfg)
        args = ic.call_args.args
        model_arg = ic.call_args.kwargs.get("model") or (args[0] if args else None)
        self.assertIsNone(model_arg)

        cfg = make_config(endpoint="https://endpoint.example")
        with mock.patch("hf_vision_mcp.client.InferenceClient") as ic:
            VisionClient(cfg)
        args = ic.call_args.args
        model_arg = ic.call_args.kwargs.get("model") or (args[0] if args else None)
        self.assertEqual(model_arg, "https://endpoint.example")

    def test_metadata_fields(self):
        cfg = make_config(retries=0)
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        result = client.complete_with_meta([{"role": "user"}], model="prim")
        self.assertIsInstance(result, Completion)
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["completion_tokens"], 1)
        self.assertEqual(result.provider, "hf")
        self.assertGreaterEqual(result.latency_s, 0.0)
        self.assertEqual(result.retries, 0)


class ErrorHelpersTests(unittest.TestCase):
    def test_is_retryable_statuses(self):
        for status in (429, 500, 502, 503, 504):
            self.assertTrue(_is_retryable(_http_error(status)), status)
        self.assertFalse(_is_retryable(_http_error(401)))

    def test_describe_error_uses_api_message(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(
            402, json={"error": {"message": "Insufficient credits"}}, request=request
        )
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "402: Insufficient credits")


if __name__ == "__main__":
    unittest.main()
