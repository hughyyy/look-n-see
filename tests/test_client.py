"""Unit tests for VisionClient: per-call model, fallback chain, retries, metadata."""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from look_n_see.client import (
    Completion,
    InferenceError,
    VisionClient,
    _describe_error,
    _is_retryable,
    _retry_after_seconds,
)
from look_n_see.config import Config
from look_n_see.openrouter import OpenRouterTransport, RawResult


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
        self.timeouts: list[float | None] = []

    def chat_completion(self, messages, model, max_tokens, timeout=None):
        self.calls.append((messages, model, max_tokens))
        self.timeouts.append(timeout)
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

    def test_model_id_is_normalized_before_routing(self):
        # Case-insensitive prefix + surrounding whitespace, bare id unchanged.
        client = VisionClient(make_config())
        fake = FakeTransport([])
        attach(fake, client)
        result = client.complete_with_meta(
            [{"role": "user"}], model="  OPENROUTER:q/q  "
        )
        self.assertEqual(fake.calls[0][1], "q/q")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.model, "openrouter:q/q")

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

    def test_dedup_applies_after_failure_too(self):
        # A repeated chain entry that was already tried and failed must not
        # be retried; the error report lists each candidate once.
        cfg = make_config(fallbacks=("fb", "fb"))
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom-a"), RuntimeError("boom-b")])
        attach(fake, client)
        with self.assertRaises(InferenceError) as ctx:
            client.complete([{"role": "user"}], model="prim")
        self.assertEqual(len(fake.calls), 2)  # prim, fb — repeated fb skipped
        message = str(ctx.exception)
        self.assertIn("[prim]", message)
        self.assertEqual(message.count("[fb]"), 1)

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

    def test_fallbacks_argument_overrides_config(self):
        cfg = make_config(fallbacks=("cfg-fb",))
        client = VisionClient(cfg)
        fake = FakeTransport([RuntimeError("boom-a"), RuntimeError("boom-b")])
        attach(fake, client)
        with self.assertRaises(InferenceError):
            client.complete([{"role": "user"}], model="prim", fallbacks=("arg-fb",))
        self.assertEqual([c[1] for c in fake.calls], ["prim", "arg-fb"])

    def test_missing_openrouter_key_falls_through_to_hf_fallback(self):
        cfg = make_config(model="openrouter:q/q", fallbacks=("hf:fb",))
        client = VisionClient(cfg)
        calls = []
        client._hf.chat_completion = lambda messages, model, max_tokens, timeout=None: (
            calls.append(model) or RawResult(text="hf ok")
        )
        self.assertEqual(client.complete([{"role": "user"}]), "hf ok")
        self.assertEqual(calls, ["fb"])  # provider prefix stripped for the call

    def test_hf_prefix_stripped_for_transport_call(self):
        client = VisionClient(make_config())
        fake = FakeTransport([])
        attach(fake, client)
        result = client.complete_with_meta([{"role": "user"}], model="hf:org/model")
        self.assertEqual(fake.calls[0][1], "org/model")
        self.assertEqual(result.provider, "hf")

    def test_provider_openrouter_routes_bare_model(self):
        cfg = make_config(provider="openrouter", openrouter_api_key="sk-test")
        payload = {"choices": [{"message": {"content": "or"}}]}
        with mock.patch(
            "look_n_see.openrouter.httpx.Client.post",
            return_value=httpx.Response(
                200,
                json=payload,
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
        ) as post:
            result = VisionClient(cfg).complete_with_meta(
                [{"role": "user"}], model="q/q"
            )
        self.assertEqual(result.text, "or")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "q/q")

    def test_provider_openrouter_bare_model_requires_key(self):
        with self.assertRaisesRegex(InferenceError, "OPENROUTER_API_KEY"):
            VisionClient(make_config(provider="openrouter")).complete(
                [{"role": "user"}], model="q/q"
            )

    def test_hf_endpoint_does_not_pin_openrouter_fallback(self):
        # HF_ENDPOINT pins only the HF transport; an OpenRouter fallback in the
        # chain must still address its own model id.
        cfg = make_config(
            model="hf:primary",
            endpoint="https://ep.example",
            fallbacks=("openrouter:q/q",),
            openrouter_api_key="sk-test",
        )
        client = VisionClient(cfg)
        client._hf.chat_completion = mock.Mock(side_effect=RuntimeError("hf down"))
        payload = {"choices": [{"message": {"content": "or ok"}}]}
        with mock.patch(
            "look_n_see.openrouter.httpx.Client.post",
            return_value=httpx.Response(
                200,
                json=payload,
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
        ) as post:
            self.assertEqual(client.complete([{"role": "user"}]), "or ok")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "q/q")

    def test_no_deadline_when_disabled(self):
        cfg = make_config(timeout=7.0, deadline=0)
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        client.complete([{"role": "user"}], model="prim")
        self.assertEqual(fake.timeouts[0], 7.0)

    def test_retry_after_non_numeric_header_ignored(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(
            429, headers={"Retry-After": "later"}, request=request
        )
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertIsNone(_retry_after_seconds(exc))
        self.assertIsNone(_retry_after_seconds(RuntimeError("no response")))

    def test_unexpected_transport_construction_error_falls_back(self):
        cfg = make_config(fallbacks=("fb",))
        client = VisionClient(cfg)
        fake = FakeTransport([])
        calls = {"n": 0}

        def transport_for(provider):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("bad config")
            return fake

        client._transport_for = transport_for
        self.assertEqual(client.complete([{"role": "user"}], model="prim"), "ok")
        self.assertEqual([c[1] for c in fake.calls], ["fb"])

    def test_close_releases_hf_and_openrouter_clients(self):
        cfg = make_config(openrouter_api_key="sk-test")
        hf = mock.Mock()
        client = VisionClient(cfg, client=hf)
        transport = client._transport_for("openrouter")
        with mock.patch.object(transport._http, "close") as or_close:
            client.close()
        hf.close.assert_called_once()
        or_close.assert_called_once()

    def test_close_without_openrouter_or_closable_hf_client(self):
        client = VisionClient(make_config(), client=object())
        client.close()  # no OpenRouter transport, HF client has no close()


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

    def test_backoff_is_exponential(self):
        cfg = make_config(retries=2, retry_backoff=2.0)
        client = VisionClient(cfg)
        fake = FakeTransport([_http_error(429), _http_error(503)])
        attach(fake, client)
        with mock.patch.object(time, "sleep") as sleep:
            client.complete([{"role": "user"}], model="prim")
        sleeps = [c.args[0] for c in sleep.call_args_list]
        # Equal jitter: half the exponential delay is fixed, half random.
        self.assertGreaterEqual(sleeps[0], 1.0)
        self.assertLessEqual(sleeps[0], 2.0)
        self.assertGreaterEqual(sleeps[1], 2.0)
        self.assertLessEqual(sleeps[1], 4.0)

    def test_retries_exhausted_then_fallback_used(self):
        # The full policy interplay: retry the primary model *then* escalate
        # to the fallback only after retries are exhausted.
        cfg = make_config(retries=2, fallbacks=("fb",))
        client = VisionClient(cfg)
        fake = FakeTransport([_http_error(429), _http_error(429), _http_error(429)])
        attach(fake, client)
        with mock.patch.object(time, "sleep"):
            result = client.complete([{"role": "user"}], model="prim")
        self.assertEqual(result, "ok")
        self.assertEqual([c[1] for c in fake.calls], ["prim", "prim", "prim", "fb"])

    def test_retry_after_header_honored(self):
        cfg = make_config(retries=1, retry_backoff=1.0)
        client = VisionClient(cfg)
        request = httpx.Request("POST", "https://x")
        limited = httpx.HTTPStatusError(
            "err",
            request=request,
            response=httpx.Response(
                429, headers={"Retry-After": "5"}, request=request
            ),
        )
        fake = FakeTransport([limited])
        attach(fake, client)
        with mock.patch.object(time, "sleep") as sleep:
            result = client.complete_with_meta([{"role": "user"}], model="prim")
        self.assertEqual(result.text, "ok")
        delay = sleep.call_args.args[0]
        self.assertGreaterEqual(delay, 2.5)  # max(exponential, Retry-After) / 2
        self.assertLessEqual(delay, 5.0)

    def test_failure_reports_retry_and_latency_metadata(self):
        cfg = make_config(retries=2)
        client = VisionClient(cfg)
        fake = FakeTransport([_http_error(429), _http_error(429), _http_error(429)])
        attach(fake, client)
        with mock.patch.object(time, "sleep"):
            with self.assertRaises(InferenceError) as ctx:
                client.complete([{"role": "user"}], model="prim")
        self.assertEqual(ctx.exception.retries, 2)
        self.assertGreaterEqual(ctx.exception.latency_s, 0.0)
        self.assertFalse(ctx.exception.deadline_exceeded)

    def test_deadline_caps_request_timeout(self):
        cfg = make_config(timeout=30.0, deadline=5.0)
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        client.complete([{"role": "user"}], model="prim")
        self.assertLessEqual(fake.timeouts[0], 5.0)
        self.assertGreater(fake.timeouts[0], 0.0)

    def test_deadline_stops_retries_and_fallbacks(self):
        cfg = make_config(
            retries=1, deadline=5.0, retry_backoff=1.0, fallbacks=("fb",)
        )
        client = VisionClient(cfg)
        fake = FakeTransport([_http_error(429), RawResult(text="fb ok")])
        attach(fake, client)
        # attempt start: budget left; post-failure: no time to retry; the
        # fallback candidate then sees an exhausted budget.
        remaining = mock.patch(
            "look_n_see.client._remaining", side_effect=[1.0, 0.0, 0.0]
        )
        with remaining, mock.patch.object(time, "sleep") as sleep:
            with self.assertRaises(InferenceError) as ctx:
                client.complete([{"role": "user"}], model="prim")
        self.assertTrue(ctx.exception.deadline_exceeded)
        self.assertEqual(len(fake.calls), 1)  # fallback never started
        self.assertEqual(sleep.call_count, 0)

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
            "look_n_see.openrouter.httpx.Client.post",
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
        with mock.patch("look_n_see.client.InferenceClient") as ic:
            VisionClient(cfg)
        args = ic.call_args.args
        model_arg = ic.call_args.kwargs.get("model") or (args[0] if args else None)
        self.assertIsNone(model_arg)

        cfg = make_config(endpoint="https://endpoint.example")
        with mock.patch("look_n_see.client.InferenceClient") as ic:
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

    def test_negative_retries_still_attempts_once(self):
        client = VisionClient(make_config(retries=-1))
        fake = FakeTransport([])
        attach(fake, client)
        self.assertEqual(client.complete([{"role": "user"}], model="prim"), "ok")
        self.assertEqual(len(fake.calls), 1)

    def test_last_error_reported_after_retries(self):
        request = httpx.Request("POST", "https://x")
        rate_limited = httpx.HTTPStatusError(
            "first",
            request=request,
            response=httpx.Response(
                429, json={"error": {"message": "rate limited"}}, request=request
            ),
        )
        bad_key = httpx.HTTPStatusError(
            "second",
            request=request,
            response=httpx.Response(
                401, json={"error": {"message": "bad key"}}, request=request
            ),
        )
        client = VisionClient(make_config(retries=1))
        fake = FakeTransport([rate_limited, bad_key])
        attach(fake, client)
        with mock.patch.object(time, "sleep"):
            with self.assertRaises(InferenceError) as ctx:
                client.complete([{"role": "user"}], model="prim")
        self.assertIn("bad key", str(ctx.exception))
        self.assertNotIn("rate limited", str(ctx.exception))


class HfTransportTests(unittest.TestCase):
    def _hf_client_for(self, cfg, response):
        """Build a VisionClient whose HF InferenceClient is fully mocked."""
        fake_inference_client = mock.Mock()
        fake_inference_client.chat_completion.return_value = response
        return VisionClient(cfg, client=fake_inference_client)

    def _hf_response(self, content="say", usage=None):
        message = mock.Mock()
        message.content = content
        choice = mock.Mock()
        choice.message = message
        choice.finish_reason = "stop"
        response = mock.Mock()
        response.choices = [choice]
        if usage is not None:
            # SimpleNamespace keeps __dict__ clean of Mock internals.
            response.usage = SimpleNamespace(completion_tokens=usage)
        else:
            response.usage = None
        return response

    def test_hf_chat_completion_roundtrip(self):
        cfg = make_config(model="some/model")
        client = self._hf_client_for(cfg, self._hf_response("hello", usage=3))
        result = client._hf.chat_completion(
            [{"role": "user"}], "some/model", 64
        )
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage, {"completion_tokens": 3})
        client._hf._client.chat_completion.assert_called_once_with(
            messages=[{"role": "user"}], model="some/model", max_tokens=64
        )

    def test_hf_endpoint_pins_call_model(self):
        # A dedicated endpoint URL replaces the per-call model id.
        cfg = make_config(model="some/model", endpoint="https://ep.example")
        client = self._hf_client_for(cfg, self._hf_response())
        client._hf.chat_completion([{"role": "user"}], "ignored/model", 64)
        call_kwargs = client._hf._client.chat_completion.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "https://ep.example")

    def test_hf_missing_content_becomes_empty(self):
        cfg = make_config()
        client = self._hf_client_for(cfg, self._hf_response(content=None))
        result = client._hf.chat_completion([], "m", 64)
        self.assertEqual(result.text, "")

    def test_hf_complete_without_usage(self):
        cfg = make_config()
        client = self._hf_client_for(cfg, self._hf_response(usage=None))
        result = client._hf.chat_completion([], "m", 64)
        self.assertIsNone(result.usage)

    def test_hf_empty_choices_raises_clear_error(self):
        client = self._hf_client_for(make_config(), SimpleNamespace(choices=[], usage=None))
        with self.assertRaisesRegex(InferenceError, "no choices"):
            client._hf.chat_completion([], "m", 64)

    def test_hf_missing_choices_raises_clear_error(self):
        client = self._hf_client_for(make_config(), SimpleNamespace())
        with self.assertRaisesRegex(InferenceError, "no choices"):
            client._hf.chat_completion([], "m", 64)

    def test_hf_transport_tolerates_client_without_timeout_attr(self):
        class BareClient:
            def __init__(self):
                self.calls = []

            def chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="hi"),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

        cfg = make_config()
        bare = BareClient()
        client = VisionClient(cfg, client=bare)
        result = client._hf.chat_completion([], "m", 64, timeout=2.0)
        self.assertEqual(result.text, "hi")
        self.assertFalse(hasattr(bare, "timeout"))

    def test_hf_usage_dict_is_preserved(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi"), finish_reason="stop"
                )
            ],
            usage={"completion_tokens": 2},
        )
        client = self._hf_client_for(make_config(), response)
        result = client._hf.chat_completion([], "m", 64)
        self.assertEqual(result.usage, {"completion_tokens": 2})

    def test_hf_message_missing_content_is_empty(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(), finish_reason=None)
            ],
            usage=None,
        )
        client = self._hf_client_for(make_config(), response)
        result = client._hf.chat_completion([], "m", 64)
        self.assertEqual(result.text, "")

    def test_hf_constructor_uses_token_and_timeout(self):
        cfg = make_config(timeout=7.5)
        with mock.patch.dict("os.environ", {"HF_TOKEN": "hf-test"}), mock.patch(
            "look_n_see.client.InferenceClient"
        ) as inference_client:
            VisionClient(cfg)
        kwargs = inference_client.call_args.kwargs
        self.assertEqual(kwargs["token"], "hf-test")
        self.assertEqual(kwargs["timeout"], 7.5)

    def test_hf_endpoint_pins_model_through_complete(self):
        cfg = make_config(model="some/model", endpoint="https://ep.example")
        client = self._hf_client_for(cfg, self._hf_response("ok"))
        self.assertEqual(
            client.complete([{"role": "user"}], model="other/model"), "ok"
        )
        call_kwargs = client._hf._client.chat_completion.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "https://ep.example")

    def test_hf_empty_usage_namespace_is_none(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x"), finish_reason=None
                )
            ],
            usage=SimpleNamespace(),
        )
        client = self._hf_client_for(make_config(), response)
        self.assertIsNone(client._hf.chat_completion([], "m", 64).usage)


class TransportSelectionTests(unittest.TestCase):
    def test_hf_branch_returns_internal_hf_transport(self):
        client = VisionClient(make_config())
        self.assertIs(client._transport_for("hf"), client._hf)

    def test_openrouter_transport_created_once_and_cached(self):
        cfg = make_config(openrouter_api_key="sk-test")
        client = VisionClient(cfg)
        first = client._transport_for("openrouter")
        second = client._transport_for("openrouter")
        self.assertIsInstance(first, OpenRouterTransport)
        self.assertIs(first, second)


class DefaultModelTests(unittest.TestCase):
    def test_default_model_when_arg_omitted(self):
        cfg = make_config(model="cfg-model")
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        client.complete([{"role": "user"}])
        self.assertEqual(fake.calls[0][1], "cfg-model")
        self.assertEqual(fake.calls[0][2], 64)  # config max_tokens

    def test_per_call_max_tokens_override(self):
        cfg = make_config()
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        client.complete([{"role": "user"}], max_tokens=999)
        self.assertEqual(fake.calls[0][2], 999)

    def test_complete_returns_text_only(self):
        cfg = make_config()
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        self.assertEqual(client.complete([{"role": "user"}]), "ok")

    def test_zero_max_tokens_uses_config(self):
        cfg = make_config()
        client = VisionClient(cfg)
        fake = FakeTransport([])
        attach(fake, client)
        client.complete([{"role": "user"}], max_tokens=0)
        self.assertEqual(fake.calls[0][2], 64)  # falsy -> config max_tokens


class ErrorHelpersTests(unittest.TestCase):
    def test_is_retryable_statuses(self):
        for status in (408, 425, 429, 500, 502, 503, 504, 529):
            self.assertTrue(_is_retryable(_http_error(status)), status)
        self.assertFalse(_is_retryable(_http_error(401)))

    def test_is_retryable_connection_failures(self):
        self.assertTrue(_is_retryable(ConnectionError("down")))
        self.assertTrue(_is_retryable(TimeoutError("slow")))

    def test_is_retryable_by_class_name(self):
        # httpx/urllib3 style exceptions without a response attribute.
        class ReadTimeout(Exception):
            pass

        class NotFound(Exception):
            pass

        self.assertTrue(_is_retryable(ReadTimeout("t")))
        self.assertFalse(_is_retryable(NotFound("x")))
        self.assertFalse(_is_retryable(ValueError("bad")))

    def test_non_retryable_status_boundaries(self):
        # Policy boundary: request-timeout, overload, and gateway/availability
        # statuses retry; other 4xx/5xx are terminal.
        for status in (400, 401, 501, 505):
            with self.subTest(status=status):
                self.assertFalse(_is_retryable(_http_error(status)), status)

    def test_is_retryable_real_httpx_timeouts(self):
        for exc in (httpx.ReadTimeout("t"), httpx.ConnectTimeout("t")):
            with self.subTest(exc=type(exc).__name__):
                self.assertTrue(_is_retryable(exc))

    def test_is_retryable_real_httpx_connect_error(self):
        # The real transient failure mode: connection refused. httpx wraps it
        # in its own ConnectError, which must be retryable.
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        try:
            httpx.Client(timeout=1.0).post(f"http://127.0.0.1:{port}/x")
        except httpx.ConnectError as exc:
            self.assertTrue(_is_retryable(exc))
        else:
            self.fail("expected httpx.ConnectError")

    def test_connect_error_retried_then_succeeds(self):
        # End-to-end: a connection failure triggers the backoff retry policy.
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        cfg = make_config(retries=1, retry_backoff=0.01)
        client = VisionClient(cfg, client=mock.Mock())
        real = client._hf
        attempts = {"n": 0}

        def flaky_chat_completion(messages, model, max_tokens, timeout=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError(
                    f"All connection attempts failed to 127.0.0.1:{port}"
                )
            return RawResult(text="recovered")

        real.chat_completion = flaky_chat_completion
        result = client.complete([{"role": "user"}], model="m")
        self.assertEqual(result, "recovered")
        self.assertEqual(attempts["n"], 2)

    def test_describe_error_uses_api_message(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(
            402, json={"error": {"message": "Insufficient credits"}}, request=request
        )
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "402: Insufficient credits")

    def test_describe_error_string_error_body(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(
            402, json={"error": "Insufficient credits"}, request=request
        )
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "402: Insufficient credits")

    def test_describe_error_non_json_body_falls_back(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(500, text="<html>boom</html>", request=request)
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "err")

    def test_describe_error_empty_json_body_falls_back(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(500, json={}, request=request)
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "err")

    def test_is_retryable_response_without_status_code(self):
        # A response-like object missing status_code falls through to the
        # class-name check and is not retryable.
        class WeirdResponse:
            pass

        exc = RuntimeError("boom")
        exc.response = WeirdResponse()
        self.assertFalse(_is_retryable(exc))

    def test_describe_error_non_dict_json_body(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(500, json=["boom"], request=request)
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "err")

    def test_describe_error_error_without_message(self):
        request = httpx.Request("POST", "https://x")
        response = httpx.Response(500, json={"error": {"code": "bad"}}, request=request)
        exc = httpx.HTTPStatusError("err", request=request, response=response)
        self.assertEqual(_describe_error(exc), "err")


if __name__ == "__main__":
    unittest.main()
