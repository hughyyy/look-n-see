"""Unit tests for the OpenRouter transport."""

import unittest
from unittest import mock

import httpx

from look_n_see.config import Config
from look_n_see.openrouter import OpenRouterError, OpenRouterTransport


def make_config(**kw) -> Config:
    base = dict(
        model="m",
        endpoint=None,
        max_tokens=64,
        timeout=5.0,
        retries=0,
        retry_backoff=1.0,
        openrouter_api_key="sk-test",
    )
    base.update(kw)
    return Config(**base)


def _response(status: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(status, json=payload or {}, request=request)


class OpenRouterTransportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config(
            openrouter_site_url="https://site.example", openrouter_app_name="bench"
        )
        self.transport = OpenRouterTransport(self.cfg)

    def test_requires_api_key(self):
        with self.assertRaises(ValueError):
            OpenRouterTransport(make_config(openrouter_api_key=None))

    def test_headers_and_body(self):
        ok_payload = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }
        with mock.patch.object(
            self.transport._http,
            "post",
            return_value=_response(200, ok_payload),
        ) as post:
            self.transport.chat_completion(
                [{"role": "user", "content": "hi"}], "q/q:free", 128
            )
        post.assert_called_once()
        kwargs = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], "/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "q/q:free")
        self.assertEqual(kwargs["json"]["max_tokens"], 128)
        self.assertEqual(kwargs["json"]["messages"], [{"role": "user", "content": "hi"}])

    def test_parses_result(self):
        payload = {
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3, "prompt_tokens": 2},
        }
        with mock.patch.object(self.transport._http, "post", return_value=_response(200, payload)):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(result.text, "42")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["completion_tokens"], 3)

    def test_content_parts_list(self):
        payload = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}
            ]
        }
        with mock.patch.object(self.transport._http, "post", return_value=_response(200, payload)):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(result.text, "ab")

    def test_http_error_raises_status_error(self):
        err_body = {"error": {"message": "Insufficient credits"}}
        with mock.patch.object(self.transport._http, "post", return_value=_response(402, err_body)):
            with self.assertRaises(httpx.HTTPStatusError) as ctx:
                self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(ctx.exception.response.status_code, 402)

    def test_site_url_headers_constructed(self):
        transport = OpenRouterTransport(self.cfg)
        self.assertEqual(transport._http.headers["HTTP-Referer"], "https://site.example")
        self.assertEqual(transport._http.headers["X-Title"], "bench")
        self.assertEqual(transport._http.headers["Authorization"], "Bearer sk-test")

    def test_optional_headers_omitted_when_unset(self):
        transport = OpenRouterTransport(make_config())
        self.assertNotIn("HTTP-Referer", transport._http.headers)
        self.assertNotIn("X-Title", transport._http.headers)

    def test_config_base_url_used(self):
        cfg = make_config(openrouter_base_url="https://or.example/v1")
        transport = OpenRouterTransport(cfg)
        self.assertEqual(str(transport._http.base_url).rstrip("/"), "https://or.example/v1")
        self.assertEqual(transport._http.timeout.read, 5.0)


class MalformedResponseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()
        self.transport = OpenRouterTransport(self.cfg)

    def test_choices_missing_raises(self):
        with mock.patch.object(self.transport._http, "post", return_value=_response(200, {})):
            with self.assertRaises(OpenRouterError) as ctx:
                self.transport.chat_completion([], "q/q", 64)
        self.assertIn("no 'choices'", str(ctx.exception))

    def test_choices_empty_raises(self):
        with mock.patch.object(
            self.transport._http, "post", return_value=_response(200, {"choices": []})
        ):
            with self.assertRaises(OpenRouterError):
                self.transport.chat_completion([], "q/q", 64)

    def test_choice_not_a_dict_raises(self):
        with mock.patch.object(
            self.transport._http, "post", return_value=_response(200, {"choices": ["nope"]})
        ):
            with self.assertRaises(OpenRouterError):
                self.transport.chat_completion([], "q/q", 64)

    def test_missing_message_returns_empty_text(self):
        payload = {"choices": [{"index": 0, "finish_reason": "stop"}]}
        with mock.patch.object(self.transport._http, "post", return_value=_response(200, payload)):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(result.text, "")

    def test_non_string_content_coerced(self):
        for content in (None, 42):
            payload = {"choices": [{"message": {"content": content}}]}
            with mock.patch.object(
                self.transport._http, "post", return_value=_response(200, payload)
            ):
                result = self.transport.chat_completion([], "q/q", 64)
            self.assertEqual(result.text, "" if content is None else "42")

    def test_content_parts_skips_non_dict_entries(self):
        payload = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "a"}, "raw", None, 42]}}
            ]
        }
        with mock.patch.object(self.transport._http, "post", return_value=_response(200, payload)):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(result.text, "a")

    def test_non_object_body_raises(self):
        for body in ([1, 2], "text"):
            with self.subTest(body=body):
                with mock.patch.object(
                    self.transport._http, "post", return_value=_response(200, body)
                ):
                    with self.assertRaisesRegex(
                        OpenRouterError, "expected a JSON object"
                    ):
                        self.transport.chat_completion([], "q/q", 64)

    def test_invalid_json_body_raises(self):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(200, text="<html>not json</html>", request=request)
        with mock.patch.object(self.transport._http, "post", return_value=response):
            with self.assertRaisesRegex(OpenRouterError, "invalid JSON"):
                self.transport.chat_completion([], "q/q", 64)

    def test_non_string_content_parts_are_skipped(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "text", "text": 42},
                            {"type": "text"},
                        ]
                    }
                }
            ]
        }
        with mock.patch.object(
            self.transport._http, "post", return_value=_response(200, payload)
        ):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(result.text, "a")

    def test_missing_usage_is_none(self):
        payload = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch.object(
            self.transport._http, "post", return_value=_response(200, payload)
        ):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertIsNone(result.usage)

    def test_missing_finish_reason_is_none(self):
        payload = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch.object(
            self.transport._http, "post", return_value=_response(200, payload)
        ):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertIsNone(result.finish_reason)

    def test_empty_content_list_is_empty_string(self):
        payload = {"choices": [{"message": {"content": []}}]}
        with mock.patch.object(
            self.transport._http, "post", return_value=_response(200, payload)
        ):
            result = self.transport.chat_completion([], "q/q", 64)
        self.assertEqual(result.text, "")


class LifecycleTests(unittest.TestCase):
    def test_close_closes_http_client(self):
        transport = OpenRouterTransport(make_config())
        with mock.patch.object(transport._http, "close") as close:
            transport.close()
        close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
