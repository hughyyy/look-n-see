"""Unit tests for provider-aware configuration."""

import os
import unittest
from unittest import mock

from look_n_see.config import DEFAULT_FALLBACKS, Config, load_config


def make_config(**kw) -> Config:
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


class ProviderResolutionTests(unittest.TestCase):
    def test_auto_prefix_wins(self):
        cfg = make_config(provider="auto")
        self.assertEqual(cfg.provider_for("openrouter:q/q"), "openrouter")
        self.assertEqual(cfg.provider_for("plain/model"), "hf")
        self.assertEqual(cfg.provider_for("hf:x/y"), "hf")

    def test_provider_openrouter_forces_bare_ids(self):
        cfg = make_config(provider="openrouter")
        self.assertEqual(cfg.provider_for("q/q"), "openrouter")
        self.assertEqual(cfg.provider_for("hf:q/q"), "hf")

    def test_provider_hf_ignores_bare_ids(self):
        cfg = make_config(provider="hf")
        self.assertEqual(cfg.provider_for("q/q"), "hf")
        self.assertEqual(cfg.provider_for("openrouter:q/q"), "openrouter")

    def test_strip_prefix(self):
        self.assertEqual(Config.strip_prefix("openrouter:a/b"), "a/b")
        self.assertEqual(Config.strip_prefix("hf:a/b"), "a/b")
        self.assertEqual(Config.strip_prefix("a/b"), "a/b")


class FallbackParsingTests(unittest.TestCase):
    def test_comma_and_space_split(self):
        cfg = make_config(fallbacks=("a", "b", "c"))
        self.assertEqual(len(cfg.fallbacks), 3)


class LoadConfigTests(unittest.TestCase):
    def test_env_parsing(self):
        with mock.patch.dict(
            os.environ,
            {
                "VISION_MODEL": "openrouter:q/q",
                "VISION_PROVIDER": "openrouter",
                "VISION_FALLBACKS": "hf:a/b, openrouter:c/d",
                "OPENROUTER_API_KEY": "sk-test",
                "OPENROUTER_BASE_URL": "https://or.example/v1",
                "OPENROUTER_SITE_URL": "https://site.example",
                "OPENROUTER_APP_NAME": "bench",
                "VISION_MAX_TOKENS": "512",
            },
            clear=False,
        ):
            cfg = load_config()
        self.assertEqual(cfg.model, "openrouter:q/q")
        self.assertEqual(cfg.provider, "openrouter")
        self.assertEqual(cfg.fallbacks, ("hf:a/b", "openrouter:c/d"))
        self.assertEqual(cfg.openrouter_api_key, "sk-test")
        self.assertEqual(cfg.openrouter_base_url, "https://or.example/v1")
        self.assertEqual(cfg.openrouter_site_url, "https://site.example")
        self.assertEqual(cfg.openrouter_app_name, "bench")
        self.assertEqual(cfg.max_tokens, 512)

    def test_defaults(self):
        with mock.patch.dict(
            os.environ,
            {
                "VISION_FALLBACKS": "",
                "OPENROUTER_API_KEY": "",
                "HF_ENDPOINT": "",
            },
            clear=False,
        ):
            cfg = load_config()
        self.assertEqual(cfg.provider, "auto")
        self.assertEqual(cfg.fallbacks, ())
        self.assertEqual(cfg.openrouter_api_key, None)
        self.assertEqual(cfg.endpoint, None)
        self.assertEqual(cfg.openrouter_base_url, "https://openrouter.ai/api/v1")

    def test_bad_int_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"VISION_MAX_TOKENS": "banana"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.max_tokens, 2048)

    def test_default_chain_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VISION_FALLBACKS", None)
            cfg = load_config()
        self.assertEqual(
            cfg.fallbacks,
            ("Qwen/Qwen3-VL-30B-A3B-Instruct", "openrouter:google/gemma-4-31b-it"),
        )
        self.assertEqual(cfg.fallbacks, DEFAULT_FALLBACKS)

    def test_empty_string_disables_chain(self):
        with mock.patch.dict(os.environ, {"VISION_FALLBACKS": " "}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.fallbacks, ())

    def test_role_flags_default_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            cfg = load_config()
        self.assertTrue(cfg.vision_enabled)
        self.assertTrue(cfg.ocr_enabled)
        self.assertEqual(cfg.ocr_engine, "vlm")
        self.assertIsNone(cfg.mistral_api_key)

    def test_role_flags_parse_off_forms(self):
        for raw in ("0", "false", "off", "no", ""):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {"VISION_ENABLED": raw, "OCR_ENABLED": "false", "OCR_ENGINE": "rapidocr"},
                    clear=False,
                ):
                    cfg = load_config()
                self.assertFalse(cfg.vision_enabled)
                self.assertFalse(cfg.ocr_enabled)
                self.assertEqual(cfg.ocr_engine, "rapidocr")

    def test_ocr_engine_validated(self):
        with mock.patch.dict(os.environ, {"OCR_ENGINE": "Mistral"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.ocr_engine, "mistral")
        with mock.patch.dict(os.environ, {"OCR_ENGINE": "wat"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.ocr_engine, "vlm")

    def test_mistral_key_loaded(self):
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "key123"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.mistral_api_key, "key123")


if __name__ == "__main__":
    unittest.main()
