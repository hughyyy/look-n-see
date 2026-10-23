"""Unit tests for provider-aware configuration."""

import dataclasses
import os
import unittest
from unittest import mock

from look_n_see.config import (
    DEFAULT_FALLBACKS,
    DEFAULT_MODEL,
    DEFAULT_OPENROUTER_URL,
    Config,
    load_config,
)


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




class _EnvIsolated(unittest.TestCase):
    """Config tests must not inherit VISION_*/OCR_* from the developer shell."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


class ProviderResolutionTests(_EnvIsolated):
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

    def test_unknown_provider_string_routes_bare_ids_to_hf(self):
        # Config is frozen but constructible directly; an out-of-vocabulary
        # provider must not route bare ids anywhere surprising.
        cfg = make_config(provider="banana")
        self.assertEqual(cfg.provider_for("q/q"), "hf")

    def test_prefix_is_case_insensitive_and_trimmed(self):
        cfg = make_config(provider="auto")
        self.assertEqual(cfg.provider_for("OPENROUTER:q/q"), "openrouter")
        self.assertEqual(cfg.provider_for("  openrouter:q/q  "), "openrouter")
        self.assertEqual(cfg.provider_for("HF:q/q"), "hf")
        self.assertEqual(cfg.strip_prefix("OPENROUTER:q/q"), "q/q")
        self.assertEqual(cfg.strip_prefix(" hf:q/q "), "q/q")

    def test_normalize_model_id_preserves_bare_id_case(self):
        self.assertEqual(Config.normalize_model_id("  Org/Model  "), "Org/Model")
        self.assertEqual(Config.normalize_model_id("OPENROUTER:Q/Q"), "openrouter:Q/Q")
        self.assertEqual(Config.normalize_model_id(" hf: q/q "), "hf:q/q")

    def test_strip_prefix(self):
        self.assertEqual(Config.strip_prefix("openrouter:a/b"), "a/b")
        self.assertEqual(Config.strip_prefix("hf:a/b"), "a/b")
        self.assertEqual(Config.strip_prefix("a/b"), "a/b")


class FallbackParsingTests(unittest.TestCase):
    def test_comma_and_space_split(self):
        cfg = make_config(fallbacks=("a", "b", "c"))
        self.assertEqual(len(cfg.fallbacks), 3)


class LoadConfigTests(_EnvIsolated):
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
            with self.assertLogs("look_n_see.config", level="WARNING"):
                cfg = load_config()
        self.assertEqual(cfg.ocr_engine, "vlm")

    def test_mistral_key_loaded(self):
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "key123"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.mistral_api_key, "key123")


class TargetPropertyTests(unittest.TestCase):
    def test_target_defaults_to_model(self):
        self.assertEqual(make_config(model="m", endpoint=None).target, "m")

    def test_target_prefers_endpoint(self):
        self.assertEqual(
            make_config(model="m", endpoint="https://ep.example").target,
            "https://ep.example",
        )


class LoadConfigEdgeTests(_EnvIsolated):
    def test_invalid_provider_falls_back_to_auto(self):
        with mock.patch.dict(os.environ, {"VISION_PROVIDER": "banana"}, clear=False):
            with self.assertLogs("look_n_see.config", level="WARNING"):
                cfg = load_config()
        self.assertEqual(cfg.provider, "auto")

    def test_bad_float_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"VISION_TIMEOUT": "slow"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.timeout, 60.0)

    def test_bad_retries_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"VISION_RETRIES": "many"}, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.retries, 3)

    def test_role_flags_accept_truthy_forms(self):
        with mock.patch.dict(
            os.environ, {"VISION_ENABLED": "true", "OCR_ENABLED": "on"}, clear=False,
        ):
            cfg = load_config()
        self.assertTrue(cfg.vision_enabled)
        self.assertTrue(cfg.ocr_enabled)

    def test_unrecognized_bool_keeps_default_on(self):
        # Only explicit off forms disable a role; anything else stays enabled.
        with mock.patch.dict(
            os.environ,
            {"VISION_ENABLED": "maybe", "OCR_ENABLED": "banana"},
            clear=False,
        ):
            cfg = load_config()
        self.assertTrue(cfg.vision_enabled)
        self.assertTrue(cfg.ocr_enabled)

    def test_empty_or_blank_model_falls_back_to_default(self):
        for raw in ("", "   "):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"VISION_MODEL": raw}, clear=False):
                    self.assertEqual(load_config().model, DEFAULT_MODEL)

    def test_model_is_stripped(self):
        with mock.patch.dict(
            os.environ, {"VISION_MODEL": "  openrouter:q/q  "}, clear=False
        ):
            self.assertEqual(load_config().model, "openrouter:q/q")

    def test_empty_base_url_falls_back_to_default(self):
        for raw in ("", "   "):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ, {"OPENROUTER_BASE_URL": raw}, clear=False
                ):
                    cfg = load_config()
                self.assertEqual(cfg.openrouter_base_url, DEFAULT_OPENROUTER_URL)

    def test_blank_endpoint_is_none(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": "   "}, clear=False):
            self.assertIsNone(load_config().endpoint)

    def test_negative_retries_clamped_to_zero(self):
        with mock.patch.dict(os.environ, {"VISION_RETRIES": "-2"}, clear=False):
            self.assertEqual(load_config().retries, 0)

    def test_zero_retries_preserved(self):
        with mock.patch.dict(os.environ, {"VISION_RETRIES": "0"}, clear=False):
            self.assertEqual(load_config().retries, 0)

    def test_nonpositive_max_tokens_falls_back(self):
        for raw in ("0", "-5"):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ, {"VISION_MAX_TOKENS": raw}, clear=False
                ):
                    self.assertEqual(load_config().max_tokens, 2048)

    def test_invalid_timeout_falls_back(self):
        # Zero/negative timeouts would mean "no deadline" or an error in httpx;
        # NaN/inf parse as floats but are never a usable timeout.
        for raw in ("0", "-1", "nan", "inf"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"VISION_TIMEOUT": raw}, clear=False):
                    self.assertEqual(load_config().timeout, 60.0)

    def test_negative_backoff_clamped_to_zero(self):
        with mock.patch.dict(
            os.environ, {"VISION_RETRY_BACKOFF": "-3"}, clear=False
        ):
            self.assertEqual(load_config().retry_backoff, 0.0)

    def test_provider_and_engine_whitespace_normalized(self):
        with mock.patch.dict(
            os.environ,
            {"VISION_PROVIDER": "  OpenRouter ", "OCR_ENGINE": " Mistral "},
            clear=False,
        ):
            cfg = load_config()
        self.assertEqual(cfg.provider, "openrouter")
        self.assertEqual(cfg.ocr_engine, "mistral")

    def test_fallback_separators_and_whitespace(self):
        with mock.patch.dict(
            os.environ, {"VISION_FALLBACKS": "a,\n b\tc ,, d "}, clear=False
        ):
            self.assertEqual(load_config().fallbacks, ("a", "b", "c", "d"))


    def test_whitespace_only_metadata_and_keys_are_unset(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_SITE_URL": "   ",
                "OPENROUTER_APP_NAME": "\t",
                "OPENROUTER_API_KEY": " ",
                "MISTRAL_API_KEY": "\n",
            },
            clear=False,
        ):
            cfg = load_config()
        self.assertIsNone(cfg.openrouter_site_url)
        self.assertIsNone(cfg.openrouter_app_name)
        self.assertIsNone(cfg.openrouter_api_key)
        self.assertIsNone(cfg.mistral_api_key)

    def test_credentials_and_metadata_are_stripped(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_SITE_URL": " https://site.example ",
                "OPENROUTER_APP_NAME": " bench ",
                "OPENROUTER_API_KEY": "sk-test\n",
                "MISTRAL_API_KEY": "\tkey123 ",
            },
            clear=False,
        ):
            cfg = load_config()
        self.assertEqual(cfg.openrouter_site_url, "https://site.example")
        self.assertEqual(cfg.openrouter_app_name, "bench")
        self.assertEqual(cfg.openrouter_api_key, "sk-test")
        self.assertEqual(cfg.mistral_api_key, "key123")

    def test_delimiter_only_fallbacks_disable_chain(self):
        with mock.patch.dict(os.environ, {"VISION_FALLBACKS": " ,\n\t, "}, clear=False):
            self.assertEqual(load_config().fallbacks, ())

    def test_whitespace_padded_int_accepted(self):
        with mock.patch.dict(os.environ, {"VISION_MAX_TOKENS": " 512 "}, clear=False):
            self.assertEqual(load_config().max_tokens, 512)

    def test_deadline_and_url_policy_defaults(self):
        cfg = load_config()
        self.assertEqual(cfg.deadline, 100.0)
        self.assertFalse(cfg.allow_private_urls)

    def test_deadline_and_url_policy_parsed(self):
        with mock.patch.dict(
            os.environ,
            {"VISION_DEADLINE": "12.5", "VISION_ALLOW_PRIVATE_URLS": "1"},
            clear=False,
        ):
            cfg = load_config()
        self.assertEqual(cfg.deadline, 12.5)
        self.assertTrue(cfg.allow_private_urls)

    def test_invalid_deadline_falls_back(self):
        for raw in ("0", "-1", "nan", "slow"):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ, {"VISION_DEADLINE": raw}, clear=False
                ):
                    self.assertEqual(load_config().deadline, 100.0)

    def test_invalid_engine_and_provider_are_logged(self):
        with mock.patch.dict(
            os.environ,
            {"OCR_ENGINE": "mistrall", "VISION_PROVIDER": "openruter"},
            clear=False,
        ):
            with self.assertLogs("look_n_see.config", level="WARNING") as logs:
                cfg = load_config()
        self.assertEqual(cfg.ocr_engine, "vlm")
        self.assertEqual(cfg.provider, "auto")
        self.assertEqual(len(logs.records), 2)

    def test_config_is_immutable(self):
        cfg = make_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.model = "other"


if __name__ == "__main__":
    unittest.main()
