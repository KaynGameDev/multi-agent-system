from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app import config as config_module
from app.config import (
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    load_settings,
    validate_core_settings,
)
from app.llm_factory import build_chat_model, build_runtime_llms
from tests.common import make_settings


class LLMConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        config_module._cached_settings = None

    def test_google_provider_uses_legacy_gemini_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEB_ENABLED": "true",
                "SLACK_ENABLED": "false",
                "GOOGLE_API_KEY": "test-google-key",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/credentials.json",
                "JADE_PROJECT_SHEET_ID": "sheet-id",
                "GEMINI_MODEL": "gemini-legacy-test",
                "GEMINI_TEMPERATURE": "0.4",
                "GEMINI_HTTP_TRUST_ENV": "true",
            },
            clear=True,
        ):
            settings = load_settings(force_reload=True)

        self.assertEqual(settings.llm_provider, "google")
        self.assertEqual(settings.llm_model, "gemini-legacy-test")
        self.assertEqual(settings.llm_temperature, 0.4)
        self.assertTrue(settings.llm_http_trust_env)
        self.assertEqual(settings.pending_action_parser_model, "gemini-legacy-test")
        self.assertEqual(settings.pending_action_parser_temperature, 0.0)

    def test_minimax_provider_uses_expected_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEB_ENABLED": "true",
                "SLACK_ENABLED": "false",
                "LLM_PROVIDER": "minimax",
                "MINIMAX_API_KEY": "test-minimax-key",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/credentials.json",
                "JADE_PROJECT_SHEET_ID": "sheet-id",
            },
            clear=True,
        ):
            settings = load_settings(force_reload=True)

        self.assertEqual(settings.llm_provider, "minimax")
        self.assertEqual(settings.llm_model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(settings.minimax_base_url, DEFAULT_MINIMAX_BASE_URL)
        self.assertEqual(settings.pending_action_parser_model, DEFAULT_MINIMAX_MODEL)
        self.assertEqual(settings.pending_action_parser_temperature, 0.01)
        self.assertFalse(settings.llm_http_trust_env)

    def test_openai_provider_uses_expected_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEB_ENABLED": "true",
                "SLACK_ENABLED": "false",
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-openai-key",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/credentials.json",
                "JADE_PROJECT_SHEET_ID": "sheet-id",
            },
            clear=True,
        ):
            settings = load_settings(force_reload=True)

        self.assertEqual(settings.llm_provider, "openai")
        self.assertEqual(settings.llm_model, DEFAULT_OPENAI_MODEL)
        self.assertEqual(settings.openai_base_url, DEFAULT_OPENAI_BASE_URL)
        self.assertEqual(settings.pending_action_parser_model, DEFAULT_OPENAI_MODEL)
        self.assertEqual(settings.pending_action_parser_temperature, 0.0)

    def test_validate_core_settings_requires_only_selected_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            settings = replace(
                make_settings(Path(tempdir)),
                llm_provider="openai",
                google_api_key="",
                openai_api_key="",
            )

        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY") as ctx:
            validate_core_settings(settings)

        self.assertNotIn("GOOGLE_API_KEY", str(ctx.exception))

    def test_validate_core_settings_rejects_explicit_non_positive_minimax_parser_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            settings = replace(
                make_settings(Path(tempdir)),
                llm_provider="minimax",
                google_api_key="",
                pending_action_parser_temperature=0.0,
            )

        with patch.dict(os.environ, {"PENDING_ACTION_PARSER_TEMPERATURE": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "greater than 0"):
                validate_core_settings(settings)


class LLMFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings = make_settings(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_google_factory_builds_chat_google_client(self) -> None:
        built = object()
        with patch("app.llm_factory.ChatGoogleGenerativeAI", return_value=built) as constructor:
            result = build_chat_model(
                self.settings,
                model="gemini-custom",
                temperature=0.3,
            )

        self.assertIs(result, built)
        constructor.assert_called_once_with(
            model="gemini-custom",
            temperature=0.3,
            google_api_key="test-google-api-key",
            client_args={"trust_env": False},
        )

    def test_minimax_factory_builds_chat_anthropic_client(self) -> None:
        built = object()
        settings = replace(self.settings, llm_provider="minimax")
        with patch("app.llm_factory.ChatAnthropic", return_value=built) as constructor:
            result = build_chat_model(
                settings,
                model="MiniMax-M2.7-highspeed",
                temperature=0.2,
            )

        self.assertIs(result, built)
        constructor.assert_called_once_with(
            model="MiniMax-M2.7-highspeed",
            temperature=0.2,
            api_key="test-minimax-api-key",
            base_url="https://api.minimaxi.com/anthropic",
        )

    def test_openai_factory_builds_chat_openai_client(self) -> None:
        built = object()
        settings = replace(self.settings, llm_provider="openai")
        with patch("app.llm_factory.ChatOpenAI", return_value=built) as constructor:
            result = build_chat_model(
                settings,
                model="gpt-5-mini",
                temperature=0.2,
            )

        self.assertIs(result, built)
        constructor.assert_called_once_with(
            model="gpt-5-mini",
            temperature=0.2,
            api_key="test-openai-api-key",
            base_url="https://api.openai.com/v1",
        )

    def test_runtime_factory_builds_primary_and_parser_models(self) -> None:
        primary = object()
        parser = object()
        with patch("app.llm_factory.build_chat_model", side_effect=[primary, parser]) as build_chat_model_mock:
            built_primary, built_parser = build_runtime_llms(self.settings)

        self.assertIs(built_primary, primary)
        self.assertIs(built_parser, parser)
        self.assertEqual(build_chat_model_mock.call_count, 2)
