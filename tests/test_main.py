from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
from core.config import Settings


class DummyListener:
    def __init__(self, *, should_interrupt: bool = False) -> None:
        self.should_interrupt = should_interrupt
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        if self.should_interrupt:
            raise KeyboardInterrupt

    def stop(self) -> None:
        self.stopped = True


def build_settings(*, include_slack: bool = True, include_web: bool = False) -> Settings:
    return Settings(
        slack_bot_token="xoxb-test" if include_slack else "",
        slack_app_token="xapp-test" if include_slack else "",
        web_enabled=include_web,
        web_host="127.0.0.1",
        web_port=8000,
        google_api_key="test-key",
        gemini_model="gemini-3-flash-preview",
        gemini_temperature=0.2,
        google_application_credentials="/tmp/fake-creds.json",
        jade_project_sheet_id="sheet-id",
        project_sheet_range="Tasks!A1:Z",
        project_sheet_cache_ttl_seconds=30,
        slack_thinking_reaction="eyes",
        project_lookup_keywords=("task",),
        knowledge_base_dir="/tmp/knowledge",
        knowledge_file_types=(".md", ".xlsx"),
    )


class MainTests(unittest.TestCase):
    def test_main_handles_keyboard_interrupt_cleanly(self) -> None:
        stdout = io.StringIO()
        with patch("main.bootstrap_system", return_value=[DummyListener(should_interrupt=True)]):
            with redirect_stdout(stdout):
                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Stopping Jade Agent...", stdout.getvalue())

    def test_main_stops_all_listeners_after_interrupt(self) -> None:
        primary_listener = DummyListener()
        secondary_listener = DummyListener(should_interrupt=True)

        with patch("main.bootstrap_system", return_value=[primary_listener, secondary_listener]):
            with patch("main._start_background_listeners", return_value=[]):
                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        self.assertTrue(primary_listener.stopped)
        self.assertTrue(secondary_listener.stopped)

    def test_bootstrap_system_builds_enabled_listener(self) -> None:
        settings = build_settings(include_slack=True)
        slack_listener = DummyListener()

        with patch("main.load_dotenv"):
            with patch("main.load_settings", return_value=settings):
                with patch("main.validate_bootstrap_settings"):
                    with patch("main.ChatGoogleGenerativeAI", return_value=object()) as llm_ctor:
                        with patch("main.build_agent_graph", return_value=object()):
                            with patch("main.SlackListener", return_value=slack_listener):
                                listeners = main.bootstrap_system()

        self.assertEqual(listeners, [slack_listener])
        llm_ctor.assert_called_once_with(
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            client_args={"trust_env": settings.gemini_http_trust_env},
        )

    def test_bootstrap_system_builds_web_listener_when_enabled(self) -> None:
        settings = build_settings(include_slack=False, include_web=True)
        web_listener = DummyListener()

        with patch("main.load_dotenv"):
            with patch("main.load_settings", return_value=settings):
                with patch("main.validate_bootstrap_settings"):
                    with patch("main.ChatGoogleGenerativeAI", return_value=object()):
                        with patch("main.build_agent_graph", return_value=object()):
                            with patch("main.WebServer", return_value=web_listener):
                                listeners = main.bootstrap_system()

        self.assertEqual(listeners, [web_listener])

    def test_bootstrap_system_builds_both_slack_and_web_listeners(self) -> None:
        settings = build_settings(include_slack=True, include_web=True)
        slack_listener = DummyListener()
        web_listener = DummyListener()

        with patch("main.load_dotenv"):
            with patch("main.load_settings", return_value=settings):
                with patch("main.validate_bootstrap_settings"):
                    with patch("main.ChatGoogleGenerativeAI", return_value=object()):
                        with patch("main.build_agent_graph", return_value=object()):
                            with patch("main.SlackListener", return_value=slack_listener):
                                with patch("main.WebServer", return_value=web_listener):
                                    listeners = main.bootstrap_system()

        self.assertEqual(listeners, [slack_listener, web_listener])


if __name__ == "__main__":
    unittest.main()
