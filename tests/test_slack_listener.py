from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from core.config import Settings
from interfaces.slack_listener import SlackListener
from tools.document_conversion import UPLOAD_ONLY_FALLBACK_TEXT


class DummyGraph:
    def __init__(self) -> None:
        self.invocations: list[dict] = []

    def invoke(self, initial_state, config=None):
        self.invocations.append({"initial_state": initial_state, "config": config})
        return {"messages": []}


class DummyClient:
    def __init__(self) -> None:
        self.views_publish_calls = []

    def views_publish(self, *, user_id, view):
        self.views_publish_calls.append({"user_id": user_id, "view": view})

    def users_info(self, *, user):
        return {
            "user": {
                "name": "kayn",
                "real_name": "Kayn",
                "profile": {
                    "display_name": "Kayn",
                    "real_name": "Kayn",
                    "email": "",
                },
            }
        }


class DummyApp:
    def __init__(self) -> None:
        self.client = DummyClient()

    def event(self, _name: str):
        def register(listener):
            return listener

        return register


def build_settings() -> Settings:
    return Settings(
        slack_bot_token="xoxb-test",
        slack_app_token="xapp-test",
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
        conversion_work_dir="/tmp/jade_conversion_tests",
    )


class SlackListenerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        app_patcher = patch("interfaces.slack_listener.App", return_value=DummyApp())
        self.addCleanup(app_patcher.stop)
        app_patcher.start()
        settings = build_settings()
        settings = Settings(
            **{
                **settings.__dict__,
                "conversion_work_dir": self.temp_dir.name,
            }
        )
        self.listener = SlackListener(agent_graph=DummyGraph(), settings=settings)

    def test_build_thread_id_uses_channel_for_direct_messages(self) -> None:
        event = {
            "channel": "D123",
            "channel_type": "im",
            "ts": "1710.100",
        }

        self.assertEqual(self.listener._build_thread_id(event), "D123")

    def test_build_thread_id_uses_root_thread_for_channel_threads(self) -> None:
        event = {
            "channel": "C123",
            "channel_type": "channel",
            "thread_ts": "1710.200",
            "ts": "1710.300",
        }

        self.assertEqual(self.listener._build_thread_id(event), "C123:1710.200")

    def test_build_reply_thread_ts_starts_thread_for_channel_mentions(self) -> None:
        event = {
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1710.400",
        }

        self.assertEqual(self.listener._build_reply_thread_ts(event), "1710.400")

    def test_build_reply_thread_ts_is_none_for_direct_messages(self) -> None:
        event = {
            "channel": "D123",
            "channel_type": "im",
            "ts": "1710.500",
        }

        self.assertIsNone(self.listener._build_reply_thread_ts(event))

    def test_handle_app_home_opened_publishes_home_view(self) -> None:
        event = {
            "user": "U123",
        }

        self.listener.handle_app_home_opened(event)

        self.assertEqual(len(self.listener.app.client.views_publish_calls), 1)
        published = self.listener.app.client.views_publish_calls[0]
        self.assertEqual(published["user_id"], "U123")
        self.assertEqual(published["view"]["type"], "home")

    def test_process_and_respond_sets_slack_interface_name(self) -> None:
        say_calls = []

        def say(**kwargs):
            say_calls.append(kwargs)

        event = {
            "user": "U123",
            "channel": "D123",
            "channel_type": "im",
            "ts": "1710.600",
            "text": "hello",
        }

        self.listener.process_and_respond(event=event, say=say, is_mention=False)

        self.assertEqual(len(say_calls), 1)
        self.assertEqual(len(self.listener.agent_graph.invocations), 1)
        invocation = self.listener.agent_graph.invocations[0]
        self.assertEqual(invocation["initial_state"]["interface_name"], "slack")

    def test_process_and_respond_routes_file_share_to_document_conversion_agent(self) -> None:
        say_calls = []

        def say(**kwargs):
            say_calls.append(kwargs)

        event = {
            "user": "U123",
            "channel": "D123",
            "channel_type": "im",
            "ts": "1710.601",
            "text": "",
            "subtype": "file_share",
            "files": [
                {
                    "id": "F123",
                    "name": "design.md",
                    "mimetype": "text/markdown",
                    "filetype": "md",
                    "url_private": "https://example.com/design.md",
                    "url_private_download": "https://example.com/design.md",
                }
            ],
        }

        self.listener.process_and_respond(event=event, say=say, is_mention=False)

        invocation = self.listener.agent_graph.invocations[0]
        self.assertEqual(invocation["initial_state"]["route"], "document_conversion_agent")
        self.assertEqual(invocation["initial_state"]["messages"][0].content, UPLOAD_ONLY_FALLBACK_TEXT)
        self.assertEqual(invocation["initial_state"]["uploaded_files"][0]["id"], "F123")
        self.assertEqual(len(say_calls), 1)

    def test_handle_message_event_processes_channel_reply_for_active_conversion_session(self) -> None:
        say_calls = []

        def say(**kwargs):
            say_calls.append(kwargs)

        self.listener.conversion_store.create_session(
            thread_id="C123:1710.700",
            channel_id="C123",
            user_id="U123",
        )
        event = {
            "user": "U123",
            "channel": "C123",
            "channel_type": "channel",
            "thread_ts": "1710.700",
            "ts": "1710.701",
            "text": "Here is the missing market info",
        }

        self.listener.handle_message_event(event=event, say=say)

        invocation = self.listener.agent_graph.invocations[0]
        self.assertEqual(invocation["initial_state"]["route"], "document_conversion_agent")
        self.assertTrue(invocation["initial_state"]["conversion_session_id"])
        self.assertEqual(len(say_calls), 1)


if __name__ == "__main__":
    unittest.main()
