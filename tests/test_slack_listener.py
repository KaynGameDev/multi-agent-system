from __future__ import annotations

import unittest
from unittest.mock import patch

from core.config import Settings
from interfaces.slack_listener import SlackListener


class DummyGraph:
    def invoke(self, initial_state, config=None):
        return {"messages": []}


class DummyApp:
    def __init__(self) -> None:
        self.client = object()

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
    )


class SlackListenerTests(unittest.TestCase):
    def setUp(self) -> None:
        app_patcher = patch("interfaces.slack_listener.App", return_value=DummyApp())
        self.addCleanup(app_patcher.stop)
        app_patcher.start()
        self.listener = SlackListener(agent_graph=DummyGraph(), settings=build_settings())

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


if __name__ == "__main__":
    unittest.main()
