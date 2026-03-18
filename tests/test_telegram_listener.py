from __future__ import annotations

from types import SimpleNamespace
import unittest

from core.config import Settings
from interfaces.telegram_listener import TelegramListener


class DummyGraph:
    def __init__(self, response_text: str = "Hello from Telegram") -> None:
        self.response_text = response_text
        self.invocations: list[dict] = []

    def invoke(self, initial_state, config=None):
        self.invocations.append({"initial_state": initial_state, "config": config})
        return {"messages": [SimpleNamespace(content=self.response_text)]}


class FakeTelegramListener(TelegramListener):
    def __init__(self, agent_graph, settings: Settings) -> None:
        super().__init__(agent_graph=agent_graph, settings=settings)
        self.sent_payloads: list[dict] = []

    def _api_request(self, method: str, payload=None, *, timeout: float = 20):
        if method == "deleteWebhook":
            return True
        if method == "getMe":
            return {"id": 999, "username": "jade_bot"}
        if method == "sendMessage":
            payload = dict(payload or {})
            self.sent_payloads.append(payload)
            return {"message_id": 100 + len(self.sent_payloads)}
        if method == "getUpdates":
            return []
        raise AssertionError(f"Unexpected Telegram API method in test: {method}")


def build_settings(*, allowed_chat_ids: tuple[str, ...] = ()) -> Settings:
    return Settings(
        slack_bot_token="",
        slack_app_token="",
        telegram_bot_token="telegram-token",
        telegram_allowed_chat_ids=allowed_chat_ids,
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


class TelegramListenerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = DummyGraph()
        self.listener = FakeTelegramListener(self.graph, build_settings())
        self.listener._ensure_bot_identity()

    def test_private_message_invokes_graph_and_replies(self) -> None:
        update = {
            "update_id": 1,
            "message": {
                "message_id": 10,
                "text": "What are my tasks?",
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 42, "username": "kayn", "first_name": "Kayn"},
            },
        }

        self.listener.handle_update(update)

        self.assertEqual(len(self.graph.invocations), 1)
        invocation = self.graph.invocations[0]
        self.assertEqual(invocation["initial_state"]["messages"][0].content, "What are my tasks?")
        self.assertEqual(invocation["initial_state"]["interface_name"], "telegram")
        self.assertEqual(invocation["config"]["configurable"]["thread_id"], "telegram:12345")
        self.assertEqual(len(self.listener.sent_payloads), 1)
        self.assertEqual(self.listener.sent_payloads[0]["chat_id"], "12345")
        self.assertEqual(self.listener.sent_payloads[0]["text"], "Hello from Telegram")
        self.assertEqual(self.listener.sent_payloads[0]["parse_mode"], "HTML")

    def test_group_message_without_direct_address_is_ignored(self) -> None:
        update = {
            "update_id": 2,
            "message": {
                "message_id": 20,
                "text": "What are my tasks?",
                "chat": {"id": -10001, "type": "supergroup"},
                "from": {"id": 42, "username": "kayn", "first_name": "Kayn"},
            },
        }

        self.listener.handle_update(update)

        self.assertEqual(len(self.graph.invocations), 0)
        self.assertEqual(len(self.listener.sent_payloads), 0)

    def test_group_follow_up_reply_reuses_prior_thread_id(self) -> None:
        first_update = {
            "update_id": 3,
            "message": {
                "message_id": 30,
                "text": "@jade_bot Which tasks are due this week?",
                "chat": {"id": -10001, "type": "supergroup"},
                "from": {"id": 42, "username": "kayn", "first_name": "Kayn"},
            },
        }
        second_update = {
            "update_id": 4,
            "message": {
                "message_id": 31,
                "text": "What are those tasks?",
                "chat": {"id": -10001, "type": "supergroup"},
                "from": {"id": 42, "username": "kayn", "first_name": "Kayn"},
                "reply_to_message": {
                    "message_id": 101,
                    "from": {"id": 999, "is_bot": True, "username": "jade_bot"},
                },
            },
        }

        self.listener.handle_update(first_update)
        self.listener.handle_update(second_update)

        self.assertEqual(len(self.graph.invocations), 2)
        first_thread_id = self.graph.invocations[0]["config"]["configurable"]["thread_id"]
        second_thread_id = self.graph.invocations[1]["config"]["configurable"]["thread_id"]
        self.assertEqual(first_thread_id, "telegram:-10001:message:30")
        self.assertEqual(second_thread_id, first_thread_id)
        self.assertEqual(
            self.graph.invocations[1]["initial_state"]["messages"][0].content,
            "What are those tasks?",
        )

    def test_help_command_replies_without_invoking_graph(self) -> None:
        update = {
            "update_id": 5,
            "message": {
                "message_id": 40,
                "text": "/help",
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 42, "username": "kayn", "first_name": "Kayn"},
            },
        }

        self.listener.handle_update(update)

        self.assertEqual(len(self.graph.invocations), 0)
        self.assertEqual(len(self.listener.sent_payloads), 1)
        self.assertIn("Ask me in a private chat", self.listener.sent_payloads[0]["text"])

    def test_allowed_chat_ids_filter_blocks_unlisted_chat(self) -> None:
        listener = FakeTelegramListener(self.graph, build_settings(allowed_chat_ids=("99999",)))
        listener._ensure_bot_identity()
        update = {
            "update_id": 6,
            "message": {
                "message_id": 50,
                "text": "What are my tasks?",
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 42, "username": "kayn", "first_name": "Kayn"},
            },
        }

        listener.handle_update(update)

        self.assertEqual(len(self.graph.invocations), 0)
        self.assertEqual(len(listener.sent_payloads), 0)


if __name__ == "__main__":
    unittest.main()
