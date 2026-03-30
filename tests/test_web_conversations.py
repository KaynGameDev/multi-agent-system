from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.web_conversations import WebConversationStore


class WebConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.storage_path = Path(self.temp_dir.name) / "web_conversations.json"

    def test_persists_conversations_across_store_reloads(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="New chat")
        store.append_message(created["conversation_id"], role="user", markdown="Hello from web")
        store.append_message(created["conversation_id"], role="assistant", markdown="Hi there")

        reloaded_store = WebConversationStore(self.storage_path)
        conversation = reloaded_store.get_conversation(created["conversation_id"])

        self.assertEqual(conversation["title"], "Hello from web")
        self.assertEqual(len(conversation["messages"]), 2)
        self.assertEqual(conversation["messages"][0]["markdown"], "Hello from web")
        self.assertIn("<p>Hello from web</p>", conversation["messages"][0]["html"])
        self.assertEqual(conversation["messages"][1]["markdown"], "Hi there")

    def test_list_conversations_orders_by_updated_at_descending_after_reload(self) -> None:
        store = WebConversationStore(self.storage_path)
        first = store.create_conversation(title="First")
        second = store.create_conversation(title="Second")
        store.append_message(first["conversation_id"], role="user", markdown="Older chat")
        store.append_message(second["conversation_id"], role="user", markdown="Newer chat")

        reloaded_store = WebConversationStore(self.storage_path)
        listed = reloaded_store.list_conversations()

        self.assertEqual([item["conversation_id"] for item in listed], [second["conversation_id"], first["conversation_id"]])


if __name__ == "__main__":
    unittest.main()
