from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from interfaces.web.conversations import (
    TRANSCRIPT_SCHEMA_VERSION,
    TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
    TRANSCRIPT_TYPE_MESSAGE,
    WebConversationStore,
    normalize_legacy_assistant_markdown,
    transcript_to_langchain_messages,
)


class WebConversationNormalizationTests(unittest.TestCase):
    def test_legacy_kb_write_denial_is_rewritten_for_history(self) -> None:
        legacy_markdown = (
            "我目前拥有**读取**和**检索**公司知识库的权限，但**没有直接修改或创建物理文件**的权限。\n\n"
            "#### 文档 1：业务组概览\n"
            "**建议路径：** `Docs/10_Projects/Shooting_TowerDefense_Group.md`\n"
        )

        normalized = normalize_legacy_assistant_markdown(legacy_markdown)

        self.assertNotIn("没有直接修改或创建物理文件", normalized)
        self.assertIn("旧版本保存的历史回复", normalized)
        self.assertIn("历史旧路径（已过时）", normalized)


class WebConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.storage_path = Path(self.temp_dir.name) / "web_conversations.json"

    def test_rename_persists_across_store_reload(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="Original")

        renamed = store.rename_conversation(created["conversation_id"], title="Renamed chat")
        reloaded_store = WebConversationStore(self.storage_path)
        loaded = reloaded_store.get_conversation(created["conversation_id"])

        self.assertEqual(renamed["title"], "Renamed chat")
        self.assertEqual(loaded["title"], "Renamed chat")

    def test_rename_normalizes_blank_titles_to_new_chat(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="Original")

        renamed = store.rename_conversation(created["conversation_id"], title="   ")

        self.assertEqual(renamed["title"], "New chat")

    def test_delete_persists_across_store_reload(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="Disposable")

        store.delete_conversation(created["conversation_id"])
        reloaded_store = WebConversationStore(self.storage_path)

        with self.assertRaises(KeyError):
            reloaded_store.get_conversation(created["conversation_id"])
        self.assertEqual(reloaded_store.list_conversations(), [])

    def test_normal_conversation_persists_and_reloads_visible_messages_unchanged(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="Persistence")

        after_user = store.append_message(
            created["conversation_id"],
            role="user",
            markdown="hello there",
        )
        after_assistant = store.append_message(
            created["conversation_id"],
            role="assistant",
            markdown="General Kenobi.",
        )
        reloaded_store = WebConversationStore(self.storage_path)
        reloaded = reloaded_store.get_conversation(created["conversation_id"])

        self.assertEqual([message["role"] for message in after_assistant["messages"]], ["user", "assistant"])
        self.assertEqual([message["type"] for message in after_assistant["messages"]], ["message", "message"])
        self.assertEqual(
            [(message["id"], message["created_at"], message["markdown"]) for message in reloaded["messages"]],
            [(message["id"], message["created_at"], message["markdown"]) for message in after_assistant["messages"]],
        )
        self.assertEqual(reloaded["messages"][0]["html"], after_user["messages"][0]["html"])

        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], TRANSCRIPT_SCHEMA_VERSION)
        self.assertEqual([message["type"] for message in payload["conversations"][0]["messages"]], ["message", "message"])

    def test_compact_boundary_round_trips_in_full_transcript_but_stays_hidden_publicly(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="Boundary")

        store.append_message(created["conversation_id"], role="user", markdown="Before compacting.")
        store.append_compact_boundary(
            created["conversation_id"],
            trigger="manual_test",
            pre_tokens=2048,
            preserved_tail={"message_ids": ["assistant-1", "assistant-2"]},
        )
        store.append_transcript_message(
            created["conversation_id"],
            role="assistant",
            message_type=TRANSCRIPT_TYPE_MESSAGE,
            markdown="After compacting.",
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

        public_conversation = store.get_conversation(created["conversation_id"])
        full_conversation = store.get_full_conversation(created["conversation_id"])
        reloaded_store = WebConversationStore(self.storage_path)
        reloaded_full = reloaded_store.get_full_conversation(created["conversation_id"])

        self.assertEqual(len(public_conversation["messages"]), 2)
        self.assertTrue(all(message["type"] == TRANSCRIPT_TYPE_MESSAGE for message in public_conversation["messages"]))

        self.assertEqual(
            [(message["role"], message["type"]) for message in full_conversation["messages"]],
            [
                ("user", TRANSCRIPT_TYPE_MESSAGE),
                ("system", TRANSCRIPT_TYPE_COMPACT_BOUNDARY),
                ("assistant", TRANSCRIPT_TYPE_MESSAGE),
            ],
        )
        boundary = reloaded_full["messages"][1]
        self.assertEqual(boundary["role"], "system")
        self.assertEqual(boundary["type"], TRANSCRIPT_TYPE_COMPACT_BOUNDARY)
        self.assertEqual(boundary["metadata"]["trigger"], "manual_test")
        self.assertEqual(boundary["metadata"]["preTokens"], 2048)
        self.assertEqual(
            boundary["metadata"]["preservedTail"],
            {"message_ids": ["assistant-1", "assistant-2"]},
        )

    def test_transcript_to_langchain_messages_preserves_ids_and_boundary_metadata(self) -> None:
        store = WebConversationStore(self.storage_path)
        created = store.create_conversation(title="Rehydrate")

        store.append_transcript_message(
            created["conversation_id"],
            role="user",
            message_type=TRANSCRIPT_TYPE_MESSAGE,
            markdown="Need a rebuild.",
            message_id="user-1",
            created_at="2026-04-16T00:00:00+00:00",
        )
        store.append_compact_boundary(
            created["conversation_id"],
            trigger="token_limit",
            pre_tokens=8192,
            preserved_tail={"tail_length": 4},
        )
        store.append_transcript_message(
            created["conversation_id"],
            role="assistant",
            message_type=TRANSCRIPT_TYPE_MESSAGE,
            markdown="Rebuild ready.",
            usage={"input_tokens": 55, "output_tokens": 13, "total_tokens": 68},
            message_id="assistant-1",
            created_at="2026-04-16T00:00:02+00:00",
        )

        full_conversation = store.get_full_conversation(created["conversation_id"])
        converted = transcript_to_langchain_messages(full_conversation["messages"])

        self.assertEqual(len(converted), 3)
        self.assertTrue(isinstance(converted[0], HumanMessage))
        self.assertTrue(isinstance(converted[1], SystemMessage))
        self.assertTrue(isinstance(converted[2], AIMessage))
        self.assertEqual(converted[0].id, "user-1")
        self.assertEqual(converted[2].id, "assistant-1")
        self.assertEqual(converted[2].usage_metadata, {"input_tokens": 55, "output_tokens": 13, "total_tokens": 68})
        self.assertEqual(converted[1].additional_kwargs["transcript_type"], TRANSCRIPT_TYPE_COMPACT_BOUNDARY)
        self.assertEqual(
            converted[1].additional_kwargs["metadata"],
            {"trigger": "token_limit", "preTokens": 8192, "preservedTail": {"tail_length": 4}},
        )


if __name__ == "__main__":
    unittest.main()
