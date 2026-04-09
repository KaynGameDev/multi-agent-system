from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interfaces.web.conversations import WebConversationStore, normalize_legacy_assistant_markdown


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


if __name__ == "__main__":
    unittest.main()
