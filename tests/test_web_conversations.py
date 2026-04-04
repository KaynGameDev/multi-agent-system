from __future__ import annotations

import unittest

from interfaces.web.conversations import normalize_legacy_assistant_markdown


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


if __name__ == "__main__":
    unittest.main()
