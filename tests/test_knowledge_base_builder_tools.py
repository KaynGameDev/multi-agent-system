from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.config import load_settings
from app.graph import build_default_agent_registrations
from app.knowledge_paths import build_knowledge_markdown_relative_path
from tools.knowledge_base import resolve_knowledge_markdown_path, write_knowledge_markdown_document


class KnowledgeBaseBuilderToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.kb_root = Path(self.tempdir.name) / "knowledge"
        self.previous_knowledge_base_dir = os.environ.get("KNOWLEDGE_BASE_DIR")
        self.previous_catalog_path = os.environ.get("KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH")
        os.environ["KNOWLEDGE_BASE_DIR"] = str(self.kb_root)
        os.environ["KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH"] = str(self.kb_root / "AI" / "Rules" / "google_sheets_catalog.json")
        load_settings(force_reload=True)

    def tearDown(self) -> None:
        if self.previous_knowledge_base_dir is None:
            os.environ.pop("KNOWLEDGE_BASE_DIR", None)
        else:
            os.environ["KNOWLEDGE_BASE_DIR"] = self.previous_knowledge_base_dir

        if self.previous_catalog_path is None:
            os.environ.pop("KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH", None)
        else:
            os.environ["KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH"] = self.previous_catalog_path

        load_settings(force_reload=True)
        self.tempdir.cleanup()

    def test_game_line_path_uses_current_hierarchy(self) -> None:
        relative_path = build_knowledge_markdown_relative_path(
            layer="game_line",
            category="line_overview",
            game_slug="BuYuDaLuanDou",
            filename="Shooting_TowerDefense_Group_Overview.md",
        )

        self.assertEqual(
            relative_path,
            "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Shooting_TowerDefense_Group_Overview.md",
        )

    def test_deployment_feature_path_uses_canonical_package_layout(self) -> None:
        relative_path = build_knowledge_markdown_relative_path(
            layer="deployment",
            category="feature",
            game_slug="BuYuDaLuanDou",
            market_slug="IndonesiaMain",
            feature_slug="daily-reward",
        )

        self.assertEqual(
            relative_path,
            "Docs/20_Deployments/IndonesiaMain/BuYuDaLuanDou/Features/daily-reward/README.md",
        )

    def test_builder_tools_can_resolve_and_write_markdown(self) -> None:
        resolved = resolve_knowledge_markdown_path.invoke(
            {
                "layer": "game_line",
                "category": "line_overview",
                "game_slug": "BuYuDaLuanDou",
                "filename": "Shooting_TowerDefense_Group_Overview.md",
            }
        )
        self.assertTrue(resolved["ok"])
        self.assertEqual(
            resolved["relative_path"],
            "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Shooting_TowerDefense_Group_Overview.md",
        )

        written = write_knowledge_markdown_document.invoke(
            {
                "relative_path": resolved["relative_path"],
                "content": "# 射击塔防组概览\n\n测试内容。\n",
            }
        )
        self.assertTrue(written["ok"])
        self.assertTrue(written["created"])
        output_path = Path(written["absolute_path"])
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.read_text(encoding="utf-8"), "# 射击塔防组概览\n\n测试内容。\n")

        second_write = write_knowledge_markdown_document.invoke(
            {
                "relative_path": resolved["relative_path"],
                "content": "# 新版本\n",
            }
        )
        self.assertFalse(second_write["ok"])
        self.assertIn("already exists", second_write["error"])

    def test_builder_agent_gets_write_tools_but_reader_does_not(self) -> None:
        registrations = {registration.name: registration for registration in build_default_agent_registrations()}

        knowledge_tool_names = {tool.name for tool in registrations["knowledge_agent"].tools}
        builder_tool_names = {tool.name for tool in registrations["knowledge_base_builder_agent"].tools}

        self.assertEqual(
            knowledge_tool_names,
            {"list_knowledge_documents", "search_knowledge_documents", "read_knowledge_document"},
        )
        self.assertEqual(
            builder_tool_names,
            {
                "list_knowledge_documents",
                "search_knowledge_documents",
                "read_knowledge_document",
                "resolve_knowledge_markdown_path",
                "write_knowledge_markdown_document",
            },
        )


if __name__ == "__main__":
    unittest.main()
