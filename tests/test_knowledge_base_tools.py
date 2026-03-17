from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook

from tools.knowledge_base import list_knowledge_documents, read_knowledge_document, search_knowledge_documents


class KnowledgeBaseToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.knowledge_root = Path(self.temp_dir.name)

        (self.knowledge_root / "architecture.md").write_text(
            "\n".join(
                [
                    "# Jade MAS",
                    "",
                    "## Overview",
                    "This system uses multiple specialist agents.",
                    "",
                    "## Formatting Strategy",
                    "Layer 1: Tools output structured data.",
                    "Layer 2: Agents reason over the data.",
                    "Layer 3: Slack formatting happens at the boundary.",
                ]
            ),
            encoding="utf-8",
        )

        workbook = Workbook()
        summary_sheet = workbook.active
        summary_sheet.title = "Summary"
        summary_sheet.append(["Topic", "Details", "Owner"])
        summary_sheet.append(["Onboarding", "Use the VPN before opening Slack.", "IT"])
        summary_sheet.append(["Build", "Run python main.py from the project root.", "Engineering"])

        routing_sheet = workbook.create_sheet("Routing")
        routing_sheet.append(["Question Type", "Agent"])
        routing_sheet.append(["Project tracker", "project_task_agent"])
        routing_sheet.append(["Documentation", "knowledge_agent"])

        workbook.save(self.knowledge_root / "operations.xlsx")

        self.settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".xlsx"),
        )

    def test_list_knowledge_documents_uses_local_knowledge_directory(self) -> None:
        with patch("tools.knowledge_base.load_settings", return_value=self.settings):
            result = list_knowledge_documents.invoke({})

        self.assertTrue(result["ok"])
        self.assertEqual(result["document_count"], 2)
        self.assertEqual(result["knowledge_base_dir"], str(self.knowledge_root.resolve()))
        self.assertEqual(
            [document["name"] for document in result["documents"]],
            ["architecture.md", "operations.xlsx"],
        )

    def test_search_knowledge_documents_finds_spreadsheet_content(self) -> None:
        with patch("tools.knowledge_base.load_settings", return_value=self.settings):
            result = search_knowledge_documents.invoke({"query": "VPN opening", "limit": 5})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["documents"][0]["name"], "operations.xlsx")
        self.assertIn("Use the VPN before opening Slack.", result["documents"][0]["snippet"])

    def test_read_knowledge_document_reads_matching_sheet_section(self) -> None:
        with patch("tools.knowledge_base.load_settings", return_value=self.settings):
            result = read_knowledge_document.invoke(
                {
                    "document_name": "operations",
                    "section_query": "Routing",
                    "max_lines": 10,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["document"]["name"], "operations.xlsx")
        self.assertIn("## Sheet: Routing", result["content"])
        self.assertIn("knowledge_agent", result["content"])


if __name__ == "__main__":
    unittest.main()
