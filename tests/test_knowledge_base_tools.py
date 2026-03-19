from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpyxl import Workbook

from tools import knowledge_base
from tools.knowledge_base import list_knowledge_documents, read_knowledge_document, search_knowledge_documents


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeValuesApi:
    def __init__(self, value_ranges_by_sheet_id):
        self.value_ranges_by_sheet_id = value_ranges_by_sheet_id

    def batchGet(self, spreadsheetId, ranges):
        sheet_values = self.value_ranges_by_sheet_id.get(spreadsheetId, {})
        return FakeRequest(
            {
                "valueRanges": [
                    {"range": requested_range, "values": sheet_values.get(requested_range, [])}
                    for requested_range in ranges
                ]
            }
        )


class FakeSpreadsheetsApi:
    def __init__(self, metadata_by_sheet_id, value_ranges_by_sheet_id):
        self.metadata_by_sheet_id = metadata_by_sheet_id
        self.value_ranges_by_sheet_id = value_ranges_by_sheet_id

    def get(self, spreadsheetId):
        return FakeRequest(self.metadata_by_sheet_id[spreadsheetId])

    def values(self):
        return FakeValuesApi(self.value_ranges_by_sheet_id)


class FakeGoogleSheetsService:
    def __init__(self, metadata_by_sheet_id, value_ranges_by_sheet_id):
        self.metadata_by_sheet_id = metadata_by_sheet_id
        self.value_ranges_by_sheet_id = value_ranges_by_sheet_id

    def spreadsheets(self):
        return FakeSpreadsheetsApi(self.metadata_by_sheet_id, self.value_ranges_by_sheet_id)


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

        (self.knowledge_root / "feature_spec.csv").write_text(
            "\n".join(
                [
                    "前言,,,,",
                    "设计目的,,,,",
                    ",·提升活跃付费率，提高付费,,,",
                    "服务器,,,,",
                    ",·礼包配置,,,",
                    ",,礼包奖励发放规则：,,",
                    ",,,1.礼包奖励由保底奖励和暴击奖励组成,",
                    ",,每日限购：,,",
                    ",,,1.每日限购5次（配置），次日0点重置,",
                    ",,礼包名称,钻石,金币,限购次数",
                    ",,精品,143,160000,5",
                    ",,进阶,330,380000,5",
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

        self.catalog_path = self.knowledge_root / "google_sheets_catalog.json"
        self.catalog_path.write_text(
            """
            {
              "documents": [
                {
                  "spreadsheet_id": "remote-sheet-1",
                  "title": "Live Design Docs",
                  "aliases": ["live design", "design docs"],
                  "tabs": ["Overview", "Rewards"]
                }
              ]
            }
            """.strip(),
            encoding="utf-8",
        )

        self.settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".csv", ".xlsx"),
            knowledge_google_sheets_catalog_path=str(self.catalog_path),
            knowledge_google_sheets_cache_ttl_seconds=120,
        )
        knowledge_base._google_sheet_index_cache.clear()
        self.addCleanup(knowledge_base._google_sheet_index_cache.clear)

        self.fake_google_sheets_service = FakeGoogleSheetsService(
            metadata_by_sheet_id={
                "remote-sheet-1": {
                    "properties": {"title": "Live Design Docs"},
                    "sheets": [
                        {"properties": {"title": "Overview"}},
                        {"properties": {"title": "Rewards"}},
                    ],
                }
            },
            value_ranges_by_sheet_id={
                "remote-sheet-1": {
                    "'Overview'": [
                        ["Section"],
                        ["Overview"],
                        ["Core loop"],
                        ["Spin the wheel and grant package rewards."],
                    ],
                    "'Rewards'": [
                        ["Rule", "Details"],
                        ["Daily purchase limit", "5 times per day"],
                        ["Reward formula", "Base reward + critical multiplier"],
                    ],
                }
            },
        )

    def test_list_knowledge_documents_uses_local_knowledge_directory(self) -> None:
        local_only_settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".csv", ".xlsx"),
            knowledge_google_sheets_catalog_path=str(self.knowledge_root / "missing.json"),
            knowledge_google_sheets_cache_ttl_seconds=120,
        )
        with patch("tools.knowledge_base.load_settings", return_value=local_only_settings):
            result = list_knowledge_documents.invoke({})

        self.assertTrue(result["ok"])
        self.assertEqual(result["document_count"], 3)
        self.assertEqual(result["knowledge_base_dir"], str(self.knowledge_root.resolve()))
        self.assertEqual(
            [document["name"] for document in result["documents"]],
            ["architecture.md", "feature_spec.csv", "operations.xlsx"],
        )

    def test_list_knowledge_documents_includes_cataloged_google_sheet(self) -> None:
        with patch("tools.knowledge_base.load_settings", return_value=self.settings):
            result = list_knowledge_documents.invoke({})

        self.assertTrue(result["ok"])
        self.assertEqual(result["document_count"], 4)
        remote_document = next(
            document for document in result["documents"] if document["file_type"] == "google_sheet"
        )
        self.assertEqual(remote_document["title"], "Live Design Docs")
        self.assertEqual(remote_document["spreadsheet_id"], "remote-sheet-1")

    def test_search_knowledge_documents_finds_spreadsheet_content(self) -> None:
        local_only_settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".csv", ".xlsx"),
            knowledge_google_sheets_catalog_path=str(self.knowledge_root / "missing.json"),
            knowledge_google_sheets_cache_ttl_seconds=120,
        )
        with patch("tools.knowledge_base.load_settings", return_value=local_only_settings):
            result = search_knowledge_documents.invoke({"query": "VPN opening", "limit": 5})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["documents"][0]["name"], "operations.xlsx")
        self.assertIn("Use the VPN before opening Slack.", result["documents"][0]["snippet"])
        self.assertEqual(result["documents"][0]["block_type"], "table")
        self.assertEqual(result["documents"][0]["section_title"], "Sheet: Summary")

    def test_read_knowledge_document_reads_matching_sheet_section(self) -> None:
        local_only_settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".csv", ".xlsx"),
            knowledge_google_sheets_catalog_path=str(self.knowledge_root / "missing.json"),
            knowledge_google_sheets_cache_ttl_seconds=120,
        )
        with patch("tools.knowledge_base.load_settings", return_value=local_only_settings):
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
        self.assertEqual(result["block_type"], "table")
        self.assertEqual(result["section_title"], "Sheet: Routing")

    def test_read_knowledge_document_renders_sparse_csv_sections_semantically(self) -> None:
        local_only_settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".csv", ".xlsx"),
            knowledge_google_sheets_catalog_path=str(self.knowledge_root / "missing.json"),
            knowledge_google_sheets_cache_ttl_seconds=120,
        )
        with patch("tools.knowledge_base.load_settings", return_value=local_only_settings):
            result = read_knowledge_document.invoke(
                {
                    "document_name": "feature_spec",
                    "section_query": "礼包奖励发放规则",
                    "max_lines": 20,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["document"]["name"], "feature_spec.csv")
        self.assertIn("### 礼包奖励发放规则", result["content"])
        self.assertIn("1.礼包奖励由保底奖励和暴击奖励组成", result["content"])
        self.assertNotIn("column_", result["content"])
        self.assertEqual(result["block_type"], "list")
        self.assertEqual(result["section_title"], "礼包奖励发放规则")

    def test_search_knowledge_documents_uses_semantic_csv_snippets(self) -> None:
        local_only_settings = SimpleNamespace(
            knowledge_base_dir=str(self.knowledge_root),
            knowledge_file_types=(".md", ".csv", ".xlsx"),
            knowledge_google_sheets_catalog_path=str(self.knowledge_root / "missing.json"),
            knowledge_google_sheets_cache_ttl_seconds=120,
        )
        with patch("tools.knowledge_base.load_settings", return_value=local_only_settings):
            result = search_knowledge_documents.invoke({"query": "每日限购", "limit": 5})

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["match_count"], 1)
        self.assertIn("### 每日限购", result["documents"][0]["snippet"])
        self.assertNotIn("column_", result["documents"][0]["snippet"])
        self.assertIn(result["documents"][0]["block_type"], {"list", "table"})
        self.assertEqual(result["documents"][0]["section_title"], "每日限购")

    def test_search_knowledge_documents_finds_cataloged_google_sheet_content(self) -> None:
        with (
            patch("tools.knowledge_base.load_settings", return_value=self.settings),
            patch("tools.knowledge_base.get_google_sheets_service", return_value=self.fake_google_sheets_service),
        ):
            result = search_knowledge_documents.invoke({"query": "critical multiplier", "limit": 5})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["documents"][0]["title"], "Live Design Docs")
        self.assertEqual(result["documents"][0]["file_type"], "google_sheet")
        self.assertIn("critical multiplier", result["documents"][0]["snippet"].lower())
        self.assertEqual(result["documents"][0]["section_title"], "Sheet: Rewards")

    def test_read_knowledge_document_reads_cataloged_google_sheet(self) -> None:
        with (
            patch("tools.knowledge_base.load_settings", return_value=self.settings),
            patch("tools.knowledge_base.get_google_sheets_service", return_value=self.fake_google_sheets_service),
        ):
            result = read_knowledge_document.invoke(
                {
                    "document_name": "live design docs",
                    "section_query": "Reward Rules",
                    "max_lines": 10,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["document"]["title"], "Live Design Docs")
        self.assertEqual(result["document"]["file_type"], "google_sheet")
        self.assertIn("## Sheet: Rewards", result["content"])
        self.assertIn("critical multiplier", result["content"].lower())
        self.assertEqual(result["section_title"], "Sheet: Rewards")


if __name__ == "__main__":
    unittest.main()
