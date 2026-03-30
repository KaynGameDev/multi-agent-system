from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agents.workers.document_conversion_agent import DocumentConversionAgentNode
from core.config import Settings
from tools.conversion_retrieval import (
    build_conversion_chunks,
    build_conversion_queries,
    build_retrieved_source_bundle,
    retrieve_conversion_chunks,
)
from tools.document_conversion import ConversionSourceRecord


class RecordingStructuredExtractor:
    def __init__(self, schema, responses: list[dict]) -> None:
        self.schema = schema
        self.responses = list(responses)
        self.invocations = 0
        self.messages: list[list] = []

    def invoke(self, messages):
        self.messages.append(list(messages))
        index = min(self.invocations, len(self.responses) - 1)
        self.invocations += 1
        return self.schema(**self.responses[index])


class RecordingLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.extractor: RecordingStructuredExtractor | None = None

    def with_structured_output(self, schema):
        self.extractor = RecordingStructuredExtractor(schema, self.responses)
        return self.extractor


def build_settings(*, knowledge_dir: str, conversion_dir: str) -> Settings:
    return Settings(
        slack_enabled=True,
        slack_bot_token="xoxb-test",
        slack_app_token="xapp-test",
        web_enabled=False,
        web_host="127.0.0.1",
        web_port=8000,
        google_api_key="test-key",
        gemini_model="gemini-3-flash-preview",
        gemini_temperature=0.2,
        google_application_credentials="/tmp/fake-creds.json",
        jade_project_sheet_id="sheet-id",
        project_sheet_range="Tasks!A1:Z",
        project_sheet_cache_ttl_seconds=30,
        slack_thinking_reaction="eyes",
        project_lookup_keywords=("task",),
        knowledge_base_dir=knowledge_dir,
        knowledge_file_types=(".md", ".txt", ".csv", ".tsv", ".xlsx", ".xlsm"),
        conversion_work_dir=conversion_dir,
    )


class ConversionRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace = Path(self.temp_dir.name)
        self.knowledge_root = self.workspace / "knowledge"
        self.conversion_root = self.workspace / "conversion"
        self.knowledge_root.mkdir(parents=True, exist_ok=True)
        self.conversion_root.mkdir(parents=True, exist_ok=True)
        self.settings = build_settings(
            knowledge_dir=str(self.knowledge_root),
            conversion_dir=str(self.conversion_root),
        )

    def _write_source_file(self, name: str, content: str) -> str:
        path = self.workspace / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _make_source(self, name: str, content: str, *, source_type: str = "google_sheet") -> ConversionSourceRecord:
        raw_path = self._write_source_file(name, content)
        return ConversionSourceRecord(
            source_doc_id=f"src-{Path(name).stem}",
            session_id="session-1",
            slack_file_id=f"file-{Path(name).stem}",
            original_name=name,
            source_type=source_type,
            author="tester",
            coverage="full",
            raw_path=raw_path,
            notes="",
            upload_ts="2026-03-30T00:00:00+00:00",
        )

    def test_build_conversion_chunks_windows_large_google_sheet_tables(self) -> None:
        table_rows = "\n".join(f"| row_{index} | reward_{index} |" for index in range(65))
        source = self._make_source(
            "weekly_sheet.md",
            "\n".join(
                [
                    "# Weekly Sheet",
                    "## Sheet: Main",
                    "| task | reward |",
                    "| --- | --- |",
                    table_rows,
                ]
            ),
        )

        chunks = build_conversion_chunks([source])
        summary_chunks = [chunk for chunk in chunks if chunk.section_title == "Source Summary"]
        table_chunks = [chunk for chunk in chunks if chunk.sheet_name == "Main" and chunk.section_title != "Source Summary"]

        self.assertEqual(len(summary_chunks), 1)
        self.assertEqual([(chunk.row_start, chunk.row_end) for chunk in table_chunks], [(1, 30), (26, 55), (51, 65)])
        self.assertTrue(all("| task | reward |" in chunk.text for chunk in table_chunks))

    def test_retrieve_conversion_chunks_prefers_relevant_sheet(self) -> None:
        source = self._make_source(
            "rules_sheet.md",
            "\n".join(
                [
                    "# Rule Book",
                    "## Sheet: Overview",
                    "General introduction for the feature.",
                    "",
                    "## Sheet: Weekly Tasks",
                    "| task | rule |",
                    "| --- | --- |",
                    "| fishing | weekly reward reset after 7 days |",
                    "| spending | reward threshold unlocks milestone chest |",
                    "",
                    "## Sheet: Economy",
                    "| item | price |",
                    "| --- | --- |",
                    "| gem pack | 100 |",
                    "| coin pack | 50 |",
                ]
            ),
        )
        chunks = build_conversion_chunks([source])

        selected = retrieve_conversion_chunks(
            chunks,
            {"rules": "weekly task reward reset threshold milestone"},
        )
        content_chunks = [chunk for chunk in selected if chunk.section_title != "Source Summary"]

        self.assertTrue(content_chunks)
        self.assertEqual({chunk.sheet_name for chunk in content_chunks}, {"Weekly Tasks"})

    def test_build_retrieved_source_bundle_respects_char_budget(self) -> None:
        rows = "\n".join(f"| task_{index} | reward_{index} |" for index in range(240))
        source = self._make_source(
            "oversized_sheet.md",
            "\n".join(
                [
                    "# Oversized Sheet",
                    "## Sheet: Weekly Tasks",
                    "| task | reward |",
                    "| --- | --- |",
                    rows,
                    "",
                    "## Sheet: Overview",
                    "Weekly activity for Indonesia players.",
                ]
            ),
        )

        bundle = build_retrieved_source_bundle(
            [source],
            shared_context="",
            existing_package_context="",
            answer_history=["Please convert this weekly activity for Indonesia."],
            latest_user_text="Please convert this weekly activity for Indonesia.",
            max_chars=4_000,
        )

        self.assertLessEqual(len(bundle), 4_000)
        self.assertIn("## Source Summary:", bundle)
        self.assertIn("## Retrieved Evidence:", bundle)

    def test_answer_history_affects_retrieval_selection(self) -> None:
        source = self._make_source(
            "market_notes.md",
            "\n".join(
                [
                    "# Market Notes",
                    "## Indonesia",
                    "This weekly feature launches in Indonesia with a seven-day reward ladder.",
                    "",
                    "## Vietnam",
                    "This weekly feature launches in Vietnam with a three-day reward ladder.",
                ]
            ),
            source_type="md",
        )
        chunks = build_conversion_chunks([source])
        queries = build_conversion_queries(
            latest_user_text="Please convert this document.",
            answer_history=["This is for the Indonesia market."],
            game_slug="",
            market_slug="",
            feature_slug="",
        )

        selected = retrieve_conversion_chunks(chunks, queries)
        content_chunks = [chunk for chunk in selected if chunk.section_title != "Source Summary"]

        self.assertTrue(any(chunk.section_title == "Indonesia" for chunk in content_chunks))

    def test_document_conversion_agent_uses_retrieved_bundle_for_large_google_sheet(self) -> None:
        payload = {
            "game_name": "BuYuDaLuanDou",
            "game_slug": "buyudalouandou",
            "market_name": "Indonesia",
            "market_slug": "indonesia",
            "feature_name": "Weekly Activity",
            "feature_slug": "weekly-activity",
            "overview": "Weekly activity rewards recurring engagement.",
            "terminology": [{"term_id": "weekly_activity", "canonical_en": "Weekly Activity", "definition": "A weekly feature."}],
            "entities": [{"entity_id": "weekly_task", "name_en": "Weekly Task", "description": "A task in the weekly activity."}],
            "rules": [{"rule_id": "weekly_reset", "title_en": "Weekly Reset", "description": "Tasks reset every cycle."}],
            "config_overview": ["Weekly tasks unlock rewards across the cycle."],
        }
        llm = RecordingLLM([payload, payload])
        node = DocumentConversionAgentNode(llm, settings=self.settings)

        filler_rows = "\n".join(f"| filler_{index} | ignore me |" for index in range(220))
        relevant_rows = "\n".join(f"| weekly_task_{index} | reward resets after 7 days |" for index in range(120))
        google_sheet_content = "\n".join(
            [
                "# Weekly Activity Spec",
                "## Sheet: Overview",
                "Design purpose: weekly activity for Indonesia players.",
                "",
                "## Sheet: Weekly Tasks",
                "| task | rule |",
                "| --- | --- |",
                relevant_rows,
                "",
                "## Sheet: Filler",
                "| name | note |",
                "| --- | --- |",
                filler_rows,
            ]
        )

        with patch(
            "tools.document_conversion.fetch_google_document_source",
            return_value=("Weekly Activity Spec", "google_sheet", google_sheet_content),
        ):
            result = node(
                {
                    "messages": [HumanMessage(content="Please convert https://docs.google.com/spreadsheets/d/sheet123/edit for Indonesia weekly activity.")],
                    "thread_id": "DRETRIEVE",
                    "channel_id": "DRETRIEVE",
                    "user_id": "URETRIEVE",
                }
            )

        self.assertEqual(result["conversion_status"], "ready_for_approval")
        self.assertIsNotNone(llm.extractor)
        first_bundle = llm.extractor.messages[0][1].content
        self.assertLessEqual(len(first_bundle), 90_000)
        self.assertIn("## Retrieved Evidence: Weekly Activity Spec", first_bundle)
        self.assertIn("### Sheet: Weekly Tasks", first_bundle)
        self.assertNotIn("filler_219", first_bundle)


if __name__ == "__main__":
    unittest.main()
