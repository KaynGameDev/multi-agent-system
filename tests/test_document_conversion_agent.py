from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

from agents.workers.document_conversion_agent import DocumentConversionAgentNode
from core.config import Settings
from tools.document_conversion import read_tsv


class DummyStructuredExtractor:
    def __init__(self, schema, responses: list[dict]) -> None:
        self.schema = schema
        self.responses = list(responses)
        self.invocations = 0

    def invoke(self, _messages):
        if not self.responses:
            return self.schema()
        index = min(self.invocations, len(self.responses) - 1)
        self.invocations += 1
        return self.schema(**self.responses[index])


class DummyLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses

    def with_structured_output(self, schema):
        return DummyStructuredExtractor(schema, self.responses)


def build_settings(*, knowledge_dir: str, conversion_dir: str) -> Settings:
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
        knowledge_base_dir=knowledge_dir,
        knowledge_file_types=(".md", ".txt", ".csv", ".tsv", ".xlsx", ".xlsm"),
        conversion_work_dir=conversion_dir,
    )


class DocumentConversionAgentTests(unittest.TestCase):
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

    def test_incomplete_conversion_session_asks_for_missing_target_info(self) -> None:
        source_path = self._write_source_file(
            "lucky_bundle.md",
            "# Lucky Bundle\n\nThis feature sells a daily bundle with crit rewards.\n",
        )
        llm = DummyLLM(
            [
                {
                    "game_name": "BuYuDaLuanDou",
                    "game_slug": "buyudalouandou",
                    "feature_name": "Lucky Bundle",
                    "feature_slug": "lucky-bundle",
                    "overview": "A paid bundle feature with critical reward multipliers.",
                    "terminology": [
                        {
                            "term_id": "lucky_bundle",
                            "canonical_zh": "幸运转转乐礼包",
                            "canonical_en": "Lucky Bundle",
                            "definition": "A purchasable bundle with a crit multiplier.",
                        }
                    ],
                    "entities": [
                        {
                            "entity_id": "bundle_tier",
                            "name_en": "Bundle Tier",
                            "description": "A purchasable tier inside the feature.",
                        }
                    ],
                    "rules": [
                        {
                            "rule_id": "reward_rule",
                            "title_en": "Reward Rule",
                            "description": "Rewards are base reward plus an optional crit multiplier.",
                        }
                    ],
                    "config_overview": ["Daily purchase limit is configurable."],
                }
            ]
        )
        node = DocumentConversionAgentNode(llm, settings=self.settings)

        result = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D123",
                "channel_id": "D123",
                "user_id": "U123",
                "uploaded_files": [
                    {
                        "id": "F123",
                        "name": "lucky_bundle.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(result["conversion_status"], "blocked_unknown_target")
        self.assertIn("Which market or package variant", result["messages"][0].content)
        self.assertIn("market_slug", result["messages"][0].content)

    def test_complete_conversion_session_stages_and_publishes_after_approval(self) -> None:
        source_path = self._write_source_file(
            "weekly_activity.md",
            "# Weekly Activity\n\nThis feature adds weekly activity rewards.\n",
        )
        payload = {
            "game_name": "BuYuDaLuanDou",
            "game_slug": "buyudalouandou",
            "market_name": "Indonesia",
            "market_slug": "indonesia",
            "feature_name": "Weekly Activity",
            "feature_slug": "weekly-activity",
            "overview": "Weekly activity rewards players for recurring engagement and spend.",
            "terminology": [
                {
                    "term_id": "weekly_activity",
                    "canonical_zh": "周常活动",
                    "canonical_en": "Weekly Activity",
                    "definition": "A weekly engagement reward feature.",
                }
            ],
            "entities": [
                {
                    "entity_id": "activity_task",
                    "name_en": "Activity Task",
                    "description": "A repeatable goal that grants activity points.",
                }
            ],
            "rules": [
                {
                    "rule_id": "reward_rule",
                    "title_en": "Activity Reward Rule",
                    "description": "Rewards are unlocked by reaching weekly activity thresholds.",
                }
            ],
            "config_overview": ["Thresholds and rewards are server-configured."],
            "modules": [
                {
                    "name": "config",
                    "content": "## Config\n\n- Activity thresholds\n- Reward ladder",
                }
            ],
        }
        node = DocumentConversionAgentNode(DummyLLM([payload, payload]), settings=self.settings)

        staged = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D456",
                "channel_id": "D456",
                "user_id": "U456",
                "uploaded_files": [
                    {
                        "id": "F456",
                        "name": "weekly_activity.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(staged["conversion_status"], "ready_for_approval")
        self.assertIn("Reply with `approve`", staged["messages"][0].content)

        restarted_node = DocumentConversionAgentNode(DummyLLM([]), settings=self.settings)
        published = restarted_node(
            {
                "messages": [HumanMessage(content="approve")],
                "thread_id": "D456",
                "channel_id": "D456",
                "user_id": "U456",
            }
        )

        self.assertEqual(published["conversion_status"], "published")
        package_root = self.knowledge_root / "games" / "buyudalouandou" / "indonesia" / "weekly-activity"
        self.assertTrue((package_root / "README.md").exists())
        self.assertTrue((package_root / "core.md").exists())
        self.assertTrue((package_root / "facts.tsv").exists())
        self.assertTrue((package_root / "sources.tsv").exists())

    def test_second_publish_supersedes_changed_fact_rows(self) -> None:
        first_source = self._write_source_file("feature_v1.md", "# Feature V1\n\nInitial overview.\n")
        second_source = self._write_source_file("feature_v2.md", "# Feature V2\n\nUpdated overview.\n")
        first_payload = {
            "game_name": "BuYuDaLuanDou",
            "game_slug": "buyudalouandou",
            "market_name": "Indonesia",
            "market_slug": "indonesia",
            "feature_name": "Lucky Bundle",
            "feature_slug": "lucky-bundle",
            "overview": "Initial canonical overview.",
            "terminology": [{"term_id": "bundle", "canonical_en": "Bundle", "definition": "Bundle term."}],
            "entities": [{"entity_id": "bundle_tier", "name_en": "Bundle Tier", "description": "Tier entity."}],
            "rules": [{"rule_id": "reward_rule", "title_en": "Reward Rule", "description": "Initial reward rule."}],
            "config_overview": ["Initial config overview."],
        }
        updated_payload = {
            **first_payload,
            "overview": "Updated canonical overview.",
        }

        first_node = DocumentConversionAgentNode(DummyLLM([first_payload, first_payload]), settings=self.settings)
        first_stage = first_node(
            {
                "messages": [HumanMessage(content="Convert this.")],
                "thread_id": "D789",
                "channel_id": "D789",
                "user_id": "U789",
                "uploaded_files": [{"id": "F789a", "name": "feature_v1.md", "local_path": first_source}],
            }
        )
        approve_node = DocumentConversionAgentNode(DummyLLM([]), settings=self.settings)
        approve_node(
            {
                "messages": [HumanMessage(content="approve")],
                "thread_id": "D789",
                "channel_id": "D789",
                "user_id": "U789",
                "conversion_session_id": first_stage["conversion_session_id"],
            }
        )

        second_node = DocumentConversionAgentNode(DummyLLM([updated_payload, updated_payload]), settings=self.settings)
        second_stage = second_node(
            {
                "messages": [HumanMessage(content="Convert this update.")],
                "thread_id": "D789",
                "channel_id": "D789",
                "user_id": "U789",
                "uploaded_files": [{"id": "F789b", "name": "feature_v2.md", "local_path": second_source}],
            }
        )
        approve_node(
            {
                "messages": [HumanMessage(content="approve")],
                "thread_id": "D789",
                "channel_id": "D789",
                "user_id": "U789",
                "conversion_session_id": second_stage["conversion_session_id"],
            }
        )

        facts_rows = read_tsv(
            self.knowledge_root / "games" / "buyudalouandou" / "indonesia" / "lucky-bundle" / "facts.tsv"
        )
        overview_rows = [
            row
            for row in facts_rows
            if row["attribute"] == "overview" and row["module"] == "core"
        ]
        self.assertEqual(len(overview_rows), 2)
        self.assertEqual(sum(1 for row in overview_rows if row["fact_status"] == "active"), 1)
        self.assertEqual(sum(1 for row in overview_rows if row["fact_status"] == "superseded"), 1)


if __name__ == "__main__":
    unittest.main()
