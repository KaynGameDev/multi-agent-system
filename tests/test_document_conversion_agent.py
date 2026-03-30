from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpcore
from langchain_core.messages import AIMessage, HumanMessage

from agents.workers.document_conversion_agent import DocumentConversionAgentNode
from core.config import Settings
from tools.document_conversion import compact_source_bundle_for_extraction, read_tsv


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


class RenderingLLM(DummyLLM):
    def __init__(self, responses: list[dict], render_content) -> None:
        super().__init__(responses)
        self.render_content = render_content
        self.render_invocations = 0

    def invoke(self, _messages):
        self.render_invocations += 1
        if isinstance(self.render_content, Exception):
            raise self.render_content
        return AIMessage(content=self.render_content)


class RaisingStructuredExtractor:
    def invoke(self, _messages):
        raise RuntimeError("Server disconnected without sending a response.")


class RaisingLLM:
    def with_structured_output(self, _schema):
        return RaisingStructuredExtractor()


class FlakyStructuredExtractor:
    def __init__(self, schema, response: dict) -> None:
        self.schema = schema
        self.response = response
        self.invocations = 0

    def invoke(self, _messages):
        self.invocations += 1
        if self.invocations == 1:
            raise httpcore.RemoteProtocolError("Server disconnected without sending a response.")
        return self.schema(**self.response)


class FlakyLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.extractor: FlakyStructuredExtractor | None = None

    def with_structured_output(self, schema):
        self.extractor = FlakyStructuredExtractor(schema, self.response)
        return self.extractor


class ProxyError(RuntimeError):
    pass


class ProxyRaisingStructuredExtractor:
    def invoke(self, _messages):
        raise ProxyError("503 Service Unavailable")


class ProxyRaisingLLM:
    def with_structured_output(self, _schema):
        return ProxyRaisingStructuredExtractor()


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

    def test_incomplete_conversion_session_replies_in_chinese_for_chinese_request(self) -> None:
        source_path = self._write_source_file(
            "lucky_bundle_zh.md",
            "# 幸运礼包\n\n这个功能售卖一个带暴击奖励的每日礼包。\n",
        )
        llm = DummyLLM(
            [
                {
                    "game_name": "捕鱼大乱斗",
                    "game_slug": "buyudalouandou",
                    "feature_name": "幸运礼包",
                    "feature_slug": "lucky-bundle",
                    "overview": "一个付费礼包功能，带有暴击奖励倍率。",
                    "terminology": [
                        {
                            "term_id": "lucky_bundle",
                            "canonical_zh": "幸运礼包",
                            "canonical_en": "Lucky Bundle",
                            "definition": "一个可购买的礼包，带有暴击倍率。",
                        }
                    ],
                    "entities": [
                        {
                            "entity_id": "bundle_tier",
                            "name_zh": "礼包档位",
                            "description": "功能中的一个可购买档位。",
                        }
                    ],
                    "rules": [
                        {
                            "rule_id": "reward_rule",
                            "title_zh": "奖励规则",
                            "description": "奖励由基础奖励和可选暴击倍率组成。",
                        }
                    ],
                    "config_overview": ["每日购买次数可以配置。"],
                }
            ]
        )
        node = DocumentConversionAgentNode(llm, settings=self.settings)

        result = node(
            {
                "messages": [HumanMessage(content="请转换这个文档。")],
                "thread_id": "D123ZH",
                "channel_id": "D123ZH",
                "user_id": "U123ZH",
                "uploaded_files": [
                    {
                        "id": "F123ZH",
                        "name": "lucky_bundle_zh.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(result["conversion_status"], "blocked_unknown_target")
        self.assertIn("我还需要一些信息", result["messages"][0].content)
        self.assertIn("这个功能属于哪个市场或包体版本", result["messages"][0].content)

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

    def test_google_doc_url_stages_and_publishes_after_approval(self) -> None:
        payload = {
            "game_name": "BuYuDaLuanDou",
            "game_slug": "buyudalouandou",
            "market_name": "Indonesia",
            "market_slug": "indonesia",
            "feature_name": "Lucky Bundle",
            "feature_slug": "lucky-bundle",
            "overview": "A paid bundle feature with rotating rewards.",
            "terminology": [
                {
                    "term_id": "lucky_bundle",
                    "canonical_zh": "幸运转转乐礼包",
                    "canonical_en": "Lucky Bundle",
                    "definition": "A rotating paid bundle offer.",
                }
            ],
            "entities": [
                {
                    "entity_id": "bundle_reward_pool",
                    "name_en": "Bundle Reward Pool",
                    "description": "The reward set used by the bundle offer.",
                }
            ],
            "rules": [
                {
                    "rule_id": "bundle_refresh_rule",
                    "title_en": "Bundle Refresh Rule",
                    "description": "The offer refreshes on a configured cadence.",
                }
            ],
            "config_overview": ["Refresh cadence and reward pool are configurable."],
        }
        node = DocumentConversionAgentNode(DummyLLM([payload, payload]), settings=self.settings)

        with patch(
            "tools.document_conversion.fetch_google_document_source",
            return_value=(
                "Lucky Bundle Spec",
                "google_doc",
                "# Lucky Bundle Spec\n\nA paid bundle with rotating rewards.\n",
            ),
        ):
            staged = node(
                {
                    "messages": [HumanMessage(content="Please convert https://docs.google.com/document/d/doc123/edit")],
                    "thread_id": "DGOOGLE",
                    "channel_id": "DGOOGLE",
                    "user_id": "UGOOGLE",
                }
            )

        self.assertEqual(staged["conversion_status"], "ready_for_approval")
        self.assertIn("Reply with `approve`", staged["messages"][0].content)

        restarted_node = DocumentConversionAgentNode(DummyLLM([]), settings=self.settings)
        published = restarted_node(
            {
                "messages": [HumanMessage(content="approve")],
                "thread_id": "DGOOGLE",
                "channel_id": "DGOOGLE",
                "user_id": "UGOOGLE",
            }
        )

        self.assertEqual(published["conversion_status"], "published")
        package_root = self.knowledge_root / "games" / "buyudalouandou" / "indonesia" / "lucky-bundle"
        self.assertTrue((package_root / "README.md").exists())
        source_rows = read_tsv(package_root / "sources.tsv")
        self.assertEqual(len(source_rows), 1)
        self.assertEqual(source_rows[0]["source_type"], "google_doc")
        self.assertEqual(source_rows[0]["slack_file_id"], "google_doc:doc123")

    def test_complete_conversion_session_replies_in_chinese_and_accepts_chinese_approval(self) -> None:
        source_path = self._write_source_file(
            "weekly_activity_zh.md",
            "# 周常活动\n\n这个功能为玩家提供每周活跃奖励。\n",
        )
        payload = {
            "game_name": "捕鱼大乱斗",
            "game_slug": "buyudalouandou",
            "market_name": "印尼",
            "market_slug": "indonesia",
            "feature_name": "周常活动",
            "feature_slug": "weekly-activity",
            "overview": "周常活动通过每周目标和奖励提升玩家活跃与付费。",
            "terminology": [
                {
                    "term_id": "weekly_activity",
                    "canonical_zh": "周常活动",
                    "canonical_en": "Weekly Activity",
                    "definition": "一个按周循环的活跃奖励功能。",
                }
            ],
            "entities": [
                {
                    "entity_id": "activity_task",
                    "name_zh": "活跃任务",
                    "description": "一个可重复完成并发放活跃点数的目标。",
                }
            ],
            "rules": [
                {
                    "rule_id": "reward_rule",
                    "title_zh": "活跃奖励规则",
                    "description": "玩家达到每周活跃阈值后解锁奖励。",
                }
            ],
            "config_overview": ["阈值和奖励由服务端配置。"],
            "modules": [
                {
                    "name": "config",
                    "content": "## 配置\n\n- 活跃阈值\n- 奖励梯度",
                }
            ],
        }
        node = DocumentConversionAgentNode(DummyLLM([payload, payload]), settings=self.settings)

        staged = node(
            {
                "messages": [HumanMessage(content="请把这个文档转换成规范知识包。")],
                "thread_id": "D456ZH",
                "channel_id": "D456ZH",
                "user_id": "U456ZH",
                "uploaded_files": [
                    {
                        "id": "F456ZH",
                        "name": "weekly_activity_zh.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(staged["conversion_status"], "ready_for_approval")
        self.assertIn("等待确认", staged["messages"][0].content)
        self.assertIn("请回复 `approve`（或“批准”）", staged["messages"][0].content)

        restarted_node = DocumentConversionAgentNode(DummyLLM([]), settings=self.settings)
        published = restarted_node(
            {
                "messages": [HumanMessage(content="批准")],
                "thread_id": "D456ZH",
                "channel_id": "D456ZH",
                "user_id": "U456ZH",
            }
        )

        self.assertEqual(published["conversion_status"], "published")
        self.assertIn("已发布规范知识包", published["messages"][0].content)

    def test_ready_for_approval_uses_llm_renderer_when_output_meets_contract(self) -> None:
        source_path = self._write_source_file(
            "weekly_activity_render.md",
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
        render_content = (
            "Custom ready message.\n\n"
            "- Target path: `games/buyudalouandou/indonesia/weekly-activity`\n"
            "Reply with `approve` or `cancel`."
        )
        node = DocumentConversionAgentNode(
            RenderingLLM([payload, payload], render_content=render_content),
            settings=self.settings,
        )

        staged = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D457",
                "channel_id": "D457",
                "user_id": "U457",
                "uploaded_files": [
                    {
                        "id": "F457",
                        "name": "weekly_activity_render.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(staged["conversion_status"], "ready_for_approval")
        self.assertEqual(staged["messages"][0].content, render_content)
        self.assertEqual(node.llm.render_invocations, 1)

    def test_ready_for_approval_falls_back_when_llm_renderer_breaks_contract(self) -> None:
        source_path = self._write_source_file(
            "weekly_activity_render_fallback.md",
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
        invalid_render_content = "Looks good to publish."
        node = DocumentConversionAgentNode(
            RenderingLLM([payload, payload], render_content=invalid_render_content),
            settings=self.settings,
        )

        staged = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D458",
                "channel_id": "D458",
                "user_id": "U458",
                "uploaded_files": [
                    {
                        "id": "F458",
                        "name": "weekly_activity_render_fallback.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(staged["conversion_status"], "ready_for_approval")
        self.assertNotEqual(staged["messages"][0].content, invalid_render_content)
        self.assertIn("games/buyudalouandou/indonesia/weekly-activity", staged["messages"][0].content)
        self.assertIn("`approve`", staged["messages"][0].content)

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

    def test_failed_conversion_marks_session_inactive(self) -> None:
        source_path = self._write_source_file(
            "broken.md",
            "# Broken\n\nThis source should trigger a failed extractor call.\n",
        )
        node = DocumentConversionAgentNode(RaisingLLM(), settings=self.settings)

        result = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D999",
                "channel_id": "D999",
                "user_id": "U999",
                "uploaded_files": [
                    {
                        "id": "F999",
                        "name": "broken.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(result["conversion_status"], "failed")
        self.assertIn("Server disconnected without sending a response.", result["messages"][0].content)
        self.assertIsNone(node.store.get_active_session_by_thread("D999"))

    def test_compact_source_bundle_for_large_sheet_exports(self) -> None:
        header = "# Oversized Sheet\n## Sheet: Main"
        table_lines = [
            "| col_a | col_b |",
            "| --- | --- |",
        ]
        table_lines.extend(f"| row_{index} | value_{index} |" for index in range(80))
        bundle = "\n".join([header, *table_lines])

        compacted = compact_source_bundle_for_extraction(bundle, max_chars=1_500, table_head_rows=5, table_tail_rows=2)

        self.assertLess(len(compacted), len(bundle))
        self.assertIn("| row_0 | value_0 |", compacted)
        self.assertIn("| row_79 | value_79 |", compacted)
        self.assertIn("_Table truncated for extraction.", compacted)

    def test_transient_disconnect_retries_and_recovers(self) -> None:
        source_path = self._write_source_file(
            "recoverable.md",
            "# Recoverable\n\nThis source should succeed after one transient disconnect.\n",
        )
        payload = {
            "game_name": "BuYuDaLuanDou",
            "game_slug": "buyudalouandou",
            "market_name": "Indonesia",
            "market_slug": "indonesia",
            "feature_name": "Weekly Activity",
            "feature_slug": "weekly-activity",
            "overview": "A weekly activity with staged tasks and rewards.",
            "terminology": [{"term_id": "weekly_activity", "canonical_en": "Weekly Activity", "definition": "A weekly feature."}],
            "entities": [{"entity_id": "weekly_task", "name_en": "Weekly Task", "description": "A task in the weekly activity."}],
            "rules": [{"rule_id": "weekly_reset", "title_en": "Weekly Reset", "description": "Tasks reset every cycle."}],
            "config_overview": ["Weekly tasks unlock rewards across the cycle."],
        }
        llm = FlakyLLM(payload)
        node = DocumentConversionAgentNode(llm, settings=self.settings)

        result = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D1002",
                "channel_id": "D1002",
                "user_id": "U1002",
                "uploaded_files": [
                    {
                        "id": "F1002",
                        "name": "recoverable.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(result["conversion_status"], "ready_for_approval")
        self.assertIn("ready for approval", result["messages"][0].content.lower())
        self.assertIsNotNone(llm.extractor)
        self.assertGreaterEqual(llm.extractor.invocations, 2)

    def test_download_failure_returns_retry_message_instead_of_failing_session(self) -> None:
        node = DocumentConversionAgentNode(DummyLLM([]), settings=self.settings)

        with patch(
            "agents.workers.document_conversion_agent.ingest_uploaded_files",
            return_value=([], [], ["weekly.xlsx: <urlopen error Tunnel connection failed: 503 Service Unavailable>"]),
        ):
            result = node(
                {
                    "messages": [HumanMessage(content="Please convert this document.")],
                    "thread_id": "D1000",
                    "channel_id": "D1000",
                    "user_id": "U1000",
                    "uploaded_files": [
                        {
                            "id": "F1000",
                            "name": "weekly.xlsx",
                            "url_private_download": "https://files.slack.com/example",
                        }
                    ],
                }
        )

        self.assertEqual(result["conversion_status"], "needs_info")
        self.assertIn("I couldn't access one or more source documents", result["messages"][0].content)
        self.assertIn("Re-upload the file and try again.", result["messages"][0].content)

    def test_proxy_failure_returns_actionable_gemini_transport_message(self) -> None:
        source_path = self._write_source_file(
            "proxy_broken.md",
            "# Proxy Broken\n\nThis source should trigger a proxy transport failure.\n",
        )
        node = DocumentConversionAgentNode(ProxyRaisingLLM(), settings=self.settings)

        result = node(
            {
                "messages": [HumanMessage(content="Please convert this document.")],
                "thread_id": "D1001",
                "channel_id": "D1001",
                "user_id": "U1001",
                "uploaded_files": [
                    {
                        "id": "F1001",
                        "name": "proxy_broken.md",
                        "local_path": source_path,
                    }
                ],
            }
        )

        self.assertEqual(result["conversion_status"], "failed")
        self.assertIn("I couldn't reach the Gemini API", result["messages"][0].content)
        self.assertIn("GEMINI_HTTP_TRUST_ENV=false", result["messages"][0].content)
        self.assertIsNone(node.store.get_active_session_by_thread("D1001"))


if __name__ == "__main__":
    unittest.main()
