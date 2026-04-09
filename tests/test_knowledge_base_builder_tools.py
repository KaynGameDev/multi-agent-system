from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import load_settings
from app.graph import build_default_agent_registrations
from app.knowledge_paths import build_knowledge_markdown_relative_path
from app.pending_actions import (
    build_pending_action,
    build_write_knowledge_approval_payload,
    compute_approval_payload_hash,
    resolve_pending_action_reply,
)
from tools.knowledge_base import resolve_knowledge_markdown_path, write_knowledge_markdown_document


class StaticInterpreter:
    def __init__(self, parsed_reply: dict) -> None:
        self.parsed_reply = dict(parsed_reply)

    def parse_pending_action_reply(self, _action, _prepared_input):
        return dict(self.parsed_reply)


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

    def _build_write_pending_action(self, *, relative_path: str, content: str, overwrite: bool = False) -> dict[str, object]:
        approval_payload = build_write_knowledge_approval_payload(
            relative_path=relative_path,
            content=content,
            overwrite=overwrite,
        )
        return build_pending_action(
            session_id="thread-1",
            action_type="write_knowledge_markdown",
            requested_by_agent="knowledge_base_builder_agent",
            summary=f"Write knowledge-base draft to `{relative_path}`.",
            target_scope={"files": [relative_path]},
            metadata={
                "relative_path": relative_path,
                "approval_payload": approval_payload,
                "approval_payload_hash": compute_approval_payload_hash(approval_payload),
            },
        )

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

        blocked = write_knowledge_markdown_document.invoke(
            {"relative_path": resolved["relative_path"], "content": "# 射击塔防组概览\n\n测试内容。\n"}
        )
        self.assertFalse(blocked["ok"])
        self.assertTrue(blocked["requires_confirmation"])
        self.assertEqual(blocked["knowledge_mutation"], "write_markdown")

        pending_action = self._build_write_pending_action(
            relative_path=resolved["relative_path"],
            content="# 射击塔防组概览\n\n测试内容。\n",
        )
        execution_contract = resolve_pending_action_reply(
            pending_action,
            "approve",
            interpreter=StaticInterpreter(
                {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.99,
                    "interpretation_source": "llm_parser",
                }
            ),
        )["contract"]

        written = write_knowledge_markdown_document.func(
            relative_path=resolved["relative_path"],
            content="# 射击塔防组概览\n\n测试内容。\n",
            state={
                "pending_action": pending_action,
                "execution_contract": execution_contract,
                "messages": [
                    HumanMessage(content="请写入知识库"),
                    ToolMessage(content=json.dumps(blocked, ensure_ascii=False), tool_call_id="call_write"),
                    HumanMessage(content="approve"),
                ]
            },
        )
        self.assertTrue(written["ok"])
        self.assertFalse(written["requires_confirmation"])
        self.assertTrue(written["created"])
        output_path = Path(written["absolute_path"])
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.read_text(encoding="utf-8"), "# 射击塔防组概览\n\n测试内容。\n")

        second_write = write_knowledge_markdown_document.func(
            relative_path=resolved["relative_path"],
            content="# 新版本\n",
            state={
                "pending_action": pending_action,
                "execution_contract": execution_contract,
                "messages": [
                    HumanMessage(content="请覆盖更新"),
                    ToolMessage(content=json.dumps(blocked, ensure_ascii=False), tool_call_id="call_write"),
                    HumanMessage(content="confirm"),
                ]
            },
        )
        self.assertFalse(second_write["ok"])
        self.assertTrue(second_write["requires_confirmation"])

    def test_builder_write_still_blocks_if_user_says_approve_without_prior_preview(self) -> None:
        result = write_knowledge_markdown_document.func(
            relative_path="Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
            content="# Test\n",
            state={"messages": [HumanMessage(content="approve")]},
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["requires_confirmation"])

    def test_builder_write_accepts_valid_execution_contract(self) -> None:
        relative_path = "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/ContractApproved.md"
        blocked = write_knowledge_markdown_document.invoke(
            {"relative_path": relative_path, "content": "# Contract\n\nApproved.\n"}
        )
        self.assertFalse(blocked["ok"])
        self.assertTrue(blocked["requires_confirmation"])

        pending_action = self._build_write_pending_action(
            relative_path=relative_path,
            content="# Contract\n\nApproved.\n",
        )
        execution_contract = resolve_pending_action_reply(
            pending_action,
            "continue",
            interpreter=StaticInterpreter(
                {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.99,
                    "interpretation_source": "llm_parser",
                }
            ),
        )["contract"]

        written = write_knowledge_markdown_document.func(
            relative_path=relative_path,
            content="# Contract\n\nApproved.\n",
            state={
                "pending_action": pending_action,
                "execution_contract": execution_contract,
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_knowledge_markdown_document",
                                "args": {"relative_path": relative_path, "content": "# Contract\n\nApproved.\n"},
                                "id": "call_write",
                            }
                        ],
                    ),
                    ToolMessage(content=json.dumps(blocked, ensure_ascii=False), tool_call_id="call_write"),
                    HumanMessage(content="continue"),
                ],
            },
        )

        self.assertTrue(written["ok"])
        self.assertTrue(Path(written["absolute_path"]).exists())

    def test_builder_write_rejects_changed_path_or_overwrite_flag_after_approval(self) -> None:
        approved_path = "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Approved.md"
        approved_content = "# Approved\n\nBody.\n"
        pending_action = self._build_write_pending_action(
            relative_path=approved_path,
            content=approved_content,
            overwrite=False,
        )
        execution_contract = resolve_pending_action_reply(
            pending_action,
            "continue",
            interpreter=StaticInterpreter(
                {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.99,
                    "interpretation_source": "llm_parser",
                }
            ),
        )["contract"]

        changed_path = write_knowledge_markdown_document.func(
            relative_path="Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Different.md",
            content=approved_content,
            state={"pending_action": pending_action, "execution_contract": execution_contract},
        )
        self.assertFalse(changed_path["ok"])
        self.assertTrue(changed_path["requires_confirmation"])

        overwrite_mismatch = write_knowledge_markdown_document.func(
            relative_path=approved_path,
            content=approved_content,
            overwrite=True,
            state={"pending_action": pending_action, "execution_contract": execution_contract},
        )
        self.assertFalse(overwrite_mismatch["ok"])
        self.assertTrue(overwrite_mismatch["requires_confirmation"])

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
