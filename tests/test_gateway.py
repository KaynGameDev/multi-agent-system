from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.skills import SkillRegistry
from gateway.agent import (
    AgentMatchResult,
    GatewayNode,
    document_conversion_matcher,
    knowledge_base_builder_matcher,
    knowledge_matcher,
)
from tests.common import build_registration, write_skill


def keyword_matcher(keyword: str, score: int = 50):
    def matcher(_state, latest_user_text: str) -> AgentMatchResult:
        if keyword in latest_user_text.lower():
            return AgentMatchResult(matched=True, score=score, reasons=(f"Matched keyword `{keyword}`.",))
        return AgentMatchResult(matched=False, score=0, reasons=())

    return matcher


class GatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def build_orchestrator(self, registrations):
        registry = SkillRegistry(tuple(registrations), project_root=self.root)
        return GatewayNode(
            None,
            agent_registrations=tuple(registrations),
            default_route=registrations[0].name,
            skill_registry=registry,
        )

    def test_forked_skill_with_delegate_agent_routes_to_that_agent(self) -> None:
        write_skill(
            self.root,
            ".jade/skills/route-beta",
            frontmatter={
                "name": "Route Beta",
                "description": "Delegate to beta.",
                "execution_mode": "forked",
                "delegate_agent": "beta_agent",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Route Beta\n\nDelegate to beta.",
        )
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha", matcher=keyword_matcher("alpha", 60)),
            build_registration("beta_agent", namespace="beta", matcher=keyword_matcher("beta", 60)),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="Please use the beta workflow.")],
                "requested_skill_ids": ["route-beta"],
            }
        )

        self.assertEqual(result["route"], "beta_agent")
        self.assertEqual(result["resolved_skill_ids"], ["route-beta"])

    def test_forked_skill_without_delegate_agent_falls_back_to_general_assistant(self) -> None:
        write_skill(
            self.root,
            ".jade/skills/general-fallback",
            frontmatter={
                "name": "General Fallback",
                "description": "Forked skill without delegate agent.",
                "execution_mode": "forked",
                "available_to_agents": ["alpha_agent"],
            },
            body="# General Fallback\n\nFallback to general assistant.",
        )
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha"),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="Fallback please.")],
                "requested_skill_ids": ["general-fallback"],
            }
        )

        self.assertEqual(result["route"], "general_chat_agent")

    def test_missing_general_assistant_uses_first_active_agent_with_warning(self) -> None:
        write_skill(
            self.root,
            ".jade/skills/general-fallback",
            frontmatter={
                "name": "General Fallback",
                "description": "Forked skill without delegate agent.",
                "execution_mode": "forked",
                "available_to_agents": ["alpha_agent"],
            },
            body="# General Fallback\n\nFallback to general assistant.",
        )
        registrations = (
            build_registration("alpha_agent", namespace="alpha", selection_order=20),
            build_registration("beta_agent", namespace="beta", selection_order=10),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="Fallback please.")],
                "requested_skill_ids": ["general-fallback"],
            }
        )

        self.assertEqual(result["route"], "beta_agent")
        self.assertTrue(any("GeneralAssistant is unavailable" in item for item in result["selection_warnings"]))

    def test_multi_agent_inline_skill_uses_selection_order_to_break_ties(self) -> None:
        write_skill(
            self.root,
            ".jade/skills/shared-inline",
            frontmatter={
                "name": "Shared Inline",
                "description": "Usable by alpha and beta.",
                "available_to_agents": ["alpha_agent", "beta_agent"],
            },
            body="# Shared Inline\n\nShared inline behavior.",
        )

        def tie_matcher(_state, latest_user_text: str) -> AgentMatchResult:
            if "shared" in latest_user_text.lower():
                return AgentMatchResult(matched=True, score=10, reasons=("Matched shared keyword.",))
            return AgentMatchResult(matched=False, score=0, reasons=())

        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True, selection_order=30),
            build_registration("alpha_agent", namespace="alpha", selection_order=20, matcher=tie_matcher),
            build_registration("beta_agent", namespace="beta", selection_order=10, matcher=tie_matcher),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="Shared behavior please.")],
                "requested_skill_ids": ["shared-inline"],
            }
        )

        self.assertEqual(result["route"], "beta_agent")
        self.assertEqual(result["resolved_skill_ids"], ["shared-inline"])

    def test_document_conversion_matcher_routes_deterministically(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration(
                "document_conversion_agent",
                namespace="document_conversion",
                matcher=document_conversion_matcher,
            ),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="Please convert this.")],
                "uploaded_files": [{"name": "design.md"}],
            }
        )

        self.assertEqual(result["route"], "document_conversion_agent")

    def test_builder_elicitation_request_routes_to_knowledge_base_builder(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30, matcher=knowledge_matcher),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
                matcher=knowledge_base_builder_matcher,
            ),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="请一步步提问，帮我们梳理这个功能并整理成 feature spec 骨架。")],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_builder_review_request_routes_to_knowledge_base_builder(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30, matcher=knowledge_matcher),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
                matcher=knowledge_base_builder_matcher,
            ),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="请 review 这份 KB 文档的 metadata 和层级归属。")],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_builder_tracking_request_routes_to_knowledge_base_builder(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30, matcher=knowledge_matcher),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
                matcher=knowledge_base_builder_matcher,
            ),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="当前 KB V1 到哪个 milestone 了？")],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_generic_repository_question_stays_on_knowledge_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30, matcher=knowledge_matcher),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
                matcher=knowledge_base_builder_matcher,
            ),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="请介绍一下这个仓库的 setup 文档和架构说明。")],
            }
        )

        self.assertEqual(result["route"], "knowledge_agent")

    def test_no_specialist_match_falls_back_to_general_assistant(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha", matcher=keyword_matcher("alpha")),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator({"messages": [HumanMessage(content="Something unrelated.")]})

        self.assertEqual(result["route"], "general_chat_agent")
        self.assertIn("GeneralAssistant fallback", result["route_reason"])


if __name__ == "__main__":
    unittest.main()
