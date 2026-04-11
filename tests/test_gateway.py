from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app.graph import build_default_agent_registrations
from app.interpretation.intent_parser import IntentParser
from app.routing.agent_router import AgentRouter
from app.routing.pending_action_router import (
    PendingActionRouter,
    build_pending_action_resolution_key,
)
from app.skills import SkillRegistry
from gateway.agent import GatewayNode
from tests.common import build_registration, write_skill


def keyword_matcher(keyword: str, score: int = 50):
    def matcher(_state, latest_user_text: str):
        return None

    return matcher


class FakeRoutingLLM:
    def __init__(self, decisions=None, *, default_agent: str = "", default_reason: str = "") -> None:
        self.decisions = dict(decisions or {})
        self.default_agent = default_agent
        self.default_reason = default_reason

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self, schema)

    def invoke(self, messages):
        latest_user_text = ""
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    latest_user_text = str(message.content)
                    break
        else:
            latest_user_text = extract_latest_user_text_from_prompt(str(messages))
        decision = self.decisions.get(
            latest_user_text,
            {
                "selected_agent": self.default_agent,
                "reason": self.default_reason,
            },
        )
        return dict(decision)


class FakeStructuredLLM:
    def __init__(self, parent: FakeRoutingLLM, schema) -> None:
        self.parent = parent
        self.schema = schema

    def invoke(self, messages):
        decision = self.parent.invoke(messages)
        schema_name = getattr(self.schema, "__name__", "")
        if schema_name in {"AssistantRequest", "AssistantRequestCandidate"}:
            likely_domain = str(decision.get("likely_domain", "")).strip().lower()
            if not likely_domain:
                likely_domain = domain_for_selected_agent(str(decision.get("selected_agent", "")).strip())
            return {
                "type": "assistant_request",
                "user_goal": str(decision.get("user_goal") or decision.get("reason") or "Handle the request.").strip(),
                "likely_domain": likely_domain or "general",
                "confidence": float(decision.get("confidence", 0.95)),
                "notes": str(decision.get("reason", "")).strip() or None,
            }
        return decision


class ValidationErrorThenRawRoutingLLM:
    def __init__(self, raw_response: AIMessage) -> None:
        self.raw_response = raw_response
        self.structured_calls = 0
        self.raw_calls = 0

    def with_structured_output(self, _schema, **_kwargs):
        parent = self

        class BrokenStructuredInvoker:
            def invoke(self, _prompt):
                parent.structured_calls += 1

                class Envelope(BaseModel):
                    value: str = ""

                Envelope.model_validate_json("<think>not-json</think>")

        return BrokenStructuredInvoker()

    def invoke(self, _prompt):
        self.raw_calls += 1
        return self.raw_response


def domain_for_selected_agent(agent_name: str) -> str:
    mapping = {
        "general_chat_agent": "general",
        "knowledge_agent": "knowledge",
        "project_task_agent": "project_task",
        "knowledge_base_builder_agent": "knowledge_base_builder",
        "document_conversion_agent": "document_conversion",
    }
    return mapping.get(agent_name, "general")


def extract_latest_user_text_from_prompt(prompt: str) -> str:
    marker = "Latest user message:"
    if marker in prompt:
        return prompt.rsplit(marker, 1)[-1].strip().splitlines()[-1].strip()
    return prompt.strip().splitlines()[-1].strip() if prompt.strip() else ""

class StaticPendingActionParser:
    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = dict(decision)

    def parse_pending_action_decision(self, action, _user_message):
        payload = dict(self.decision)
        payload.setdefault("type", "pending_action_decision")
        payload.setdefault("pending_action_id", str(action.get("id", "")).strip())
        payload.setdefault("notes", None)
        payload.setdefault("selected_item_id", None)
        payload.setdefault("constraints", [])
        return payload


class ReplyMapPendingActionParser:
    def __init__(self, mapping: dict[str, dict[str, object]]) -> None:
        self.mapping = {str(key): dict(value) for key, value in mapping.items()}
        self.calls = 0

    def parse_pending_action_decision(self, action, user_message):
        self.calls += 1
        payload = dict(self.mapping.get(str(user_message), {}))
        payload.setdefault("type", "pending_action_decision")
        payload.setdefault("pending_action_id", str(action.get("id", "")).strip())
        payload.setdefault("decision", "unclear")
        payload.setdefault("notes", None)
        payload.setdefault("selected_item_id", None)
        payload.setdefault("constraints", [])
        return payload


class TrackingAssistantRequestParser:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = dict(payload)
        self.calls = 0

    def parse_assistant_request(self, _user_message, **_kwargs):
        self.calls += 1
        return dict(self.payload)


class RecentContextCapturingAssistantRequestParser(TrackingAssistantRequestParser):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(payload)
        self.last_recent_messages: list[str] = []

    def parse_assistant_request(self, _user_message, **kwargs):
        self.calls += 1
        self.last_recent_messages = list(kwargs.get("recent_messages") or [])
        return dict(self.payload)


class ParserOnlyRoutingLLM(FakeRoutingLLM):
    def invoke(self, messages):
        if isinstance(messages, list):
            raise AssertionError("Legacy heuristic routing should not be called in production.")
        return super().invoke(messages)


class GatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def build_orchestrator(self, registrations, *, routing_llm=None, pending_action_router=None, agent_router=None):
        registry = SkillRegistry(tuple(registrations), project_root=self.root)
        return GatewayNode(
            routing_llm,
            agent_registrations=tuple(registrations),
            default_route=registrations[0].name,
            skill_registry=registry,
            pending_action_router=pending_action_router,
            agent_router=agent_router,
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
        self.assertEqual(result["skill_invocation_contracts"][0]["mode"], "fork")
        self.assertEqual(result["skill_invocation_contracts"][0]["target_agent"], "beta_agent")
        self.assertEqual(result["active_skill_invocation_contracts"][0]["skill_id"], "route-beta")
        self.assertEqual(result["skill_execution_diagnostics"][0]["executed_by_agent"], "beta_agent")

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
        self.assertEqual(result["route_policy_step"], "forked_skill_fallback")
        self.assertEqual(result["routing_decision"]["policy_step"], "forked_skill_fallback")

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
        self.assertEqual(result["route_policy_step"], "forked_skill_fallback")

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

        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True, selection_order=30),
            build_registration("alpha_agent", namespace="alpha", selection_order=20),
            build_registration("beta_agent", namespace="beta", selection_order=10),
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
        self.assertEqual(result["route_policy_step"], "inline_skill_compatibility")

    def test_pending_action_owner_takes_priority_over_requested_agent_and_skills(self) -> None:
        write_skill(
            self.root,
            ".jade/skills/route-alpha",
            frontmatter={
                "name": "Route Alpha",
                "description": "Delegate to alpha.",
                "execution_mode": "forked",
                "delegate_agent": "alpha_agent",
                "available_to_agents": ["beta_agent"],
            },
            body="# Route Alpha\n\nDelegate to alpha.",
        )
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha", matcher=keyword_matcher("alpha", 90)),
            build_registration("beta_agent", namespace="beta", matcher=keyword_matcher("beta", 80)),
            build_registration("project_task_agent", namespace="project_task", selection_order=10),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="alpha please")],
                "requested_agent": "beta_agent",
                "requested_skill_ids": ["route-alpha"],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")
        self.assertEqual(result["routing_decision"]["policy_step"], "pending_action_owner")

    def test_pending_action_owner_beats_explicit_forked_skill_delegate(self) -> None:
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
            build_registration("alpha_agent", namespace="alpha", matcher=keyword_matcher("alpha", 90)),
            build_registration("beta_agent", namespace="beta", matcher=keyword_matcher("beta", 80)),
            build_registration("project_task_agent", namespace="project_task", selection_order=10),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="alpha please")],
                "requested_skill_ids": ["route-beta"],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")

    def test_pending_action_owner_beats_forked_skill_general_fallback(self) -> None:
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
            build_registration("project_task_agent", namespace="project_task", selection_order=10),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="continue")],
                "requested_skill_ids": ["general-fallback"],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")

    def test_pending_action_owner_beats_inline_skill_compatibility(self) -> None:
        write_skill(
            self.root,
            ".jade/skills/shared-inline",
            frontmatter={
                "name": "Shared Inline",
                "description": "Usable by alpha.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Shared Inline\n\nShared inline behavior.",
        )
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha", selection_order=20),
            build_registration("project_task_agent", namespace="project_task", selection_order=10),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="something unrelated")],
                "requested_skill_ids": ["shared-inline"],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")

    def test_pending_action_owner_beats_parser_based_fresh_route(self) -> None:
        registrations = build_default_agent_registrations()
        routing_llm = FakeRoutingLLM(
            {
                "can you write files?": {
                    "selected_agent": "knowledge_base_builder_agent",
                    "reason": "The model router would otherwise choose the knowledge-base builder.",
                }
            }
        )
        orchestrator = self.build_orchestrator(registrations, routing_llm=routing_llm)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="can you write files?")],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")

    def test_parser_contract_selects_specialist_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha", selection_order=10, matcher=keyword_matcher("alpha", 95)),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=30,
            ),
        )
        routing_llm = FakeRoutingLLM(
            {
                "alpha can you write files?": {
                    "selected_agent": "knowledge_base_builder_agent",
                    "reason": "This request is best handled by the knowledge-base builder.",
                }
            }
        )
        orchestrator = self.build_orchestrator(registrations, routing_llm=routing_llm)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="alpha can you write files?")],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "knowledge_base_builder_agent")

    def test_uploaded_files_route_to_document_conversion_from_state(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration(
                "document_conversion_agent",
                namespace="document_conversion",
            ),
        )
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    "Please convert this.": {
                        "selected_agent": "document_conversion_agent",
                        "reason": "Uploaded files indicate a document conversion request.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content="Please convert this.")],
                "uploaded_files": [{"name": "design.md"}],
            }
        )

        self.assertEqual(result["route"], "document_conversion_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "document_conversion_agent")

    def test_builder_elicitation_request_routes_to_knowledge_base_builder(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
            ),
        )
        text = "请一步步提问，帮我们梳理这个功能并整理成 feature spec 骨架。"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "The user is asking for structured knowledge elicitation.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_builder_review_request_routes_to_knowledge_base_builder(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
            ),
        )
        text = "请 review 这份 KB 文档的 metadata 和层级归属。"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "The user is asking for KB document review.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_builder_tracking_request_routes_to_knowledge_base_builder(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
            ),
        )
        text = "当前 KB V1 到哪个 milestone 了？"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "The user is asking for KB execution tracking.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_builder_confirmation_follow_up_without_pending_action_uses_general(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
            ),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            {
                                "ok": False,
                                "knowledge_mutation": "write_markdown",
                                "requires_confirmation": True,
                                "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Shooting_TowerDefense_Group_Overview.md",
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id="call_write",
                    ),
                    HumanMessage(content="approve"),
                ],
            }
        )

        self.assertEqual(result["route"], "general_chat_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_direct_file_write_capability_question_routes_to_knowledge_base_builder(self) -> None:
        registrations = build_default_agent_registrations()
        text = "你可以写文件吗"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "This is a request about KB write capability.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_english_tool_availability_question_routes_to_knowledge_base_builder(self) -> None:
        registrations = build_default_agent_registrations()
        text = "can you write files?"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "This asks about KB file-writing capability.",
                    }
                }
            ),
        )

        result = orchestrator({"messages": [HumanMessage(content=text)]})

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_write_into_file_phrase_routes_to_knowledge_base_builder(self) -> None:
        registrations = build_default_agent_registrations()
        text = "你能帮我写入文件吗"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "This is a KB write request.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_save_discussion_to_knowledge_base_routes_to_builder(self) -> None:
        registrations = build_default_agent_registrations()
        text = "can you save our discussion to the knowledge base?"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "This requests saving discussion into the KB.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_update_into_knowledge_base_phrase_routes_to_knowledge_base_builder(self) -> None:
        registrations = build_default_agent_registrations()
        text = "你能帮我把内容更新到知识库吗"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "This asks to update the knowledge base.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")

    def test_generic_repository_question_stays_on_knowledge_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
            build_registration(
                "knowledge_base_builder_agent",
                namespace="knowledge_base_builder",
                selection_order=35,
            ),
        )
        text = "请介绍一下这个仓库的 setup 文档和架构说明。"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_agent",
                        "reason": "This is a repository documentation question.",
                    }
                }
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
            }
        )

        self.assertEqual(result["route"], "knowledge_agent")

    def test_pending_action_short_circuits_to_owner_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
            build_registration("project_task_agent", namespace="project_task", selection_order=20),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="details")],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                    "metadata": {
                        "source_tool_id": "project.read_tasks",
                        "prompt_context": "Reply with a task number.",
                        "selection_options": [],
                    },
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertIn("pending action", result["route_reason"].lower())
        self.assertEqual(result["route_policy_step"], "pending_action_owner")

    def test_legacy_pending_interaction_is_ignored_by_gateway(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
            build_registration("project_task_agent", namespace="project_task", selection_order=20),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="details")],
                "pending_interaction": {
                    "kind": "selection",
                    "owner_agent": "project_task_agent",
                    "source_tool_id": "project.read_tasks",
                    "status": "awaiting_reply",
                    "prompt_context": "Reply with a task number.",
                    "options": [],
                    "accepted_replies": [],
                    "cancel_replies": ["cancel"],
                    "payload": {},
                },
            }
        )

        self.assertEqual(result["route"], "general_chat_agent")
        self.assertNotIn("pending interaction", result["route_reason"].lower())
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_pending_action_short_circuits_to_owner_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_base_builder_agent", namespace="knowledge_base_builder", selection_order=20),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="go ahead")],
                "pending_action": {
                    "id": "pending_write",
                    "session_id": "thread-1",
                    "type": "write_knowledge_markdown",
                    "requested_by_agent": "knowledge_base_builder_agent",
                    "summary": "Write KB draft",
                    "risk_level": "medium",
                    "requires_explicit_approval": True,
                    "created_at": "2026-04-05T00:00:00Z",
                    "status": "awaiting_confirmation",
                },
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertIn("pending action", result["route_reason"].lower())
        self.assertEqual(result["route_policy_step"], "pending_action_owner")

    def test_pending_action_with_missing_owner_records_warning_and_falls_back(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator(
            {
                "messages": [HumanMessage(content="continue")],
                "pending_action": {
                    "id": "pending_write",
                    "session_id": "thread-1",
                    "type": "write_knowledge_markdown",
                    "requested_by_agent": "missing_agent",
                    "summary": "Write KB draft",
                    "risk_level": "medium",
                    "requires_explicit_approval": True,
                    "created_at": "2026-04-05T00:00:00Z",
                    "status": "awaiting_confirmation",
                },
            }
        )

        self.assertEqual(result["route"], "general_chat_agent")
        self.assertTrue(any("Pending action owner `missing_agent`" in item for item in result["selection_warnings"]))
        self.assertEqual(result["route_policy_step"], "pending_action_owner_fallback")
        # Pending action must be expired so the user is not stuck
        self.assertEqual(result["pending_action"]["status"], "expired")
        # An expiry message must be injected so the user is informed
        expiry_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        self.assertTrue(any("cancelled" in m.content.lower() for m in expiry_messages))

    def test_unrelated_pending_action_reply_allows_fresh_routing(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
            build_registration("project_task_agent", namespace="project_task", selection_order=20),
        )
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                default_agent="knowledge_agent",
                default_reason="Knowledge question.",
            ),
            pending_action_router=PendingActionRouter(
                StaticPendingActionParser(
                    {
                        "decision": "unrelated",
                        "notes": "The user started a fresh knowledge question.",
                    }
                )
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content="What docs are available?")],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                    "metadata": {
                        "source_tool_id": "project.read_tasks",
                        "prompt_context": "Reply with a task number.",
                        "selection_options": [],
                    },
                },
            }
        )

        self.assertEqual(result["route"], "knowledge_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["pending_action_decision"]["decision"], "unrelated")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "knowledge_agent")
        self.assertIsNone(result["execution_contract"])

    def test_gateway_reparses_stale_pending_action_decision_on_new_turn(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
        )
        parser = ReplyMapPendingActionParser(
            {
                "approve": {
                    "decision": "approve",
                    "notes": "The user approved the change.",
                }
            }
        )
        orchestrator = self.build_orchestrator(
            registrations,
            pending_action_router=PendingActionRouter(parser),
        )
        pending_action = {
            "id": "pending_apply_edit",
            "session_id": "thread-1",
            "type": "apply_edit",
            "requested_by_agent": "general_chat_agent",
            "summary": "Apply the proposed edit.",
            "status": "ask_clarification",
            "created_at": "2026-04-05T00:00:00Z",
            "metadata": {},
        }
        stale_state = {
            "messages": [HumanMessage(content="maybe")],
            "pending_action": pending_action,
        }

        result = orchestrator(
            {
                "messages": [
                    HumanMessage(content="maybe"),
                    HumanMessage(content="approve"),
                ],
                "pending_action": pending_action,
                "pending_action_decision": {
                    "type": "pending_action_decision",
                    "pending_action_id": "pending_apply_edit",
                    "decision": "unclear",
                    "notes": "The reply was ambiguous.",
                    "selected_item_id": None,
                    "constraints": [],
                },
                "pending_action_resolution_key": build_pending_action_resolution_key(stale_state, pending_action),
            }
        )

        self.assertEqual(parser.calls, 1)
        self.assertEqual(result["route"], "general_chat_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")
        self.assertEqual(result["pending_action_decision"]["decision"], "approve")
        self.assertEqual(result["execution_contract"]["decision"], "approve")
        self.assertEqual(
            result["pending_action_resolution_key"],
            build_pending_action_resolution_key(
                {
                    "messages": [
                        HumanMessage(content="maybe"),
                        HumanMessage(content="approve"),
                    ]
                },
                pending_action,
            ),
        )

    def test_no_specialist_match_falls_back_to_general_assistant(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha", matcher=keyword_matcher("alpha")),
        )
        orchestrator = self.build_orchestrator(registrations)

        result = orchestrator({"messages": [HumanMessage(content="Something unrelated.")]})

        self.assertEqual(result["route"], "general_chat_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_parser_routes_knowledge_request_to_knowledge_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
        )
        text = "How is this repo structured?"
        parser = TrackingAssistantRequestParser(
            {
                "type": "assistant_request",
                "user_goal": "Explain the repository architecture docs.",
                "likely_domain": "knowledge",
                "confidence": 0.94,
                "notes": "Documentation request.",
            }
        )
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_agent",
                        "reason": "Repository documentation request.",
                    }
                }
            ),
            agent_router=AgentRouter(parser),
        )

        result = orchestrator({"messages": [HumanMessage(content=text)]})

        self.assertEqual(result["route"], "knowledge_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "knowledge_agent")
        self.assertEqual(parser.calls, 1)

    def test_parser_routes_task_request_to_project_task_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("project_task_agent", namespace="project_task", selection_order=20),
        )
        text = "Who owns the current sprint tasks?"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "project_task_agent",
                        "reason": "Project tracker request.",
                    }
                }
            ),
            agent_router=AgentRouter(
                TrackingAssistantRequestParser(
                    {
                        "type": "assistant_request",
                        "user_goal": "Show current task ownership and deadlines.",
                        "likely_domain": "project_task",
                        "confidence": 0.95,
                        "notes": "Task tracker request.",
                    }
                )
            ),
        )

        result = orchestrator({"messages": [HumanMessage(content=text)]})

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "project_task_agent")

    def test_parser_routes_builder_request_to_knowledge_base_builder_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_base_builder_agent", namespace="knowledge_base_builder", selection_order=35),
        )
        text = "Help me build a feature spec skeleton."
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_base_builder_agent",
                        "reason": "Feature spec builder request.",
                    }
                }
            ),
            agent_router=AgentRouter(
                TrackingAssistantRequestParser(
                    {
                        "type": "assistant_request",
                        "user_goal": "Help elicit and structure a feature spec.",
                        "likely_domain": "knowledge_base_builder",
                        "confidence": 0.96,
                        "notes": "KB builder request.",
                    }
                )
            ),
        )

        result = orchestrator({"messages": [HumanMessage(content=text)]})

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "knowledge_base_builder_agent")

    def test_parser_routes_document_conversion_request_to_document_conversion_agent(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("document_conversion_agent", namespace="document_conversion", selection_order=10),
        )
        text = "Please convert this design package."
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "document_conversion_agent",
                        "reason": "Document conversion request.",
                    }
                }
            ),
            agent_router=AgentRouter(
                TrackingAssistantRequestParser(
                    {
                        "type": "assistant_request",
                        "user_goal": "Convert the uploaded design package.",
                        "likely_domain": "document_conversion",
                        "confidence": 0.97,
                        "notes": "Conversion request.",
                    }
                )
            ),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content=text)],
                "uploaded_files": [{"name": "design.md"}],
            }
        )

        self.assertEqual(result["route"], "document_conversion_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "document_conversion_agent")

    def test_low_confidence_parser_request_trusts_parsed_domain(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
        )
        text = "Can you take care of this?"
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=FakeRoutingLLM(
                {
                    text: {
                        "selected_agent": "knowledge_agent",
                        "reason": "Treat this as a knowledge request.",
                    }
                }
            ),
            agent_router=AgentRouter(
                TrackingAssistantRequestParser(
                    {
                        "type": "assistant_request",
                        "user_goal": "Handle this somehow.",
                        "likely_domain": "knowledge",
                        "confidence": 0.12,
                        "notes": "Too ambiguous to classify safely.",
                    }
                )
            ),
        )

        result = orchestrator({"messages": [HumanMessage(content=text)]})

        self.assertEqual(result["route"], "knowledge_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "knowledge_agent")

    def test_gateway_recovers_parser_route_from_raw_llm_after_structured_validation_error(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_base_builder_agent", namespace="knowledge_base_builder", selection_order=35),
        )
        llm = ValidationErrorThenRawRoutingLLM(
            AIMessage(
                content="""<think>
The user wants to write the discussed company knowledge into the knowledge base.
</think>

```json
{"type":"assistant_request","user_goal":"Write the discussed company knowledge into the knowledge base.","likely_domain":"knowledge_base_builder","confidence":0.96,"notes":"Explicit KB write request."}
```"""
            )
        )
        orchestrator = self.build_orchestrator(
            registrations,
            agent_router=AgentRouter(IntentParser(llm)),
        )

        result = orchestrator(
            {
                "messages": [
                    HumanMessage(content="我整理完了公司知识"),
                    AIMessage(content="抱歉，我目前没有写入知识库的权限或工具。"),
                    HumanMessage(content="请把这些内容录入到知识库"),
                ]
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(llm.structured_calls, 1)
        self.assertEqual(llm.raw_calls, 1)

    def test_pending_action_override_still_works(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("project_task_agent", namespace="project_task", selection_order=20),
        )
        parser = TrackingAssistantRequestParser(
            {
                "type": "assistant_request",
                "user_goal": "This should not be used.",
                "likely_domain": "knowledge",
                "confidence": 0.99,
                "notes": None,
            }
        )
        orchestrator = self.build_orchestrator(
            registrations,
            agent_router=AgentRouter(parser),
        )

        result = orchestrator(
            {
                "messages": [HumanMessage(content="continue")],
                "pending_action": {
                    "id": "pending_select",
                    "session_id": "thread-1",
                    "type": "select_project_task",
                    "requested_by_agent": "project_task_agent",
                    "summary": "Select a task to inspect.",
                    "status": "awaiting_confirmation",
                    "created_at": "2026-04-05T00:00:00Z",
                },
            }
        )

        self.assertEqual(result["route"], "project_task_agent")
        self.assertEqual(result["route_policy_step"], "pending_action_owner")
        self.assertEqual(parser.calls, 0)
        self.assertEqual(result["agent_route_decision"]["selected_agent"], "project_task_agent")
        self.assertFalse(result["agent_route_decision"]["fallback_used"])

    def test_gateway_does_not_call_legacy_heuristic_router_in_production_path(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
        )
        orchestrator = self.build_orchestrator(
            registrations,
            routing_llm=ParserOnlyRoutingLLM(
                {
                    "How is this repo structured?": {
                        "selected_agent": "knowledge_agent",
                        "reason": "Repository documentation request.",
                    }
                }
            ),
        )

        result = orchestrator({"messages": [HumanMessage(content="How is this repo structured?")]})

        self.assertEqual(result["route"], "knowledge_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")

    def test_route_diagnostics_include_parsed_contract_and_selected_route(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_agent", namespace="knowledge", selection_order=30),
        )
        orchestrator = self.build_orchestrator(
            registrations,
            agent_router=AgentRouter(
                TrackingAssistantRequestParser(
                    {
                        "type": "assistant_request",
                        "user_goal": "Explain the repository architecture docs.",
                        "likely_domain": "knowledge",
                        "confidence": 0.93,
                        "notes": "Documentation request.",
                    }
                )
            ),
        )

        result = orchestrator({"messages": [HumanMessage(content="How is this repo structured?")]})

        assistant_request_diagnostic = result["agent_selection_diagnostics"][0]
        route_diagnostic = result["agent_selection_diagnostics"][1]

        self.assertEqual(assistant_request_diagnostic["kind"], "assistant_request")
        self.assertEqual(assistant_request_diagnostic["likely_domain"], "knowledge")
        self.assertEqual(route_diagnostic["kind"], "agent_route_decision")
        self.assertEqual(route_diagnostic["selected_agent"], "knowledge_agent")
        self.assertFalse(route_diagnostic["fallback_used"])

    def test_recent_context_is_sanitized_before_reaching_top_level_parser(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("knowledge_base_builder_agent", namespace="knowledge_base_builder", selection_order=35),
        )
        parser = RecentContextCapturingAssistantRequestParser(
            {
                "type": "assistant_request",
                "user_goal": "Write the captured company knowledge into the knowledge base.",
                "likely_domain": "knowledge_base_builder",
                "confidence": 0.96,
                "notes": "The user explicitly wants KB write behavior.",
            }
        )
        orchestrator = self.build_orchestrator(
            registrations,
            agent_router=AgentRouter(parser),
        )

        result = orchestrator(
            {
                "messages": [
                    HumanMessage(content="我整理完了公司知识"),
                    AIMessage(
                        content=(
                            "<think>\nI should refuse because I am only a general chat agent.\n</think>\n\n"
                            "抱歉，我目前没有写入知识库的权限或工具。"
                        )
                    ),
                    AIMessage(content=[{"type": "text", "text": "结构化块内容"}]),
                    ToolMessage(content='{"ok":false,"tool":"write"}', tool_call_id="call_write"),
                    HumanMessage(content="不是记录到对话中，我要你写入知识库"),
                ]
            }
        )

        self.assertEqual(result["route"], "knowledge_base_builder_agent")
        self.assertEqual(result["route_policy_step"], "assistant_request_domain")
        self.assertEqual(
            parser.last_recent_messages,
            [
                "user: 我整理完了公司知识",
                "assistant: 抱歉，我目前没有写入知识库的权限或工具。",
                "assistant: 结构化块内容",
                "user: 不是记录到对话中，我要你写入知识库",
            ],
        )
        self.assertFalse(any("<think>" in item for item in parser.last_recent_messages))
        self.assertFalse(any('{"ok":false' in item for item in parser.last_recent_messages))
        self.assertFalse(any("[{'type':" in item for item in parser.last_recent_messages))


if __name__ == "__main__":
    unittest.main()
