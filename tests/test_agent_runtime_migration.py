from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.knowledge.agent import KnowledgeAgentNode
from agents.knowledge_base_builder.agent import KnowledgeBaseBuilderAgentNode
from agents.project_task.agent import ProjectTaskAgentNode
from app.contracts import validate_pending_action_decision
from app.graph import build_default_agent_registrations
from app.pending_actions import get_pending_action_selection_options
from app.routing.pending_action_router import PendingActionRouter
from app.tool_registry import resolve_tool_ids_for_runtime_tools


class NoopLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        raise RuntimeError("LLM should not be called during deterministic follow-up tests.")


class FallbackLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content=self.content)


class ReplyMapInterpreter:
    def __init__(self, mapping: dict[str, dict], *, default: dict | None = None) -> None:
        self.mapping = {str(key).strip().casefold(): dict(value) for key, value in mapping.items()}
        self.default = dict(
            default
            or {
                "decision": "unclear",
                "reason": "The reply was ambiguous.",
                "constraints": [],
                "selected_item_id": None,
            }
        )

    def parse_pending_action_decision(self, action, user_message):
        normalized_reply = str(user_message or "").strip().casefold()
        payload = dict(self.mapping.get(normalized_reply, self.default))
        return validate_pending_action_decision(
            {
                "type": "pending_action_decision",
                "pending_action_id": str(payload.get("pending_action_id") or action.get("id") or "").strip(),
                "decision": str(payload.get("decision", "unclear")).strip().lower() or "unclear",
                "notes": str(payload.get("notes") or payload.get("reason") or "").strip() or None,
                "selected_item_id": resolve_selected_item_id(action, payload),
                "constraints": normalize_constraints(payload),
            }
        )


def build_pending_action_router(mapping: dict[str, dict], *, default: dict | None = None) -> PendingActionRouter:
    return PendingActionRouter(ReplyMapInterpreter(mapping, default=default))


def normalize_constraints(payload: dict[str, object]) -> list[str]:
    raw_constraints = payload.get("constraints")
    if isinstance(raw_constraints, list):
        return [str(item).strip() for item in raw_constraints if str(item).strip()]

    constraints: list[str] = []
    requested_outputs = payload.get("requested_outputs")
    if isinstance(requested_outputs, list):
        constraints.extend(
            f"output:{str(item).strip()}"
            for item in requested_outputs
            if str(item).strip()
        )

    target_scope = payload.get("target_scope")
    if isinstance(target_scope, dict):
        for field_name in ("files", "modules"):
            raw_items = target_scope.get(field_name)
            if isinstance(raw_items, list):
                constraints.extend(
                    f"{field_name}:{str(item).strip()}"
                    for item in raw_items
                    if str(item).strip()
                )
        skill_name = str(target_scope.get("skill_name", "")).strip()
        if skill_name:
            constraints.append(f"skill_name:{skill_name}")
    return constraints


def resolve_selected_item_id(action, payload: dict[str, object]) -> str | None:
    selected_item_id = str(payload.get("selected_item_id", "")).strip()
    if selected_item_id:
        return selected_item_id

    selected_index = payload.get("selected_index")
    if not isinstance(selected_index, int) or selected_index < 0:
        return None

    options = get_pending_action_selection_options(action)
    if selected_index >= len(options):
        return None
    option = options[selected_index]
    payload_dict = option.get("payload")
    for candidate in (
        option.get("id"),
        option.get("value"),
        option.get("label"),
        payload_dict.get("id") if isinstance(payload_dict, dict) else None,
        str(selected_index + 1),
    ):
        cleaned = str(candidate or "").strip()
        if cleaned:
            return cleaned
    return None


class AgentRuntimeMigrationTests(unittest.TestCase):
    def test_knowledge_agent_list_follow_up_uses_pending_action(self) -> None:
        payload = {
            "ok": True,
            "document_count": 2,
            "documents": [
                {
                    "name": "SetupGuide",
                    "title": "Setup Guide",
                    "path": "knowledge/Docs/00_Shared/SetupGuide.md",
                },
                {
                    "name": "ArchitectureOverview",
                    "title": "Architecture Overview",
                    "path": "knowledge/Docs/00_Shared/ArchitectureOverview.md",
                },
            ],
        }
        node = KnowledgeAgentNode(
            NoopLLM(),
            [],
            pending_action_router=build_pending_action_router(
                {
                    "第2个": {
                        "decision": "select",
                        "selected_index": 1,
                        "reason": "The user selected the second document.",
                    }
                }
            ),
            agent_name="knowledge_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="what docs are available"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                ],
            }
        )
        self.assertIn("pending_action", first_result)
        self.assertEqual(first_result["pending_action"]["metadata"]["selection_phase"], "awaiting_selection")

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="what docs are available"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                    *first_result["messages"],
                    HumanMessage(content="第2个"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        last_message = second_result["messages"][-1]
        self.assertEqual(last_message.tool_calls[0]["name"], "read_knowledge_document")
        self.assertEqual(last_message.tool_calls[0]["args"]["document_name"], "ArchitectureOverview")
        self.assertEqual(second_result["pending_action"]["metadata"]["selection_phase"], "render_after_tool_result")

        read_payload = {
            "ok": True,
            "document": {
                "name": "ArchitectureOverview",
                "title": "Architecture Overview",
                "path": "knowledge/Docs/00_Shared/ArchitectureOverview.md",
            },
            "content": "Architecture excerpt",
            "start_line": 1,
            "end_line": 10,
            "section_query": "",
            "truncated": False,
        }
        third_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="what docs are available"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                    *first_result["messages"],
                    HumanMessage(content="第2个"),
                    second_result["messages"][-1],
                    ToolMessage(content=json.dumps(read_payload), tool_call_id="call_read"),
                ],
                "pending_action": second_result["pending_action"],
            }
        )
        self.assertIn("Architecture Overview", third_result["messages"][-1].content)
        self.assertEqual(third_result["pending_action"]["metadata"]["source_tool_id"], "knowledge.read_document")
        self.assertEqual(third_result["pending_action"]["metadata"]["selection_phase"], "awaiting_selection")

    def test_knowledge_agent_single_document_alias_supports_referential_replies(self) -> None:
        payload = {
            "ok": True,
            "document_count": 1,
            "documents": [
                {
                    "name": "SetupGuide",
                    "title": "Setup Guide",
                    "path": "knowledge/Docs/00_Shared/SetupGuide.md",
                }
            ],
        }
        node = KnowledgeAgentNode(
            NoopLLM(),
            [],
            pending_action_router=build_pending_action_router(
                {
                    "details": {
                        "decision": "select",
                        "selected_index": 0,
                        "reason": "The user selected the only document.",
                    },
                    "that one": {
                        "decision": "select",
                        "selected_index": 0,
                        "reason": "The user selected the only document.",
                    },
                    "详情": {
                        "decision": "select",
                        "selected_index": 0,
                        "reason": "The user selected the only document.",
                    },
                }
            ),
            agent_name="knowledge_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show me the available docs"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                ],
            }
        )
        for reply in ("details", "that one", "详情"):
            with self.subTest(reply=reply):
                second_result = node(
                    {
                        "thread_id": "thread-1",
                        "messages": [
                            HumanMessage(content="show me the available docs"),
                            ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                            *first_result["messages"],
                            HumanMessage(content=reply),
                        ],
                        "pending_action": first_result["pending_action"],
                    }
                )

                last_message = second_result["messages"][-1]
                self.assertEqual(last_message.tool_calls[0]["name"], "read_knowledge_document")
                self.assertEqual(last_message.tool_calls[0]["args"]["document_name"], "SetupGuide")

    def test_knowledge_agent_follow_up_without_pending_action_falls_back_to_llm(self) -> None:
        payload = {
            "ok": True,
            "document_count": 1,
            "documents": [
                {
                    "name": "SetupGuide",
                    "title": "Setup Guide",
                    "path": "knowledge/Docs/00_Shared/SetupGuide.md",
                }
            ],
        }
        node = KnowledgeAgentNode(FallbackLLM("llm fallback"), [], agent_name="knowledge_agent")

        result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show me the available docs"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                    AIMessage(content="Setup Guide"),
                    HumanMessage(content="details"),
                ],
            }
        )

        self.assertEqual(result["messages"][-1].content, "llm fallback")
        self.assertNotIn("pending_action", result)

    def test_knowledge_agent_ambiguous_follow_up_blocks_execution_deterministically(self) -> None:
        payload = {
            "ok": True,
            "document_count": 1,
            "documents": [
                {
                    "name": "SetupGuide",
                    "title": "Setup Guide",
                    "path": "knowledge/Docs/00_Shared/SetupGuide.md",
                }
            ],
        }
        node = KnowledgeAgentNode(NoopLLM(), [], agent_name="knowledge_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show me the available docs"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                ],
            }
        )

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show me the available docs"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                    *first_result["messages"],
                    HumanMessage(content="maybe later"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        self.assertIn("can't determine", second_result["messages"][-1].content)
        self.assertEqual(second_result["pending_action"]["status"], "ask_clarification")
        self.assertNotIn("tool_invocation", second_result)

    def test_project_task_tool_result_renders_deterministically_and_details_reuse_payload(self) -> None:
        payload = {
            "ok": True,
            "tasks": [
                {
                    "content": "Ship durable memory",
                    "project": "Infra",
                    "iteration": "S1",
                    "platform": "Web",
                    "priority": "P1",
                    "assignee": "Tester",
                    "end_date": "2026-04-04",
                    "due_status": "today",
                }
            ],
            "match_count": 1,
            "filters": {
                "due_scope": "today",
                "assignee": "Tester",
            },
        }
        node = ProjectTaskAgentNode(
            NoopLLM(),
            [],
            pending_action_router=build_pending_action_router(
                {
                    "details": {
                        "decision": "select",
                        "selected_index": 0,
                        "reason": "The user selected the only task.",
                    }
                }
            ),
            agent_name="project_task_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show my tasks due today"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_tasks"),
                ],
            }
        )
        markdown = first_result["messages"][-1].content
        self.assertIn("Tasks due today for Tester", markdown)
        self.assertIn("Ship durable memory", markdown)
        self.assertIn("Reply with the task number or exact task title for details", markdown)
        self.assertIn("pending_action", first_result)

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show my tasks due today"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_tasks"),
                    *first_result["messages"],
                    HumanMessage(content="details"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )
        markdown = second_result["messages"][-1].content
        self.assertIn("Tasks due today for Tester", markdown)
        self.assertIn("Ship durable memory", markdown)
        self.assertIsNone(second_result["pending_action"])

    def test_project_task_follow_up_without_pending_action_falls_back_to_llm(self) -> None:
        payload = {
            "ok": True,
            "tasks": [
                {
                    "content": "Ship durable memory",
                    "project": "Infra",
                    "iteration": "S1",
                    "platform": "Web",
                    "priority": "P1",
                    "assignee": "Tester",
                    "end_date": "2026-04-04",
                    "due_status": "today",
                }
            ],
            "match_count": 1,
            "filters": {
                "due_scope": "today",
                "assignee": "Tester",
            },
        }
        node = ProjectTaskAgentNode(FallbackLLM("task fallback"), [], agent_name="project_task_agent")

        result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show my tasks due today"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_tasks"),
                    AIMessage(content="Tasks due today for Tester (1)\n\n1. Ship durable memory"),
                    HumanMessage(content="details"),
                ],
            }
        )

        self.assertEqual(result["messages"][-1].content, "task fallback")
        self.assertNotIn("pending_action", result)

    def test_project_task_ambiguous_follow_up_blocks_execution_deterministically(self) -> None:
        payload = {
            "ok": True,
            "tasks": [
                {
                    "content": "Ship durable memory",
                    "project": "Infra",
                    "iteration": "S1",
                    "platform": "Web",
                    "priority": "P1",
                    "assignee": "Tester",
                    "end_date": "2026-04-04",
                    "due_status": "today",
                }
            ],
            "match_count": 1,
            "filters": {
                "due_scope": "today",
                "assignee": "Tester",
            },
        }
        node = ProjectTaskAgentNode(NoopLLM(), [], agent_name="project_task_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show my tasks due today"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_tasks"),
                ],
            }
        )

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show my tasks due today"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_tasks"),
                    *first_result["messages"],
                    HumanMessage(content="maybe later"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        self.assertIn("can't determine", second_result["messages"][-1].content)
        self.assertEqual(second_result["pending_action"]["status"], "ask_clarification")

    def test_builder_pending_action_retries_write_on_natural_confirmation(self) -> None:
        write_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_knowledge_markdown_document",
                    "args": {
                        "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
                        "content": "# Test\n",
                    },
                    "id": "call_write",
                }
            ],
        )
        blocked_payload = {
            "ok": False,
            "knowledge_mutation": "write_markdown",
            "requires_confirmation": True,
            "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
            "absolute_path": "/tmp/Test.md",
            "target_exists": False,
        }
        confirmation_interpreter = ReplyMapInterpreter(
            {
                "approve": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
                "confirm": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
                "批准": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
                "确认": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
                "continue": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
                "go ahead": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
                "ok": {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The user approved the write.",
                    "confidence": 0.98,
                    "interpretation_source": "llm_parser",
                },
            }
        )
        node = KnowledgeBaseBuilderAgentNode(
            NoopLLM(),
            [],
            pending_action_router=PendingActionRouter(confirmation_interpreter),
            agent_name="knowledge_base_builder_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ],
            }
        )
        self.assertIn("pending_action", first_result)

        for reply in ("approve", "confirm", "批准", "确认", "continue", "go ahead", "ok"):
            with self.subTest(reply=reply):
                second_result = node(
                    {
                        "thread_id": "thread-1",
                        "messages": [
                            write_request,
                            ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                            *first_result["messages"],
                            HumanMessage(content=reply),
                        ],
                        "pending_action": first_result["pending_action"],
                    }
                )

                last_message = second_result["messages"][-1]
                self.assertEqual(last_message.tool_calls[0]["name"], "write_knowledge_markdown_document")
                self.assertEqual(second_result["execution_contract"]["decision"], "approve")
                self.assertEqual(second_result["pending_action"]["status"], "approved")

    def test_builder_pending_action_can_render_diff_before_execution(self) -> None:
        write_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_knowledge_markdown_document",
                    "args": {
                        "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
                        "content": "# Test\n\nNew content.\n",
                    },
                    "id": "call_write",
                }
            ],
        )
        blocked_payload = {
            "ok": False,
            "knowledge_mutation": "write_markdown",
            "requires_confirmation": True,
            "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
            "absolute_path": "/tmp/Test.md",
            "target_exists": False,
        }
        diff_interpreter = ReplyMapInterpreter(
            {
                "show me the diff first": {
                    "decision": "modify",
                    "requested_outputs": ["diff"],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": False,
                    "reason": "The user asked to review the diff first.",
                    "confidence": 0.97,
                    "interpretation_source": "llm_parser",
                }
            }
        )
        node = KnowledgeBaseBuilderAgentNode(
            NoopLLM(),
            [],
            pending_action_router=PendingActionRouter(diff_interpreter),
            agent_name="knowledge_base_builder_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ],
            }
        )

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                    *first_result["messages"],
                    HumanMessage(content="show me the diff first"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        self.assertIn("```diff", second_result["messages"][-1].content)
        self.assertEqual(second_result["pending_action"]["status"], "request_revision")
        self.assertIsNone(second_result["execution_contract"])

    def test_builder_pending_action_can_render_summary_before_execution(self) -> None:
        write_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_knowledge_markdown_document",
                    "args": {
                        "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
                        "content": "# Test\n\nNew content.\n",
                    },
                    "id": "call_write",
                }
            ],
        )
        blocked_payload = {
            "ok": False,
            "knowledge_mutation": "write_markdown",
            "requires_confirmation": True,
            "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
            "absolute_path": "/tmp/Test.md",
            "target_exists": False,
        }
        summary_interpreter = ReplyMapInterpreter(
            {
                "show me a summary": {
                    "decision": "modify",
                    "requested_outputs": ["summary"],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": False,
                    "reason": "The user asked to review a summary first.",
                    "confidence": 0.97,
                    "interpretation_source": "llm_parser",
                }
            }
        )
        node = KnowledgeBaseBuilderAgentNode(
            NoopLLM(),
            [],
            pending_action_router=PendingActionRouter(summary_interpreter),
            agent_name="knowledge_base_builder_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ],
            }
        )

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                    *first_result["messages"],
                    HumanMessage(content="show me a summary"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        self.assertIn("summary for the pending knowledge-base write", second_result["messages"][-1].content)
        self.assertNotIn("```diff", second_result["messages"][-1].content)
        self.assertEqual(second_result["pending_action"]["status"], "request_revision")
        self.assertIsNone(second_result["execution_contract"])

    def test_builder_ambiguous_follow_up_blocks_execution_deterministically(self) -> None:
        write_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_knowledge_markdown_document",
                    "args": {
                        "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
                        "content": "# Test\n",
                    },
                    "id": "call_write",
                }
            ],
        )
        blocked_payload = {
            "ok": False,
            "knowledge_mutation": "write_markdown",
            "requires_confirmation": True,
            "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
            "absolute_path": "/tmp/Test.md",
            "target_exists": False,
        }
        cancel_interpreter = ReplyMapInterpreter(
            {
                "取消": {
                    "decision": "reject",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": False,
                    "reason": "The user cancelled the write.",
                    "confidence": 0.96,
                    "interpretation_source": "llm_parser",
                }
            }
        )
        node = KnowledgeBaseBuilderAgentNode(
            NoopLLM(),
            [],
            pending_action_router=PendingActionRouter(cancel_interpreter),
            agent_name="knowledge_base_builder_agent",
        )

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ],
            }
        )

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                    *first_result["messages"],
                    HumanMessage(content="maybe later"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        self.assertIn("cannot execute", second_result["messages"][-1].content)
        self.assertEqual(second_result["pending_action"]["status"], "ask_clarification")
        self.assertIsNone(second_result["execution_contract"])

    def test_builder_pending_action_cancels(self) -> None:
        write_request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_knowledge_markdown_document",
                    "args": {
                        "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
                        "content": "# Test\n",
                    },
                    "id": "call_write",
                }
            ],
        )
        blocked_payload = {
            "ok": False,
            "knowledge_mutation": "write_markdown",
            "requires_confirmation": True,
            "relative_path": "Docs/10_GameLines/BuYuDaLuanDou/LineOverview/Test.md",
            "absolute_path": "/tmp/Test.md",
            "target_exists": False,
        }
        cancel_interpreter = ReplyMapInterpreter(
            {
                "取消": {
                    "decision": "reject",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": False,
                    "reason": "The user cancelled the write.",
                    "confidence": 0.96,
                    "interpretation_source": "llm_parser",
                }
            }
        )
        node = KnowledgeBaseBuilderAgentNode(
            NoopLLM(),
            [],
            pending_action_router=PendingActionRouter(cancel_interpreter),
            agent_name="knowledge_base_builder_agent",
        )
        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ],
            }
        )

        second_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                    *first_result["messages"],
                    HumanMessage(content="取消"),
                ],
                "pending_action": first_result["pending_action"],
            }
        )

        self.assertIn("取消", second_result["messages"][-1].content)
        self.assertIsNone(second_result["pending_action"])

    def test_default_registration_tool_ids_match_runtime_tools(self) -> None:
        registrations = {
            registration.name: registration
            for registration in build_default_agent_registrations()
            if registration.tool_ids
        }
        self.assertEqual(
            registrations["knowledge_agent"].tool_ids,
            resolve_tool_ids_for_runtime_tools(registrations["knowledge_agent"].tools),
        )
        self.assertEqual(
            registrations["knowledge_base_builder_agent"].tool_ids,
            resolve_tool_ids_for_runtime_tools(registrations["knowledge_base_builder_agent"].tools),
        )
        self.assertEqual(
            registrations["project_task_agent"].tool_ids,
            resolve_tool_ids_for_runtime_tools(registrations["project_task_agent"].tools),
        )


if __name__ == "__main__":
    unittest.main()
