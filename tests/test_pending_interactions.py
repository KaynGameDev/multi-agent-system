from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.knowledge.agent import KnowledgeAgentNode
from agents.knowledge_base_builder.agent import KnowledgeBaseBuilderAgentNode
from agents.project_task.agent import ProjectTaskAgentNode
from app.graph import build_default_agent_registrations
from app.tool_registry import resolve_tool_ids_for_runtime_tools


class NoopLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        raise RuntimeError("LLM should not be called during deterministic follow-up tests.")


class SummaryLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="Here is the short task summary.")


class FallbackLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content=self.content)


class PendingActionFlowTests(unittest.TestCase):
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
        node = KnowledgeAgentNode(NoopLLM(), [], agent_name="knowledge_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="what docs are available"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                ]
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
        node = KnowledgeAgentNode(NoopLLM(), [], agent_name="knowledge_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show me the available docs"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_list"),
                ]
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
                ]
            }
        )

        self.assertEqual(result["messages"][-1].content, "llm fallback")
        self.assertNotIn("pending_action", result)

    def test_project_task_agent_summary_sets_pending_action_and_details_reuse_payload(self) -> None:
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
        node = ProjectTaskAgentNode(SummaryLLM(), [], agent_name="project_task_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    HumanMessage(content="show my tasks due today"),
                    ToolMessage(content=json.dumps(payload), tool_call_id="call_tasks"),
                ]
            }
        )
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
                    AIMessage(content="Here is the short task summary."),
                    HumanMessage(content="details"),
                ]
            }
        )

        self.assertEqual(result["messages"][-1].content, "task fallback")
        self.assertNotIn("pending_action", result)

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
        node = KnowledgeBaseBuilderAgentNode(NoopLLM(), [], agent_name="knowledge_base_builder_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ]
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
        node = KnowledgeBaseBuilderAgentNode(NoopLLM(), [], agent_name="knowledge_base_builder_agent")

        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ]
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
        node = KnowledgeBaseBuilderAgentNode(NoopLLM(), [], agent_name="knowledge_base_builder_agent")
        first_result = node(
            {
                "thread_id": "thread-1",
                "messages": [
                    write_request,
                    ToolMessage(content=json.dumps(blocked_payload), tool_call_id="call_write"),
                ]
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
