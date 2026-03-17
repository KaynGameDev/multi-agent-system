from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.workers.project_task_agent import ProjectTaskAgentNode, build_task_list_response


TASK_PAYLOAD = {
    "ok": True,
    "match_count": 2,
    "filters": {
        "due_scope": "this_week",
        "assignee": "刘煜",
    },
    "tasks": [
        {
            "content": "大厅活动优化",
            "project": "Jade Poker",
            "iteration": "S24",
            "platform": "iOS",
            "priority": "P1",
            "assignee": "刘煜",
            "end_date": "2026-03-14",
            "due_status": "upcoming",
            "client_owner": "刘煜",
            "server_owner": "郑煜钊",
            "test_owner": "刘静芳",
            "product_owner": "叶俊杰",
        },
        {
            "content": "支付重构",
            "project": "Jade Poker",
            "iteration": "S24",
            "platform": "Android",
            "priority": "P2",
            "assignee": "刘煜",
            "end_date": "2026-03-15",
            "due_status": "upcoming",
            "client_owner": "刘煜",
            "server_owner": "郑煜钊",
            "test_owner": "刘静芳",
            "product_owner": "叶俊杰",
        },
    ],
}


class DummyLLM:
    def __init__(self, response_content: str = "LLM fallback", *, fail_on_invoke: bool = False) -> None:
        self.response_content = response_content
        self.fail_on_invoke = fail_on_invoke
        self.invocations = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations += 1
        if self.fail_on_invoke:
            raise AssertionError("LLM should not be called for deterministic task list formatting")
        return AIMessage(content=self.response_content)


class ProjectTaskAgentTests(unittest.TestCase):
    def test_build_task_list_response_formats_numbered_task_blocks(self) -> None:
        state = {
            "messages": [
                ToolMessage(content=json.dumps(TASK_PAYLOAD, ensure_ascii=False), tool_call_id="tool-1"),
                HumanMessage(content="What are those tasks?"),
            ]
        }

        response = build_task_list_response(state)

        self.assertIsNotNone(response)
        self.assertIn("Tasks due this week for 刘煜 (2)", response)
        self.assertIn("1. 大厅活动优化", response)
        self.assertIn("Project: Jade Poker | Iteration: S24 | Platform: iOS | Priority: P1", response)
        self.assertIn("Assignee: 刘煜 | Due: 2026-03-14 | Status: Upcoming", response)
        self.assertNotIn("• Project", response)

    def test_project_task_agent_uses_deterministic_list_response_for_follow_up(self) -> None:
        node = ProjectTaskAgentNode(DummyLLM(fail_on_invoke=True), tools=[])
        state = {
            "messages": [
                ToolMessage(content=json.dumps(TASK_PAYLOAD, ensure_ascii=False), tool_call_id="tool-1"),
                HumanMessage(content="What are those tasks?"),
            ]
        }

        result = node(state)

        self.assertEqual(len(result["messages"]), 1)
        self.assertIn("1. 大厅活动优化", result["messages"][0].content)

    def test_build_task_list_response_ignores_new_filtered_query(self) -> None:
        state = {
            "messages": [
                ToolMessage(content=json.dumps(TASK_PAYLOAD, ensure_ascii=False), tool_call_id="tool-1"),
                HumanMessage(content="Show overdue tasks"),
            ]
        }

        response = build_task_list_response(state)

        self.assertIsNone(response)

    def test_project_task_agent_calls_llm_for_new_filtered_query(self) -> None:
        llm = DummyLLM(response_content="Fresh tool-backed answer")
        node = ProjectTaskAgentNode(llm, tools=[])
        state = {
            "messages": [
                ToolMessage(content=json.dumps(TASK_PAYLOAD, ensure_ascii=False), tool_call_id="tool-1"),
                HumanMessage(content="Show overdue tasks"),
            ]
        }

        result = node(state)

        self.assertEqual(llm.invocations, 1)
        self.assertEqual(result["messages"][0].content, "Fresh tool-backed answer")


if __name__ == "__main__":
    unittest.main()
