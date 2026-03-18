from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.workers.knowledge_agent import KnowledgeAgentNode
from core.slack_formatting import to_slack_mrkdwn


LIST_PAYLOAD = {
    "ok": True,
    "document_count": 2,
    "documents": [
        {
            "title": "Game Design Spec",
            "path": "design/game_design.csv",
            "file_type": ".csv",
        },
        {
            "title": "Ops Guide",
            "path": "ops/guide.md",
            "file_type": ".md",
        },
    ],
}

READ_PAYLOAD = {
    "ok": True,
    "document": {
        "title": "Game Design Spec",
        "path": "design/game_design.csv",
        "file_type": ".csv",
    },
    "section_query": "Rewards",
    "start_line": 10,
    "end_line": 14,
    "content": "## Sheet: Rewards\nColumns: Rule | Value\nRow 2: Base | 100",
    "truncated": False,
}


class DummyLLM:
    def __init__(self, response_content: str = "LLM synthesis", *, fail_on_invoke: bool = False) -> None:
        self.response_content = response_content
        self.fail_on_invoke = fail_on_invoke
        self.invocations = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations += 1
        if self.fail_on_invoke:
            raise AssertionError("LLM should not be called for deterministic knowledge rendering")
        return AIMessage(content=self.response_content)


class KnowledgeAgentTests(unittest.TestCase):
    def test_latest_knowledge_tool_message_renders_deterministically_for_list_request(self) -> None:
        node = KnowledgeAgentNode(DummyLLM(fail_on_invoke=True), tools=[])
        state = {
            "messages": [
                HumanMessage(content="List the docs"),
                ToolMessage(content=json.dumps(LIST_PAYLOAD), tool_call_id="tool-1"),
            ]
        }

        result = node(state)

        self.assertEqual(len(result["messages"]), 1)
        self.assertIn("Documents (2)", result["messages"][0].content)
        self.assertIn("1. Game Design Spec", result["messages"][0].content)

    def test_referential_follow_up_reuses_latest_knowledge_payload(self) -> None:
        node = KnowledgeAgentNode(DummyLLM(fail_on_invoke=True), tools=[])
        state = {
            "messages": [
                ToolMessage(content=json.dumps(LIST_PAYLOAD), tool_call_id="tool-1"),
                AIMessage(content="Old model-shaped reply"),
                HumanMessage(content="What are those docs?"),
            ]
        }

        result = node(state)

        self.assertIn("Documents (2)", result["messages"][0].content)
        self.assertIn("2. Ops Guide", result["messages"][0].content)

    def test_synthesis_question_without_referential_follow_up_uses_llm(self) -> None:
        llm = DummyLLM(response_content="Synthesized answer from the model")
        node = KnowledgeAgentNode(llm, tools=[])
        state = {
            "messages": [
                HumanMessage(content="Explain the reward logic in simple terms"),
                ToolMessage(content=json.dumps(READ_PAYLOAD), tool_call_id="tool-1"),
            ]
        }

        result = node(state)

        self.assertEqual(llm.invocations, 1)
        self.assertEqual(result["messages"][0].content, "Synthesized answer from the model")

    def test_explicit_read_request_renders_read_payload_deterministically(self) -> None:
        node = KnowledgeAgentNode(DummyLLM(fail_on_invoke=True), tools=[])
        state = {
            "messages": [
                HumanMessage(content="Read the rewards section"),
                ToolMessage(content=json.dumps(READ_PAYLOAD), tool_call_id="tool-1"),
            ]
        }

        result = node(state)
        slack_text = to_slack_mrkdwn(result["messages"][0].content)

        self.assertIn("```text", slack_text)
        self.assertIn("## Sheet: Rewards", slack_text)
        self.assertIn("Lines: 10-14", slack_text)


if __name__ == "__main__":
    unittest.main()
