from __future__ import annotations

import unittest

from pydantic import BaseModel
from langchain_core.messages import AIMessage

from agents.general_chat.agent import GeneralChatAgentNode


class StructuredContentLLM:
    def invoke(self, _messages):
        return AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "Hello! I'm Jade. How can I help you today?",
                    "extras": {"signature": "signed-block"},
                }
            ]
        )


class StructuredReply(BaseModel):
    final_answer: str = ""


class StructuredOutputLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return StructuredReply(final_answer="Hello! I'm Jade. How can I help you today?")


class FallbackStructuredOutputLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.raw_calls = 0

    def with_structured_output(self, _schema):
        parent = self

        class BrokenStructuredInvoker:
            def invoke(self, _messages):
                parent.structured_calls += 1
                StructuredReply.model_validate_json("<think>not json</think>")

        return BrokenStructuredInvoker()

    def invoke(self, _messages):
        self.raw_calls += 1
        return AIMessage(content="Hello from the plain-text fallback.")


class GeneralChatAgentTests(unittest.TestCase):
    def test_general_chat_normalizes_structured_content_blocks(self) -> None:
        node = GeneralChatAgentNode(StructuredContentLLM(), agent_name="general_chat_agent")

        result = node(
            {
                "messages": [],
                "interface_name": "web",
            }
        )

        self.assertEqual(
            result["assistant_response"]["content"],
            "Hello! I'm Jade. How can I help you today?",
        )
        self.assertNotIn("[{", result["assistant_response"]["content"])

    def test_general_chat_prefers_structured_output_contract(self) -> None:
        node = GeneralChatAgentNode(StructuredOutputLLM(), agent_name="general_chat_agent")

        result = node(
            {
                "messages": [],
                "interface_name": "web",
            }
        )

        self.assertEqual(
            result["assistant_response"]["content"],
            "Hello! I'm Jade. How can I help you today?",
        )
        self.assertEqual(result["messages"][-1].content, "Hello! I'm Jade. How can I help you today?")

    def test_general_chat_falls_back_to_plain_text_when_structured_output_is_invalid(self) -> None:
        llm = FallbackStructuredOutputLLM()
        node = GeneralChatAgentNode(llm, agent_name="general_chat_agent")

        first_result = node(
            {
                "messages": [],
                "interface_name": "web",
            }
        )
        second_result = node(
            {
                "messages": [],
                "interface_name": "web",
            }
        )

        self.assertEqual(first_result["assistant_response"]["content"], "Hello from the plain-text fallback.")
        self.assertEqual(second_result["assistant_response"]["content"], "Hello from the plain-text fallback.")
        self.assertEqual(llm.structured_calls, 1)
        self.assertEqual(llm.raw_calls, 2)


if __name__ == "__main__":
    unittest.main()
