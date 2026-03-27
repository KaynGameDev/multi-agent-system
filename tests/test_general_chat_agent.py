from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agents.workers.general_chat_agent import GeneralChatAgentNode


class DummyLLM:
    def __init__(self) -> None:
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content="hello")


class GeneralChatAgentTests(unittest.TestCase):
    def test_slack_prompt_uses_slack_specific_format_guidance(self) -> None:
        llm = DummyLLM()
        node = GeneralChatAgentNode(llm)

        result = node(
            {
                "interface_name": "slack",
                "messages": [HumanMessage(content="hello there")],
            }
        )

        self.assertEqual(result["messages"][0].content, "hello")
        prompt = llm.last_messages[0].content
        self.assertIn("The current interface is Slack.", prompt)
        self.assertIn("Slack boundary converts it to mrkdwn", prompt)
        self.assertIn("Avoid raw Slack entities", prompt)
        self.assertIn("Reply in the same language as the user's latest message", prompt)

    def test_unknown_interface_uses_default_format_guidance(self) -> None:
        llm = DummyLLM()
        node = GeneralChatAgentNode(llm)

        node(
            {
                "interface_name": "unknown",
                "messages": [HumanMessage(content="hello there")],
            }
        )

        prompt = llm.last_messages[0].content
        self.assertIn("Write concise, plain Markdown", prompt)
        self.assertNotIn("The current interface is Slack.", prompt)


if __name__ == "__main__":
    unittest.main()
