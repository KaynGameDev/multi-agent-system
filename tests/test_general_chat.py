from __future__ import annotations

import unittest

from pydantic import BaseModel, ValidationError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.general_chat.agent import GeneralChatAgentNode, GeneralChatReply, build_general_chat_prompt
from interfaces.web.conversations import TRANSCRIPT_TYPE_COMPACT_BOUNDARY


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


class UsageTrackingLLM:
    def invoke(self, _messages):
        return AIMessage(
            content="Usage-aware reply.",
            usage_metadata={"input_tokens": 50, "output_tokens": 12, "total_tokens": 62},
            response_metadata={"stop_reason": "end_turn"},
            additional_kwargs={"provider": "test"},
            id="ai-usage-1",
        )


class CapturingLLM:
    def __init__(self) -> None:
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = list(messages)
        return AIMessage(content="Tail reply.")


class GeneralChatAgentTests(unittest.TestCase):
    def test_general_chat_reply_schema_requires_non_empty_answer(self) -> None:
        with self.assertRaises(ValidationError):
            GeneralChatReply.model_validate({"final_answer": "   "})

    def test_general_chat_reply_schema_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            GeneralChatReply.model_validate(
                {
                    "final_answer": "Hello! I'm Jade.",
                    "role": "assistant",
                }
            )

    def test_general_chat_prompt_forbids_claiming_persistent_save_without_real_action(self) -> None:
        prompt = build_general_chat_prompt(
            {
                "messages": [],
                "interface_name": "web",
            },
            agent_name="general_chat_agent",
        )

        self.assertIn("Do not claim that content was saved", prompt)
        self.assertIn("not that it has been saved to a file or knowledge base", prompt)

    def test_general_chat_prompt_redirects_kb_write_requests_to_builder_flow(self) -> None:
        prompt = build_general_chat_prompt(
            {
                "messages": [],
                "interface_name": "web",
            },
            agent_name="general_chat_agent",
        )

        self.assertIn("write, save, sync, capture, update, or record new company knowledge", prompt)
        self.assertIn("knowledge-base-builder flow", prompt)

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
        self.assertEqual(llm.structured_calls, 2)
        self.assertEqual(llm.raw_calls, 2)

    def test_general_chat_preserves_usage_metadata_on_ai_message(self) -> None:
        node = GeneralChatAgentNode(UsageTrackingLLM(), agent_name="general_chat_agent")

        result = node(
            {
                "messages": [],
                "interface_name": "web",
            }
        )

        reply_message = result["messages"][-1]
        self.assertEqual(reply_message.content, "Usage-aware reply.")
        self.assertEqual(reply_message.id, "ai-usage-1")
        self.assertEqual(
            reply_message.usage_metadata,
            {"input_tokens": 50, "output_tokens": 12, "total_tokens": 62},
        )
        self.assertEqual(reply_message.response_metadata, {"stop_reason": "end_turn"})
        self.assertEqual(reply_message.additional_kwargs, {"provider": "test"})

    def test_general_chat_request_builder_uses_only_messages_after_boundary(self) -> None:
        llm = CapturingLLM()
        node = GeneralChatAgentNode(llm, agent_name="general_chat_agent")

        result = node(
            {
                "messages": [
                    HumanMessage(content="older request"),
                    SystemMessage(
                        content="",
                        additional_kwargs={"transcript_type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY},
                    ),
                    HumanMessage(content="latest request"),
                ],
                "interface_name": "web",
            }
        )

        self.assertEqual(result["assistant_response"]["content"], "Tail reply.")
        self.assertIsNotNone(llm.last_messages)
        self.assertEqual(len(llm.last_messages), 2)
        self.assertTrue(isinstance(llm.last_messages[0], SystemMessage))
        self.assertEqual(llm.last_messages[1].content, "latest request")
        self.assertNotIn(
            TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
            str(getattr(llm.last_messages[1], "additional_kwargs", {})),
        )


if __name__ == "__main__":
    unittest.main()
