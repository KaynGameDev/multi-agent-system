from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from app.contracts import (
    validate_assistant_request,
    validate_pending_action_decision,
)
from app.interpretation.intent_parser import (
    DEFAULT_ASSISTANT_REQUEST_LOW_CONFIDENCE_REASON,
    DEFAULT_INTENT_PARSER_MALFORMED_REASON,
    IntentParser,
    build_assistant_request_prompt,
)
from app.interpretation.model_config import IntentParserModelConfig
from app.routing.agent_router import AgentRouter
from app.pending_actions import build_pending_action


class StructuredOutputLLM:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)

    def with_structured_output(self, _schema, **_kwargs):
        return self

    def invoke(self, _prompt):
        if not self.outputs:
            raise RuntimeError("No more queued outputs.")
        current = self.outputs.pop(0)
        if isinstance(current, Exception):
            raise current
        return current


class StructuredValidationErrorThenRawLLM:
    def __init__(self, raw_output: object) -> None:
        self.raw_output = raw_output
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
        return self.raw_output


class IntentParserTests(unittest.TestCase):
    def test_parse_assistant_request_returns_valid_contract(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "assistant_request",
                        "user_goal": "Summarize the active sprint blockers.",
                        "likely_domain": "project_task",
                        "confidence": 0.94,
                        "notes": "Mentions sprint blockers and ownership.",
                    }
                ]
            )
        )

        result = parser.parse_assistant_request("Who owns the current sprint blockers?")

        self.assertEqual(result.type, "assistant_request")
        self.assertEqual(result.likely_domain, "project_task")
        self.assertEqual(result.user_goal, "Summarize the active sprint blockers.")
        self.assertGreaterEqual(result.confidence, 0.94)

    def test_parse_assistant_request_normalizes_whitespace_fields(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "assistant_request",
                        "user_goal": "  Summarize repository docs.  ",
                        "likely_domain": "knowledge",
                        "confidence": 0.91,
                        "notes": "  Documentation request.  ",
                    }
                ]
            )
        )

        result = parser.parse_assistant_request("  explain the repo docs  ")

        self.assertEqual(result.user_goal, "Summarize repository docs.")
        self.assertEqual(result.notes, "Documentation request.")

    def test_parse_assistant_request_accepts_pascal_case_contract_type(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "AssistantRequest",
                        "user_goal": "Record new knowledge into the knowledge base.",
                        "likely_domain": "knowledge_base_builder",
                        "confidence": 0.97,
                        "notes": "The user wants to sync new company knowledge.",
                    }
                ]
            )
        )

        result = parser.parse_assistant_request("我想记录新知识")

        self.assertEqual(result.likely_domain, "knowledge_base_builder")
        self.assertEqual(result.user_goal, "Record new knowledge into the knowledge base.")
        self.assertEqual(result.notes, "The user wants to sync new company knowledge.")

    def test_parse_assistant_request_recovers_when_model_omits_user_goal(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "assistant_request",
                        "likely_domain": "knowledge_base_builder",
                        "confidence": 0.93,
                        "notes": "The user wants to sync company knowledge into the KB flow.",
                    }
                ]
            )
        )

        result = parser.parse_assistant_request("我希望跟你同步一下公司知识")

        self.assertEqual(result.likely_domain, "knowledge_base_builder")
        self.assertEqual(result.user_goal, "我希望跟你同步一下公司知识")
        self.assertGreaterEqual(result.confidence, 0.93)

    def test_parse_assistant_request_recovers_from_raw_json_content(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "raw": AIMessage(
                            content="""```json
{"type":"assistant_request","likely_domain":"knowledge_base_builder","confidence":0.92,"notes":"The user wants to sync company knowledge.","user_goal":"Capture and organize company knowledge."}
```"""
                        ),
                        "parsed": None,
                        "parsing_error": "schema mismatch",
                    }
                ]
            )
        )

        result = parser.parse_assistant_request("我希望跟你同步一下公司知识")

        self.assertEqual(result.likely_domain, "knowledge_base_builder")
        self.assertEqual(result.user_goal, "Capture and organize company knowledge.")
        self.assertGreaterEqual(result.confidence, 0.92)

    def test_parse_assistant_request_recovers_from_raw_llm_after_structured_validation_error(self) -> None:
        llm = StructuredValidationErrorThenRawLLM(
            AIMessage(
                content="""<think>
The user wants to write the discussed content into the knowledge base.
</think>

```json
{"type":"assistant_request","likely_domain":"knowledge_base_builder","confidence":0.96,"notes":"The user explicitly wants KB write behavior.","user_goal":"Write the discussed company knowledge into the knowledge base."}
```"""
            )
        )
        parser = IntentParser(llm)

        result = parser.parse_assistant_request("请把这些内容录入到知识库")

        self.assertEqual(result.likely_domain, "knowledge_base_builder")
        self.assertEqual(result.user_goal, "Write the discussed company knowledge into the knowledge base.")
        self.assertEqual(llm.structured_calls, 1)
        self.assertEqual(llm.raw_calls, 1)

    def test_parse_pending_action_decision_returns_valid_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="select_knowledge_document",
            requested_by_agent="knowledge_agent",
            summary="Select a knowledge document to open.",
            metadata={
                "selection_options": [
                    {
                        "id": "doc_setup",
                        "label": "Setup Guide",
                        "value": "Setup Guide",
                        "payload": {"id": "doc_setup"},
                    }
                ]
            },
        )
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "pending_action_decision",
                        "pending_action_id": action["id"],
                        "decision": "select",
                        "notes": "The user selected the setup guide.",
                        "selected_item_id": "doc_setup",
                        "constraints": [],
                        "confidence": 0.98,
                    }
                ]
            )
        )

        result = parser.parse_pending_action_decision(action, "Open the setup guide.")

        self.assertEqual(result.type, "pending_action_decision")
        self.assertEqual(result.pending_action_id, action["id"])
        self.assertEqual(result.decision, "select")
        self.assertEqual(result.selected_item_id, "doc_setup")

    def test_parse_pending_action_decision_accepts_pascal_case_contract_type(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edits.",
        )
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "PendingActionDecision",
                        "pending_action_id": action["id"],
                        "decision": "approve",
                        "notes": "The user approved the pending action.",
                        "selected_item_id": None,
                        "constraints": [],
                        "confidence": 0.98,
                    }
                ]
            )
        )

        result = parser.parse_pending_action_decision(action, "继续")

        self.assertEqual(result.decision, "approve")
        self.assertEqual(result.pending_action_id, action["id"])

    def test_parse_pending_action_decision_recovers_from_raw_llm_after_structured_validation_error(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edits.",
        )
        llm = StructuredValidationErrorThenRawLLM(
            AIMessage(
                content=f"""<think>
The user approved the pending action.
</think>

```json
{{"type":"pending_action_decision","pending_action_id":"{action['id']}","decision":"approve","notes":"The user approved the change.","selected_item_id":null,"constraints":[],"confidence":0.97}}
```"""
            )
        )
        parser = IntentParser(llm)

        result = parser.parse_pending_action_decision(action, "继续")

        self.assertEqual(result.decision, "approve")
        self.assertEqual(result.pending_action_id, action["id"])
        self.assertEqual(llm.structured_calls, 1)
        self.assertEqual(llm.raw_calls, 1)

    def test_contract_validators_reject_invalid_schema(self) -> None:
        with self.assertRaises(ValidationError):
            validate_assistant_request(
                {
                    "type": "assistant_request",
                    "user_goal": "Route this request.",
                    "likely_domain": "project_task",
                    "confidence": 1.4,
                    "notes": None,
                }
            )

        with self.assertRaises(ValidationError):
            validate_pending_action_decision(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": "pending_123",
                    "decision": "select",
                    "notes": "Pick the first one.",
                    "selected_item_id": None,
                    "constraints": "not-a-list",
                }
            )

    def test_low_confidence_assistant_request_preserves_parsed_domain(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM(
                [
                    {
                        "type": "assistant_request",
                        "user_goal": "Handle the request somehow.",
                        "likely_domain": "knowledge",
                        "confidence": 0.31,
                        "notes": "",
                    }
                ]
            ),
            config=IntentParserModelConfig(confidence_threshold=0.7),
        )

        result = parser.parse_assistant_request("Can you take care of this for me?")

        self.assertEqual(result.likely_domain, "knowledge")
        self.assertEqual(result.confidence, 0.31)

    def test_malformed_model_output_falls_back_to_unclear_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edits.",
        )
        parser = IntentParser(StructuredOutputLLM([{"unexpected": "payload"}]))

        result = parser.parse_pending_action_decision(action, "Sounds good.")

        self.assertEqual(result.decision, "unclear")
        self.assertEqual(result.pending_action_id, action["id"])
        self.assertEqual(result.notes, DEFAULT_INTENT_PARSER_MALFORMED_REASON)

    def test_agent_router_uses_parser_config_confidence_threshold_by_default(self) -> None:
        parser = IntentParser(
            StructuredOutputLLM([]),
            config=IntentParserModelConfig(confidence_threshold=0.83),
        )

        router = AgentRouter(parser)

        self.assertEqual(router.confidence_threshold, 0.83)

    def test_assistant_request_prompt_documents_knowledge_sync_as_builder_work(self) -> None:
        prompt = build_assistant_request_prompt(
            "我希望跟你同步一下公司知识",
            recent_messages=["我们刚才在讨论新流程。"],
            routing_context={
                "interface_name": "web",
                "uploaded_files_count": 0,
                "conversion_session_active": False,
            },
        )

        self.assertIn("knowledge_base_builder", prompt)
        self.assertIn("`knowledge` is for consuming or querying existing knowledge.", prompt)
        self.assertIn("`knowledge_base_builder` is for producing, curating, syncing, or storing new knowledge-base content.", prompt)
        self.assertIn("我希望跟你同步一下公司知识", prompt)
        self.assertIn("请把这些内容录入到知识库", prompt)
        self.assertIn("不是记录到对话中，我要你写入知识库", prompt)

    def test_assistant_request_prompt_mentions_natural_chinese_routing_examples(self) -> None:
        prompt = build_assistant_request_prompt(
            "知识库里目前都知道什么，我想补充还没有的部分",
            recent_messages=["user: 关于我们公司的知识，你能教我些什么"],
            routing_context={
                "interface_name": "web",
                "uploaded_files_count": 0,
                "conversion_session_active": False,
            },
        )

        self.assertIn("关于我们公司的知识，你能教我些什么", prompt)
        self.assertIn("好的，请问你还需要我同步哪方面的知识呢", prompt)
        self.assertIn("知识库里目前都知道什么，我想补充还没有的部分", prompt)


if __name__ == "__main__":
    unittest.main()
