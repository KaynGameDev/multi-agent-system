from __future__ import annotations

import json
import unittest

from agents.knowledge.rendering import is_knowledge_payload, render_knowledge_payload
from langchain_core.messages import AIMessage, ToolMessage

from app.contracts import (
    build_routing_decision,
    build_skill_invocation_contract,
    normalize_tool_result_envelope,
    tool_invocation_to_tool_call,
)
from app.messages import extract_final_text
from app.skill_runtime import build_skill_runtime_state, get_active_skill_invocation_contracts
from app.tool_runtime import extract_tool_result_from_message


class RuntimeContractTests(unittest.TestCase):
    def test_extract_final_text_prefers_assistant_response(self) -> None:
        final_state = {
            "assistant_response": {
                "kind": "text",
                "content": "From the shared contract.",
            },
            "messages": [AIMessage(content="Fallback content.")],
        }

        self.assertEqual(extract_final_text(final_state), "From the shared contract.")

    def test_extract_final_text_falls_back_to_last_message_when_contract_is_empty(self) -> None:
        final_state = {
            "assistant_response": {
                "kind": "invoke_tool",
                "content": "",
            },
            "messages": [AIMessage(content="Fallback content.")],
        }

        self.assertEqual(extract_final_text(final_state), "Fallback content.")

    def test_tool_result_envelope_wraps_legacy_payload(self) -> None:
        envelope = normalize_tool_result_envelope(
            {
                "ok": True,
                "tasks": [{"content": "Ship durable memory"}],
                "match_count": 1,
            },
            tool_name="project_task",
        )

        self.assertEqual(envelope["tool_name"], "project_task")
        self.assertEqual(envelope["status"], "ok")
        self.assertEqual(envelope["payload"]["match_count"], 1)
        self.assertEqual(envelope["payload"]["tasks"][0]["content"], "Ship durable memory")

    def test_skill_invocation_contract_normalizes_fork_mode(self) -> None:
        contract = build_skill_invocation_contract(
            skill_id="shared-inline",
            mode="forked",
            target_agent="general_chat_agent",
            source="gateway",
            reason="Skill routing test",
        )

        self.assertEqual(contract["skill_id"], "shared-inline")
        self.assertEqual(contract["mode"], "fork")
        self.assertEqual(contract["target_agent"], "general_chat_agent")
        self.assertEqual(contract["source"], "gateway")

    def test_tool_invocation_contract_becomes_langchain_tool_call(self) -> None:
        tool_call = tool_invocation_to_tool_call(
            {
                "tool_name": "read_knowledge_document",
                "arguments": {"document_name": "SetupGuide"},
                "tool_call_id": "call_123",
                "status": "requested",
            }
        )

        self.assertEqual(tool_call["name"], "read_knowledge_document")
        self.assertEqual(tool_call["args"]["document_name"], "SetupGuide")
        self.assertEqual(tool_call["id"], "call_123")

    def test_tool_runtime_adapter_recovers_tool_metadata_from_tool_message(self) -> None:
        request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_knowledge_document",
                    "args": {"document_name": "SetupGuide"},
                    "id": "call_read_setup",
                }
            ],
        )
        result_message = ToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "document": {"name": "SetupGuide"},
                    "content": "Setup excerpt",
                }
            ),
            tool_call_id="call_read_setup",
        )

        envelope = extract_tool_result_from_message(
            result_message,
            messages=[request, result_message],
            source="knowledge_agent",
            reason="Tool runtime adapter test",
        )

        self.assertIsNotNone(envelope)
        assert envelope is not None
        self.assertEqual(envelope["tool_name"], "read_knowledge_document")
        self.assertEqual(envelope["tool_id"], "knowledge.read_document")
        self.assertEqual(envelope["tool_family"], "knowledge_read")
        self.assertEqual(envelope["arguments"]["document_name"], "SetupGuide")
        self.assertEqual(envelope["execution_backend"], "langgraph_tool_node")

    def test_tool_runtime_adapter_prefers_matched_request_name_over_group_alias(self) -> None:
        request = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_documents",
                    "args": {"query": "setup", "limit": 5},
                    "id": "call_search_setup",
                }
            ],
        )
        result_message = ToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "query": "setup",
                    "documents": [{"name": "SetupGuide"}],
                    "match_count": 1,
                }
            ),
            tool_call_id="call_search_setup",
        )

        envelope = extract_tool_result_from_message(
            result_message,
            messages=[request, result_message],
            tool_name="knowledge_documents",
            source="knowledge_agent",
            reason="Grouped knowledge tool alias should not override the matched tool request name.",
        )

        self.assertIsNotNone(envelope)
        assert envelope is not None
        self.assertEqual(envelope["tool_name"], "search_knowledge_documents")
        self.assertEqual(envelope["tool_id"], "knowledge.search_documents")
        self.assertEqual(envelope["arguments"]["query"], "setup")
        self.assertEqual(envelope["arguments"]["limit"], 5)

    def test_retrieval_payload_is_a_knowledge_payload_but_not_directly_rendered(self) -> None:
        payload = {
            "ok": True,
            "query": "How does setup work?",
            "retrieved_context": "Retrieved context block",
            "retrieved_chunks": [
                {
                    "title": "Setup Guide",
                    "path": "knowledge/Docs/00_Shared/SetupGuide.md",
                }
            ],
        }

        self.assertTrue(is_knowledge_payload(payload))
        self.assertIsNone(render_knowledge_payload(payload))

    def test_skill_runtime_state_tracks_active_contracts_for_selected_agent(self) -> None:
        contracts = [
            build_skill_invocation_contract(
                skill_id="shared-inline",
                name="Shared Inline",
                target_agent="knowledge_agent",
                source="gateway.auto_skill_match",
                reason="Token overlap matched: shared, inline.",
            ),
            build_skill_invocation_contract(
                skill_id="forked-review",
                mode="forked",
                target_agent="knowledge_base_builder_agent",
                source="gateway.explicit_skill_request",
                reason="Explicit forked skill request.",
            ),
        ]

        active_contracts = get_active_skill_invocation_contracts(
            {"skill_invocation_contracts": contracts},
            agent_name="knowledge_agent",
        )
        runtime_state = build_skill_runtime_state(active_contracts, agent_name="knowledge_agent")

        self.assertEqual(len(active_contracts), 1)
        self.assertEqual(active_contracts[0]["skill_id"], "shared-inline")
        self.assertEqual(runtime_state["active_skill_invocation_contracts"][0]["target_agent"], "knowledge_agent")
        self.assertEqual(runtime_state["skill_execution_diagnostics"][0]["mode"], "inline")

    def test_routing_decision_tracks_policy_step(self) -> None:
        decision = build_routing_decision(
            "knowledge_base_builder_agent",
            reason="AssistantRequest mapped the request to the KB writer.",
            policy_step="assistant_request_domain",
            warnings=["none"],
            diagnostics=[{"kind": "selected", "policy_step": "assistant_request_domain"}],
        )

        self.assertEqual(decision["route"], "knowledge_base_builder_agent")
        self.assertEqual(decision["policy_step"], "assistant_request_domain")


if __name__ == "__main__":
    unittest.main()
