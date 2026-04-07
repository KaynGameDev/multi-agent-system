from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from app.contracts import (
    build_skill_invocation_contract,
    normalize_tool_result_envelope,
    tool_invocation_to_tool_call,
)
from app.messages import extract_final_text
from app.skill_runtime import build_skill_runtime_state, get_active_skill_invocation_contracts


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


if __name__ == "__main__":
    unittest.main()
