from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from app.contracts import validate_pending_action_decision
from app.pending_action_responder import run_pending_action_response
from app.routing.pending_action_router import PendingActionTurnResult


def build_pending_action() -> dict[str, object]:
    return {
        "id": "pending_123",
        "session_id": "thread-1",
        "type": "select_project_task",
        "requested_by_agent": "project_task_agent",
        "summary": "Select a task.",
        "status": "awaiting_confirmation",
        "target_scope": {},
        "metadata": {"prompt_context": "Reply with a selection."},
    }


def build_turn(
    *,
    pending_action: dict[str, object],
    decision: str = "approve",
    execution_contract: dict[str, object] | None = None,
    validation: dict[str, object] | None = None,
) -> PendingActionTurnResult:
    return PendingActionTurnResult(
        pending_action=pending_action,
        pending_action_decision=validate_pending_action_decision(
            {
                "type": "pending_action_decision",
                "pending_action_id": pending_action["id"],
                "decision": decision,
                "notes": "resolved",
                "selected_item_id": "task-1" if decision == "select" else None,
                "constraints": [],
            }
        ),
        pending_action_resolution_key="pending_123:1:hash",
        execution_contract=execution_contract,
        validation=validation,
        allow_fresh_routing=False,
    )


class PendingActionResponderTests(unittest.TestCase):
    def test_missing_validation_returns_cannot_validate_response(self) -> None:
        pending_action = build_pending_action()
        result = run_pending_action_response(
            pending_action=pending_action,
            pending_action_turn=build_turn(pending_action=pending_action, execution_contract=None, validation=None),
            cannot_validate_text="Could not validate.",
            cancel_text="Cancelled.",
            build_clarification_text=lambda _action, _validation: "clarify",
            on_valid_resolution=lambda _action, _decision, _validation: {"unexpected": True},
        )

        self.assertEqual(result["messages"][0].content, "Could not validate.")
        self.assertEqual(result["assistant_response"]["kind"], "await_confirmation")
        self.assertEqual(result["pending_action"], pending_action)

    def test_cancel_clears_pending_action(self) -> None:
        pending_action = build_pending_action()
        result = run_pending_action_response(
            pending_action=pending_action,
            pending_action_turn=build_turn(
                pending_action=pending_action,
                decision="reject",
                execution_contract={"decision": "reject"},
                validation={"valid": True, "runtime_action": "cancel", "next_status": "rejected"},
            ),
            cannot_validate_text="Could not validate.",
            cancel_text="Cancelled.",
            build_clarification_text=lambda _action, _validation: "clarify",
            on_valid_resolution=lambda _action, _decision, _validation: {"unexpected": True},
            cancel_result_updates={"execution_contract": None},
        )

        self.assertEqual(result["messages"][0].content, "Cancelled.")
        self.assertEqual(result["assistant_response"]["kind"], "text")
        self.assertIsNone(result["pending_action"])
        self.assertIn("execution_contract", result)
        self.assertIsNone(result["execution_contract"])

    def test_invalid_validation_keeps_updated_pending_action(self) -> None:
        pending_action = build_pending_action()
        contract = {"decision": "unclear", "reply_text": "maybe"}
        validation = {
            "valid": False,
            "runtime_action": "ask_clarification",
            "next_status": "ask_clarification",
            "reason": "Ambiguous reply.",
            "normalized_scope": {"files": ["a.md"]},
        }
        result = run_pending_action_response(
            pending_action=pending_action,
            pending_action_turn=build_turn(
                pending_action=pending_action,
                decision="unclear",
                execution_contract=contract,
                validation=validation,
            ),
            cannot_validate_text="Could not validate.",
            cancel_text="Cancelled.",
            build_clarification_text=lambda _action, current_validation: f"clarify:{current_validation['reason']}",
            on_valid_resolution=lambda _action, _decision, _validation: {"unexpected": True},
        )

        self.assertEqual(result["messages"][0].content, "clarify:Ambiguous reply.")
        self.assertEqual(result["assistant_response"]["kind"], "await_confirmation")
        self.assertEqual(result["pending_action"]["status"], "ask_clarification")
        self.assertEqual(result["pending_action"]["target_scope"], {"files": ["a.md"]})
        self.assertEqual(result["pending_action"]["metadata"]["last_contract"], contract)

    def test_valid_resolution_delegates_to_callback_without_modifying_result(self) -> None:
        pending_action = build_pending_action()
        contract = {"decision": "approve", "reply_text": "go ahead"}
        validation = {
            "valid": True,
            "runtime_action": "execute",
            "next_status": "approved",
            "normalized_scope": {"files": ["a.md"]},
        }
        captured: dict[str, object] = {}
        expected_result = {
            "messages": [AIMessage(content="delegated")],
            "assistant_response": {"kind": "text", "content": "delegated"},
            "pending_action": None,
            "extra": "sentinel",
        }

        def on_valid_resolution(updated_action, decision, current_validation):
            captured["updated_action"] = updated_action
            captured["decision"] = decision
            captured["validation"] = current_validation
            return expected_result

        result = run_pending_action_response(
            pending_action=pending_action,
            pending_action_turn=build_turn(
                pending_action=pending_action,
                execution_contract=contract,
                validation=validation,
            ),
            cannot_validate_text="Could not validate.",
            cancel_text="Cancelled.",
            build_clarification_text=lambda _action, _validation: "clarify",
            on_valid_resolution=on_valid_resolution,
        )

        self.assertIs(result, expected_result)
        self.assertEqual(captured["updated_action"]["status"], "approved")
        self.assertEqual(captured["updated_action"]["target_scope"], {"files": ["a.md"]})
        self.assertEqual(captured["updated_action"]["metadata"]["last_contract"], contract)
        self.assertEqual(captured["decision"].decision, "approve")
        self.assertEqual(captured["validation"], validation)


if __name__ == "__main__":
    unittest.main()
