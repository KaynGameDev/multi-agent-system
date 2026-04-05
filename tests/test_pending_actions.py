from __future__ import annotations

import unittest

from app.pending_actions import (
    build_pending_action,
    interpret_pending_action_reply,
    validate_execution_contract,
)


class PendingActionTests(unittest.TestCase):
    def test_simple_approval_reply_becomes_execute_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="run_skill",
            requested_by_agent="general_chat_agent",
            summary="Run the review skill.",
            target_scope={"skill_name": "review-kb-doc"},
        )

        contract = interpret_pending_action_reply(action, "go ahead")
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "approve")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "execute")

    def test_scope_limited_reply_becomes_modified_execute_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply edits across backend and frontend.",
            target_scope={"modules": ["backend", "frontend"]},
        )

        contract = interpret_pending_action_reply(action, "yes but only backend")
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "execute")
        self.assertEqual(validation["normalized_scope"]["modules"], ["backend"])

    def test_diff_request_becomes_modified_non_execute_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
            target_scope={"files": ["src/main.py"]},
        )

        contract = interpret_pending_action_reply(action, "show me the diff first")
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertIn("diff", contract["requested_outputs"])
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "request_revision")

    def test_unsupported_scope_modification_stays_non_executable(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="delete_file",
            requested_by_agent="general_chat_agent",
            summary="Delete the generated file.",
            target_scope={"files": ["build/output.txt"]},
        )

        contract = interpret_pending_action_reply(action, "yes but only backend")
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["runtime_action"], "ask_clarification")


if __name__ == "__main__":
    unittest.main()
