from __future__ import annotations

import unittest

from langchain_core.messages import HumanMessage

from app.contracts import validate_pending_action_decision
from app.pending_actions import (
    build_pending_action,
    resolve_pending_action_decision,
)
from app.routing.pending_action_router import (
    PendingActionRouter,
    build_pending_action_resolution_key,
    resolve_pending_action_turn_from_state,
)


class StaticDecisionParser:
    def __init__(self, payload) -> None:
        self.payload = payload

    def parse_pending_action_decision(self, _action, _user_message):
        return self.payload


class ReplyMapDecisionParser:
    def __init__(self, mapping: dict[str, dict], *, default: dict | None = None) -> None:
        self.mapping = {str(key): dict(value) for key, value in mapping.items()}
        self.default = dict(default or {})
        self.calls = 0

    def parse_pending_action_decision(self, action, user_message):
        self.calls += 1
        payload = dict(self.mapping.get(str(user_message), self.default))
        payload.setdefault("type", "pending_action_decision")
        payload.setdefault("pending_action_id", str(action.get("id", "")).strip())
        payload.setdefault("decision", "unclear")
        payload.setdefault("notes", None)
        payload.setdefault("selected_item_id", None)
        payload.setdefault("constraints", [])
        return payload


class PendingActionTests(unittest.TestCase):
    def test_approval_reply_continues_pending_action(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        router = PendingActionRouter(
            StaticDecisionParser(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "approve",
                    "notes": "The user approved the change.",
                    "selected_item_id": None,
                    "constraints": [],
                }
            )
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="ok")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.allow_fresh_routing)
        self.assertEqual(result.execution_contract["decision"], "approve")
        self.assertEqual(result.validation["runtime_action"], "execute")

    def test_rejection_reply_cancels_pending_action(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged package.",
        )
        router = PendingActionRouter(
            StaticDecisionParser(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "reject",
                    "notes": "The user rejected the publish.",
                    "selected_item_id": None,
                    "constraints": [],
                }
            )
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="not yet")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.validation["runtime_action"], "cancel")
        self.assertEqual(result.validation["next_status"], "rejected")

    def test_selection_reply_resolves_shared_pending_action_option(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="select_project_task",
            requested_by_agent="project_task_agent",
            summary="Select a task to inspect.",
            metadata={
                "selection_options": [
                    {"id": "task_1", "label": "Task One", "value": "Task One", "payload": {"task_id": "task_1"}},
                    {"id": "task_2", "label": "Task Two", "value": "Task Two", "payload": {"task_id": "task_2"}},
                ]
            },
        )
        router = PendingActionRouter(
            StaticDecisionParser(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "select",
                    "notes": "The user selected the second task.",
                    "selected_item_id": "task_2",
                    "constraints": [],
                }
            )
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="the second one")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.validation["runtime_action"], "select")
        self.assertEqual(result.validation["selected_index"], 1)
        self.assertEqual(result.validation["selected_option"]["id"], "task_2")

    def test_modify_reply_applies_valid_constraints(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply edits across backend and frontend.",
            target_scope={"modules": ["backend", "frontend"]},
        )
        router = PendingActionRouter(
            StaticDecisionParser(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "modify",
                    "notes": "Only continue for backend.",
                    "selected_item_id": None,
                    "constraints": ["modules:backend"],
                }
            )
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="backend only")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.validation["runtime_action"], "execute")
        self.assertEqual(result.validation["normalized_scope"]["modules"], ["backend"])

    def test_unrelated_reply_allows_fresh_routing(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        router = PendingActionRouter(
            StaticDecisionParser(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "unrelated",
                    "notes": "The user started a new topic.",
                    "selected_item_id": None,
                    "constraints": [],
                }
            )
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="What docs do we have?")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.allow_fresh_routing)
        self.assertIsNone(result.execution_contract)
        self.assertIsNone(result.validation)

    def test_unclear_reply_routes_to_clarification(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        router = PendingActionRouter(
            StaticDecisionParser(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "unclear",
                    "notes": "The reply was ambiguous.",
                    "selected_item_id": None,
                    "constraints": [],
                }
            )
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="maybe")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.allow_fresh_routing)
        self.assertEqual(result.validation["runtime_action"], "ask_clarification")
        self.assertFalse(result.validation["valid"])

    def test_malformed_parser_output_falls_back_to_clarification(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        router = PendingActionRouter(StaticDecisionParser({"unexpected": "payload"}))

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="go ahead")],
                "pending_action": action,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.pending_action_decision.decision, "unclear")
        self.assertIn("malformed", str(result.pending_action_decision.notes).lower())
        self.assertEqual(result.validation["runtime_action"], "ask_clarification")

    def test_cached_pending_action_resolution_reused_within_same_turn(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        state = {
            "messages": [HumanMessage(content="approve")],
            "pending_action": action,
        }
        resolution_key = build_pending_action_resolution_key(state, action)
        parser = ReplyMapDecisionParser(
            {
                "approve": {
                    "decision": "approve",
                    "notes": "The user approved the change.",
                }
            }
        )

        first_result = resolve_pending_action_turn_from_state(
            state,
            pending_action_router=PendingActionRouter(parser),
        )

        self.assertIsNotNone(first_result)
        assert first_result is not None
        self.assertEqual(parser.calls, 1)

        second_result = resolve_pending_action_turn_from_state(
            {
                **state,
                "pending_action_decision": first_result.pending_action_decision.model_dump(),
                "pending_action_resolution_key": resolution_key,
                "execution_contract": first_result.execution_contract,
            },
            pending_action_router=PendingActionRouter(parser),
        )

        self.assertIsNotNone(second_result)
        assert second_result is not None
        self.assertEqual(parser.calls, 1)
        self.assertEqual(second_result.pending_action_decision.decision, "approve")
        self.assertEqual(second_result.execution_contract["decision"], "approve")

    def test_stale_pending_action_decision_is_reparsed_on_next_turn(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        parser = ReplyMapDecisionParser(
            {
                "maybe": {
                    "decision": "unclear",
                    "notes": "The reply was ambiguous.",
                },
                "approve": {
                    "decision": "approve",
                    "notes": "The user approved the change.",
                },
            }
        )
        router = PendingActionRouter(parser)
        first_state = {
            "messages": [HumanMessage(content="maybe")],
            "pending_action": action,
        }
        first_result = resolve_pending_action_turn_from_state(
            first_state,
            pending_action_router=router,
        )

        self.assertIsNotNone(first_result)
        assert first_result is not None
        self.assertEqual(first_result.pending_action_decision.decision, "unclear")
        self.assertEqual(first_result.validation["runtime_action"], "ask_clarification")

        second_result = resolve_pending_action_turn_from_state(
            {
                "messages": [
                    HumanMessage(content="maybe"),
                    HumanMessage(content="approve"),
                ],
                "pending_action": action,
                "pending_action_decision": first_result.pending_action_decision.model_dump(),
                "pending_action_resolution_key": first_result.pending_action_resolution_key,
                "execution_contract": first_result.execution_contract,
            },
            pending_action_router=router,
        )

        self.assertIsNotNone(second_result)
        assert second_result is not None
        self.assertEqual(parser.calls, 2)
        self.assertEqual(second_result.pending_action_decision.decision, "approve")
        self.assertEqual(second_result.validation["runtime_action"], "execute")
        self.assertEqual(second_result.execution_contract["decision"], "approve")

    def test_cached_pending_action_resolution_with_mismatched_key_is_ignored(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        parser = ReplyMapDecisionParser(
            {
                "approve": {
                    "decision": "approve",
                    "notes": "The user approved the change.",
                }
            }
        )

        result = resolve_pending_action_turn_from_state(
            {
                "messages": [HumanMessage(content="approve")],
                "pending_action": action,
                "pending_action_decision": {
                    "type": "pending_action_decision",
                    "pending_action_id": action["id"],
                    "decision": "unclear",
                    "notes": "Old ambiguous reply.",
                    "selected_item_id": None,
                    "constraints": [],
                },
                "pending_action_resolution_key": "pending_old:1:stale",
            },
            pending_action_router=PendingActionRouter(parser),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(parser.calls, 1)
        self.assertEqual(result.pending_action_decision.decision, "approve")
        self.assertEqual(result.execution_contract["decision"], "approve")

    def test_wrong_pending_action_id_stays_safe(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
        )
        decision = validate_pending_action_decision(
            {
                "type": "pending_action_decision",
                "pending_action_id": "pending_wrong",
                "decision": "approve",
                "notes": "The user approved the action.",
                "selected_item_id": None,
                "constraints": [],
            }
        )

        result = resolve_pending_action_decision(action, decision, user_text="approve")
        validation = result["validation"]

        self.assertIsNotNone(validation)
        assert validation is not None
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["runtime_action"], "ask_clarification")
        self.assertIn("did not match", validation["reason"])


if __name__ == "__main__":
    unittest.main()
