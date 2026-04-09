from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from app.pending_action_parser import LLMPendingActionInterpreter
from app.pending_actions import (
    build_pending_action,
    interpret_pending_action_reply,
    prepare_pending_action_reply_input,
    validate_execution_contract,
)


class FailIfCalledInterpreter:
    def parse_pending_action_reply(self, _action, _prepared_input):
        raise AssertionError("The LLM interpreter should not run for exact deterministic selections.")


class StaticInterpreter:
    def __init__(self, parsed_reply: dict) -> None:
        self.parsed_reply = dict(parsed_reply)

    def parse_pending_action_reply(self, _action, _prepared_input):
        return dict(self.parsed_reply)


class RaisingInterpreter:
    def parse_pending_action_reply(self, _action, _prepared_input):
        raise RuntimeError("parser unavailable")


class StructuredOutputLLM:
    def __init__(self, raw_output=None, *, error: Exception | None = None) -> None:
        self.raw_output = raw_output
        self.error = error

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        if self.error is not None:
            raise self.error
        return self.raw_output


class SequencedLLM:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        if not self.outputs:
            raise RuntimeError("No more queued outputs.")
        current = self.outputs.pop(0)
        if isinstance(current, Exception):
            raise current
        return current


class PendingActionTests(unittest.TestCase):
    def test_exact_numeric_selection_bypasses_interpreter(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="select_knowledge_document",
            requested_by_agent="knowledge_agent",
            summary="Select a document to open.",
            metadata={
                "selection_options": [
                    {
                        "label": "Setup Guide",
                        "aliases": ["1"],
                        "value": "Setup Guide",
                        "payload": {"document_name": "Setup Guide"},
                    }
                ]
            },
        )

        contract = interpret_pending_action_reply(
            action,
            "1",
            interpreter=FailIfCalledInterpreter(),
        )
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "select")
        self.assertEqual(contract["interpretation_source"], "deterministic_selection")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "select")

    def test_exact_alias_selection_bypasses_interpreter(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="select_knowledge_document",
            requested_by_agent="knowledge_agent",
            summary="Select a document to open.",
            metadata={
                "selection_options": [
                    {
                        "label": "Setup Guide",
                        "aliases": ["setup guide", "guide"],
                        "value": "Setup Guide",
                        "payload": {"document_name": "Setup Guide"},
                    }
                ]
            },
        )

        contract = interpret_pending_action_reply(
            action,
            "setup guide",
            interpreter=FailIfCalledInterpreter(),
        )
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "select")
        self.assertEqual(contract["selected_option"]["payload"]["document_name"], "Setup Guide")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "select")

    def test_simple_approval_from_interpreter_becomes_execute_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="run_skill",
            requested_by_agent="general_chat_agent",
            summary="Run the review skill.",
            target_scope={"skill_name": "review-kb-doc"},
        )
        interpreter = StaticInterpreter(
            {
                "decision": "approve",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": None,
                "should_execute": True,
                "reason": "The user approved the action.",
                "confidence": 0.97,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "go ahead", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "approve")
        self.assertEqual(contract["interpretation_source"], "llm_parser")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "execute")

    def test_rejection_from_interpreter_becomes_cancel(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = StaticInterpreter(
            {
                "decision": "reject",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": None,
                "should_execute": False,
                "reason": "The user rejected the action.",
                "confidence": 0.92,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "cancel", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "reject")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "cancel")

    def test_scope_limited_reply_becomes_narrowed_execute_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply edits across backend and frontend.",
            target_scope={"modules": ["backend", "frontend"]},
        )
        interpreter = StaticInterpreter(
            {
                "decision": "modify",
                "requested_outputs": [],
                "target_scope": {"modules": ["backend"]},
                "selected_index": None,
                "should_execute": True,
                "reason": "The user approved a narrowed backend-only execution.",
                "confidence": 0.94,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "yes but only backend", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "execute")
        self.assertEqual(validation["normalized_scope"]["modules"], ["backend"])

    def test_diff_request_becomes_request_revision_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
            target_scope={"files": ["src/main.py"]},
        )
        interpreter = StaticInterpreter(
            {
                "decision": "modify",
                "requested_outputs": ["diff"],
                "target_scope": {},
                "selected_index": None,
                "should_execute": False,
                "reason": "The user asked to see the diff first.",
                "confidence": 0.95,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "show me the diff first", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertIn("diff", contract["requested_outputs"])
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "request_revision")
        self.assertEqual(validation["next_status"], "request_revision")

    def test_summary_request_becomes_request_revision_contract(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
            target_scope={"files": ["src/main.py"]},
        )
        interpreter = StaticInterpreter(
            {
                "decision": "modify",
                "requested_outputs": ["summary"],
                "target_scope": {},
                "selected_index": None,
                "should_execute": False,
                "reason": "The user asked to see a summary first.",
                "confidence": 0.95,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "show me a summary", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertEqual(contract["requested_outputs"], ["summary"])
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "request_revision")
        self.assertEqual(validation["next_status"], "request_revision")

    def test_unsupported_scope_modification_stays_non_executable(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="delete_file",
            requested_by_agent="general_chat_agent",
            summary="Delete the generated file.",
            target_scope={"files": ["build/output.txt"]},
        )
        interpreter = StaticInterpreter(
            {
                "decision": "modify",
                "requested_outputs": [],
                "target_scope": {"modules": ["backend"]},
                "selected_index": None,
                "should_execute": True,
                "reason": "The user requested a narrowed backend-only deletion.",
                "confidence": 0.81,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "yes but only backend", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "modify")
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["runtime_action"], "ask_clarification")
        self.assertEqual(validation["next_status"], "ask_clarification")

    def test_interpreter_exception_becomes_unclear(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )

        contract = interpret_pending_action_reply(action, "go ahead", interpreter=RaisingInterpreter())
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "unclear")
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["runtime_action"], "ask_clarification")

    def test_malformed_interpreter_output_becomes_unclear(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = StaticInterpreter(
            {
                "decision": "launch",
                "requested_outputs": "diff",
                "target_scope": "backend",
                "selected_index": "two",
                "should_execute": True,
                "reason": "Malformed parse.",
                "confidence": "not-a-number",
                "interpretation_source": "",
            }
        )

        contract = interpret_pending_action_reply(action, "do it", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "unclear")
        self.assertEqual(contract["requested_outputs"], [])
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["runtime_action"], "ask_clarification")

    def test_llm_inferred_selection_without_exact_match_becomes_selection(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="select_knowledge_document",
            requested_by_agent="knowledge_agent",
            summary="Select a document to open.",
            metadata={
                "selection_options": [
                    {
                        "label": "Setup Guide",
                        "aliases": ["guide"],
                        "value": "Setup Guide",
                        "payload": {"document_name": "Setup Guide"},
                    },
                    {
                        "label": "Architecture Overview",
                        "aliases": ["architecture overview"],
                        "value": "Architecture Overview",
                        "payload": {"document_name": "Architecture Overview"},
                    },
                ]
            },
        )
        interpreter = StaticInterpreter(
            {
                "decision": "select",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": 1,
                "should_execute": True,
                "reason": "The user appears to mean the second document.",
                "confidence": 0.9,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "the second one sounds right", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "select")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "select")
        self.assertEqual(validation["selected_index"], 1)

    def test_inconsistent_approve_with_requested_outputs_becomes_revision(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
            target_scope={"files": ["src/main.py"]},
        )
        interpreter = StaticInterpreter(
            {
                "decision": "approve",
                "requested_outputs": ["diff"],
                "target_scope": {},
                "selected_index": None,
                "should_execute": False,
                "reason": "The user asked to see the diff first.",
                "confidence": 0.92,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "show me the diff first", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "approve")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["runtime_action"], "request_revision")

    def test_inconsistent_approve_without_execution_stays_non_executable(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = StaticInterpreter(
            {
                "decision": "approve",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": None,
                "should_execute": False,
                "reason": "The parser was unsure.",
                "confidence": 0.92,
                "interpretation_source": "llm_parser",
            }
        )

        contract = interpret_pending_action_reply(action, "approve", interpreter=interpreter)
        validation = validate_execution_contract(action, contract)

        self.assertEqual(contract["decision"], "approve")
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["runtime_action"], "ask_clarification")


class PendingActionParserTests(unittest.TestCase):
    def test_parser_low_confidence_result_becomes_unclear(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = LLMPendingActionInterpreter(
            StructuredOutputLLM(
                {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "Low confidence approval.",
                    "confidence": 0.4,
                }
            ),
            confidence_threshold=0.75,
        )

        parsed = interpreter.parse_pending_action_reply(
            action,
            prepare_pending_action_reply_input(action, "go ahead"),
        )

        self.assertEqual(parsed["decision"], "unclear")
        self.assertFalse(parsed["should_execute"])

    def test_parser_malformed_output_becomes_unclear(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = LLMPendingActionInterpreter(
            StructuredOutputLLM({"decision": "launch", "confidence": 0.99}),
            confidence_threshold=0.75,
        )

        parsed = interpreter.parse_pending_action_reply(
            action,
            prepare_pending_action_reply_input(action, "do it"),
        )

        self.assertEqual(parsed["decision"], "unclear")
        self.assertFalse(parsed["should_execute"])

    def test_parser_exception_becomes_unclear(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = LLMPendingActionInterpreter(
            StructuredOutputLLM(error=RuntimeError("network unavailable")),
            confidence_threshold=0.75,
        )

        parsed = interpreter.parse_pending_action_reply(
            action,
            prepare_pending_action_reply_input(action, "approve"),
        )

        self.assertEqual(parsed["decision"], "unclear")
        self.assertFalse(parsed["should_execute"])

    def test_parser_uses_backup_model_when_primary_is_unavailable(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish the staged conversion package.",
        )
        interpreter = LLMPendingActionInterpreter(
            SequencedLLM([RuntimeError("network unavailable"), RuntimeError("network unavailable")]),
            backup_llm=StructuredOutputLLM(
                {
                    "decision": "approve",
                    "requested_outputs": [],
                    "target_scope": {},
                    "selected_index": None,
                    "should_execute": True,
                    "reason": "The backup parser recovered the approval.",
                    "confidence": 0.91,
                }
            ),
            confidence_threshold=0.75,
        )

        parsed = interpreter.parse_pending_action_reply(
            action,
            prepare_pending_action_reply_input(action, "approve"),
        )

        self.assertEqual(parsed["decision"], "approve")
        self.assertTrue(parsed["should_execute"])

    def test_parser_repairs_low_confidence_output(self) -> None:
        action = build_pending_action(
            session_id="thread-1",
            action_type="apply_edit",
            requested_by_agent="general_chat_agent",
            summary="Apply the proposed edit.",
            target_scope={"files": ["src/main.py"]},
        )
        interpreter = LLMPendingActionInterpreter(
            SequencedLLM(
                [
                    {
                        "decision": "modify",
                        "requested_outputs": ["preview"],
                        "target_scope": {},
                        "selected_index": None,
                        "should_execute": False,
                        "reason": "Unsure whether the user wanted a preview or summary.",
                        "confidence": 0.41,
                    },
                    AIMessage(content='{"decision":"modify","requested_outputs":["summary"],"should_execute":false}'),
                    {
                        "decision": "modify",
                        "requested_outputs": ["summary"],
                        "target_scope": {},
                        "selected_index": None,
                        "should_execute": False,
                        "reason": "The user asked for a summary first.",
                        "confidence": 0.91,
                    },
                ]
            ),
            confidence_threshold=0.75,
            max_parse_attempts=1,
        )

        parsed = interpreter.parse_pending_action_reply(
            action,
            prepare_pending_action_reply_input(action, "show me a summary"),
        )

        self.assertEqual(parsed["decision"], "modify")
        self.assertEqual(parsed["requested_outputs"], ["summary"])
        self.assertFalse(parsed["should_execute"])


if __name__ == "__main__":
    unittest.main()
