from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.document_conversion.agent import DocumentConversionAgentNode, build_conversion_pending_action
from tests.common import make_settings


class ConversionLLM:
    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return AIMessage(content="rendered")


class StaticInterpreter:
    def __init__(self, parsed_reply: dict) -> None:
        self.parsed_reply = dict(parsed_reply)

    def parse_pending_action_reply(self, _action, _prepared_input):
        return dict(self.parsed_reply)


class DocumentConversionConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.settings = replace(
            make_settings(self.root / "runtime"),
            knowledge_base_dir=str(self.root / "knowledge"),
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _build_node_and_session(self) -> tuple[DocumentConversionAgentNode, object]:
        node = DocumentConversionAgentNode(
            ConversionLLM(),
            settings=self.settings,
            agent_name="document_conversion_agent",
        )
        staged_package_path = self.root / "runtime" / "staging" / "package"
        staged_package_path.mkdir(parents=True, exist_ok=True)
        (staged_package_path / "README.md").write_text("# Weekly Activity\n", encoding="utf-8")
        session = node.store.create_session(thread_id="thread-1", channel_id="channel-1", user_id="user-1")
        session = node.store.update_session(
            session.session_id,
            status="ready_for_approval",
            game_slug="buyu-da-luan-dou",
            market_slug="indonesia-main",
            feature_slug="weekly-activity",
            staged_package_path=str(staged_package_path),
            draft_payload={"feature": "weekly-activity"},
            missing_required_fields=[],
            approval_state="pending",
        )
        return node, session

    def _build_pending_action(self, session) -> dict[str, object]:
        pending_action = build_conversion_pending_action(
            state={"thread_id": "thread-1"},
            session=session,
            source_count=1,
        )
        assert pending_action is not None
        return pending_action

    def test_document_conversion_pending_action_publishes_on_approval(self) -> None:
        node, session = self._build_node_and_session()
        node.pending_action_interpreter = StaticInterpreter(
            {
                "decision": "approve",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": None,
                "should_execute": True,
                "reason": "The user approved publishing.",
                "confidence": 0.97,
                "interpretation_source": "llm_parser",
            }
        )
        pending_action = self._build_pending_action(session)

        def fake_publish(store, current_session, knowledge_root):
            _ = knowledge_root
            store.update_session(
                current_session.session_id,
                status="published",
                approval_state="approved",
            )
            return "Docs/20_Deployments/IndonesiaMain/BuYuDaLuanDou/Features/weekly-activity"

        with patch("agents.document_conversion.agent.render_conversion_response", side_effect=lambda *args, **kwargs: f"{kwargs['response_kind']}:{kwargs['preferred_language']}"):
            with patch("agents.document_conversion.agent.publish_conversion_package", side_effect=fake_publish):
                result = node._build_pending_action_response(
                    {
                        "messages": [HumanMessage(content="approve")],
                    },
                    session=session,
                    pending_action=pending_action,
                    preferred_language="en",
                )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["content"], "published:en")
        self.assertEqual(result["session"].status, "published")
        self.assertEqual(result["session"].approval_state, "approved")
        self.assertIsNone(result["pending_action"])
        self.assertEqual(result["execution_contract"]["decision"], "approve")
        self.assertEqual(result["tool_invocation"]["tool_name"], "conversion_publish_package")
        self.assertEqual(result["tool_result"]["tool_name"], "conversion_publish_package")
        self.assertEqual(result["tool_result"]["tool_id"], "conversion.publish_package")
        self.assertEqual(result["tool_result"]["execution_backend"], "internal_workflow")
        self.assertEqual(result["tool_execution_trace"][0]["result"]["tool_name"], "conversion_publish_package")

    def test_document_conversion_pending_action_cancels(self) -> None:
        node, session = self._build_node_and_session()
        node.pending_action_interpreter = StaticInterpreter(
            {
                "decision": "reject",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": None,
                "should_execute": False,
                "reason": "The user rejected publishing.",
                "confidence": 0.97,
                "interpretation_source": "llm_parser",
            }
        )
        pending_action = self._build_pending_action(session)

        with patch("agents.document_conversion.agent.render_conversion_response", side_effect=lambda *args, **kwargs: f"{kwargs['response_kind']}:{kwargs['preferred_language']}"):
            result = node._build_pending_action_response(
                {
                    "messages": [HumanMessage(content="cancel")],
                },
                session=session,
                pending_action=pending_action,
                preferred_language="en",
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["content"], "cancelled:en")
        self.assertEqual(result["session"].status, "cancelled")
        self.assertEqual(result["session"].approval_state, "cancelled")
        self.assertIsNone(result["pending_action"])
        self.assertIsNone(result["execution_contract"])

    def test_document_conversion_pending_action_requests_clarification_for_ambiguous_reply(self) -> None:
        node, session = self._build_node_and_session()
        pending_action = self._build_pending_action(session)

        result = node._build_pending_action_response(
            {
                "messages": [HumanMessage(content="maybe later")],
            },
            session=session,
            pending_action=pending_action,
            preferred_language="en",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("can't determine", result["content"])
        self.assertEqual(result["pending_action"]["status"], "ask_clarification")
        self.assertIsNone(result["execution_contract"])

    def test_document_conversion_pending_action_requires_reapproval_when_staged_package_changes(self) -> None:
        node, session = self._build_node_and_session()
        node.pending_action_interpreter = StaticInterpreter(
            {
                "decision": "approve",
                "requested_outputs": [],
                "target_scope": {},
                "selected_index": None,
                "should_execute": True,
                "reason": "The user approved publishing.",
                "confidence": 0.97,
                "interpretation_source": "llm_parser",
            }
        )
        pending_action = self._build_pending_action(session)
        staged_package = Path(session.staged_package_path)
        (staged_package / "README.md").write_text("# Updated Weekly Activity\n", encoding="utf-8")

        with patch("agents.document_conversion.agent.render_conversion_response", side_effect=lambda *args, **kwargs: f"{kwargs['response_kind']}:{kwargs['preferred_language']}"):
            with patch("agents.document_conversion.agent.publish_conversion_package") as publish_mock:
                result = node._build_pending_action_response(
                    {
                        "messages": [HumanMessage(content="approve")],
                    },
                    session=session,
                    pending_action=pending_action,
                    preferred_language="en",
                )

        self.assertIsNotNone(result)
        assert result is not None
        publish_mock.assert_not_called()
        self.assertIn("changed since approval was requested", result["content"])
        self.assertEqual(result["pending_action"]["status"], "ask_clarification")
        self.assertIsNone(result["execution_contract"])


if __name__ == "__main__":
    unittest.main()
