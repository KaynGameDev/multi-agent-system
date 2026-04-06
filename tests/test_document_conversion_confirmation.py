from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.document_conversion.agent import DocumentConversionAgentNode
from app.pending_actions import build_pending_action
from tests.common import make_settings


class ConversionLLM:
    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return AIMessage(content="rendered")


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
        node = DocumentConversionAgentNode(ConversionLLM(), settings=self.settings, agent_name="document_conversion_agent")
        session = node.store.create_session(thread_id="thread-1", channel_id="channel-1", user_id="user-1")
        session = node.store.update_session(
            session.session_id,
            status="ready_for_approval",
            game_slug="buyu-da-luan-dou",
            market_slug="indonesia-main",
            feature_slug="weekly-activity",
            staged_package_path=str(self.root / "runtime" / "staging" / "package"),
            draft_payload={"feature": "weekly-activity"},
            missing_required_fields=[],
            approval_state="pending",
        )
        return node, session

    def _build_pending_action(self, session) -> dict[str, object]:
        return build_pending_action(
            session_id=session.session_id,
            action_type="publish_conversion_package",
            requested_by_agent="document_conversion_agent",
            summary="Publish staged conversion package for `Docs/20_Deployments/IndonesiaMain/BuYuDaLuanDou/Features/weekly-activity`.",
            metadata={
                "conversion_session_id": session.session_id,
                "staged_package_path": session.staged_package_path,
                "relative_package_path": "Docs/20_Deployments/IndonesiaMain/BuYuDaLuanDou/Features/weekly-activity",
            },
        )

    def test_document_conversion_pending_action_publishes_on_approval(self) -> None:
        node, session = self._build_node_and_session()
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

    def test_document_conversion_pending_action_cancels(self) -> None:
        node, session = self._build_node_and_session()
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


if __name__ == "__main__":
    unittest.main()
