from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from interfaces.web.server import WebServer
from tests.common import make_settings


class DummyGraph:
    def invoke(self, initial_state, config=None):
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Dummy route.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content="Dummy reply.")],
        }


class WebServerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.settings = make_settings(self.root / "runtime")
        self.client = TestClient(WebServer(agent_graph=DummyGraph(), settings=self.settings).app)

    def test_can_rename_conversation(self) -> None:
        conversation = self.client.post("/api/conversations", json={"title": "Original"}).json()

        response = self.client.patch(
            f"/api/conversations/{conversation['conversation_id']}",
            json={"title": "Renamed chat"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["title"], "Renamed chat")

        fetched = self.client.get(f"/api/conversations/{conversation['conversation_id']}").json()
        self.assertEqual(fetched["title"], "Renamed chat")

    def test_rename_returns_not_found_for_missing_conversation(self) -> None:
        response = self.client.patch(
            "/api/conversations/missing",
            json={"title": "Renamed chat"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Conversation not found.")


if __name__ == "__main__":
    unittest.main()
