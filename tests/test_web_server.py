from __future__ import annotations

import unittest
import tempfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from core.config import Settings
from interfaces.web_server import WebServer


class DummyGraph:
    def __init__(self) -> None:
        self.invocations: list[dict] = []

    def invoke(self, initial_state, config=None):
        self.invocations.append({"initial_state": initial_state, "config": config})
        message_text = initial_state["messages"][0].content
        if "tasks" in message_text.lower():
            route = "project_task_agent"
            reason = "Task query"
        else:
            route = "general_chat_agent"
            reason = "General chat"
        return {
            "route": route,
            "route_reason": reason,
            "messages": [AIMessage(content=f"Answer for: {message_text}")],
        }


class DummyConversionStore:
    def __init__(self, active_session=None) -> None:
        self.active_session = active_session
        self.thread_ids: list[str] = []

    def get_active_session_by_thread(self, thread_id: str):
        self.thread_ids.append(thread_id)
        return self.active_session


def build_settings(*, conversion_dir: str) -> Settings:
    return Settings(
        slack_enabled=False,
        slack_bot_token="",
        slack_app_token="",
        web_enabled=True,
        web_host="127.0.0.1",
        web_port=8000,
        google_api_key="test-key",
        gemini_model="gemini-3-flash-preview",
        gemini_temperature=0.2,
        google_application_credentials="/tmp/fake-creds.json",
        jade_project_sheet_id="sheet-id",
        project_sheet_range="Tasks!A1:Z",
        project_sheet_cache_ttl_seconds=30,
        slack_thinking_reaction="eyes",
        project_lookup_keywords=("task",),
        knowledge_base_dir="/tmp/knowledge",
        knowledge_file_types=(".md", ".xlsx"),
        conversion_work_dir=conversion_dir,
    )


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.graph = DummyGraph()
        self.conversion_store = DummyConversionStore()
        self.server = WebServer(
            agent_graph=self.graph,
            settings=build_settings(conversion_dir=self.temp_dir.name),
            conversion_store=self.conversion_store,
        )
        self.client = TestClient(self.server.app)

    def test_create_and_fetch_conversation(self) -> None:
        created = self.client.post("/api/conversations", json={"title": "New chat"}).json()

        fetched = self.client.get(f"/api/conversations/{created['conversation_id']}")

        self.assertEqual(fetched.status_code, 200)
        payload = fetched.json()
        self.assertEqual(payload["title"], "New chat")
        self.assertEqual(payload["messages"], [])

    def test_send_message_sets_web_interface_and_thread_id(self) -> None:
        created = self.client.post("/api/conversations", json={"title": "New chat"}).json()

        response = self.client.post(
            f"/api/conversations/{created['conversation_id']}/messages",
            json={
                "message": "What are my tasks?",
                "display_name": "Kayn",
                "email": "kayn@songkegame.com",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["assistant_message"]["markdown"], "Answer for: What are my tasks?")
        self.assertEqual(self.graph.invocations[0]["initial_state"]["interface_name"], "web")
        self.assertEqual(
            self.graph.invocations[0]["config"],
            {"configurable": {"thread_id": f"web:{created['conversation_id']}"}},
        )
        self.assertEqual(self.graph.invocations[0]["initial_state"]["user_sheet_name"], "刘煜")

    def test_send_message_returns_not_found_for_unknown_conversation(self) -> None:
        response = self.client.post(
            "/api/conversations/missing/messages",
            json={"message": "hello"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Conversation not found.")

    def test_google_doc_conversion_request_routes_to_document_conversion_agent(self) -> None:
        created = self.client.post("/api/conversations", json={"title": "New chat"}).json()

        response = self.client.post(
            f"/api/conversations/{created['conversation_id']}/messages",
            json={"message": "Please convert https://docs.google.com/document/d/abc123/edit"},
        )

        self.assertEqual(response.status_code, 200)
        invocation = self.graph.invocations[0]
        self.assertEqual(invocation["initial_state"]["route"], "document_conversion_agent")
        self.assertEqual(invocation["initial_state"]["route_reason"], "Web document conversion session.")

    def test_capability_guard_blocks_raw_upload_requests_without_google_links(self) -> None:
        created = self.client.post("/api/conversations", json={"title": "New chat"}).json()

        response = self.client.post(
            f"/api/conversations/{created['conversation_id']}/messages",
            json={"message": "Please upload and convert this file into the knowledge base."},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["assistant_message"]["markdown"],
            "Google Docs and Sheets links can be converted here. Raw file uploads still live in Slack for now.",
        )
        self.assertEqual(self.graph.invocations, [])

    def test_active_conversion_session_routes_approval_to_document_conversion_agent(self) -> None:
        self.graph = DummyGraph()
        self.server = WebServer(
            agent_graph=self.graph,
            settings=build_settings(conversion_dir=self.temp_dir.name),
            conversion_store=DummyConversionStore(active_session=SimpleNamespace(session_id="session-123")),
        )
        self.client = TestClient(self.server.app)
        created = self.client.post("/api/conversations", json={"title": "New chat"}).json()

        response = self.client.post(
            f"/api/conversations/{created['conversation_id']}/messages",
            json={"message": "approve"},
        )

        self.assertEqual(response.status_code, 200)
        invocation = self.graph.invocations[0]
        self.assertEqual(invocation["initial_state"]["route"], "document_conversion_agent")
        self.assertEqual(invocation["initial_state"]["conversion_session_id"], "session-123")


if __name__ == "__main__":
    unittest.main()
