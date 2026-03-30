from __future__ import annotations

import unittest

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


def build_settings() -> Settings:
    return Settings(
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
    )


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = DummyGraph()
        self.server = WebServer(agent_graph=self.graph, settings=build_settings())
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

    def test_capability_guard_keeps_conversion_requests_on_slack(self) -> None:
        created = self.client.post("/api/conversations", json={"title": "New chat"}).json()

        response = self.client.post(
            f"/api/conversations/{created['conversation_id']}/messages",
            json={"message": "Please convert https://docs.google.com/document/d/abc123/edit"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["assistant_message"]["markdown"], "Document conversion and uploads still live in Slack for now. This web chat supports general chat, knowledge lookups, and project task questions.")
        self.assertEqual(self.graph.invocations, [])


if __name__ == "__main__":
    unittest.main()
