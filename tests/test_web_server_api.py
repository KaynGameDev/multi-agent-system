from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.config import WebAuthCredential
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
        self.server = WebServer(agent_graph=DummyGraph(), settings=self.settings)
        self.client = TestClient(self.server.app)

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

    def test_can_delete_conversation_and_linked_conversion_session(self) -> None:
        conversation = self.client.post("/api/conversations", json={"title": "Disposable"}).json()
        self.server.conversion_store.create_session(
            thread_id=f"web:{conversation['conversation_id']}",
            channel_id=conversation["conversation_id"],
            user_id="tester@example.com",
        )

        response = self.client.delete(f"/api/conversations/{conversation['conversation_id']}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"deleted": True, "conversation_id": conversation["conversation_id"]},
        )
        self.assertIsNone(
            self.server.conversion_store.get_active_session_by_thread(
                f"web:{conversation['conversation_id']}"
            )
        )

        missing_response = self.client.get(f"/api/conversations/{conversation['conversation_id']}")
        self.assertEqual(missing_response.status_code, 404)

    def test_delete_returns_not_found_for_missing_conversation(self) -> None:
        response = self.client.delete("/api/conversations/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Conversation not found.")

    def test_frontend_responses_disable_caching(self) -> None:
        index_response = self.client.get("/")
        self.assertEqual(index_response.status_code, 200)
        self.assertEqual(
            index_response.headers.get("cache-control"),
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.assertIn("/static/app.css?v=", index_response.text)
        self.assertIn("/static/app.js?v=", index_response.text)

        static_response = self.client.get("/static/app.js")
        self.assertEqual(static_response.status_code, 200)
        self.assertEqual(
            static_response.headers.get("cache-control"),
            "no-store, no-cache, must-revalidate, max-age=0",
        )

    def test_web_auth_redirects_browser_requests_to_login(self) -> None:
        settings = replace(
            self.settings,
            web_auth_enabled=True,
            web_auth_credentials=(WebAuthCredential(username="jade", password="secret-pass"),),
            web_auth_session_secret="session-secret",
            web_auth_cookie_secure=False,
        )
        client = TestClient(WebServer(agent_graph=DummyGraph(), settings=settings).app)

        response = client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers.get("location"), "/login")

    def test_web_auth_blocks_api_until_login(self) -> None:
        settings = replace(
            self.settings,
            web_auth_enabled=True,
            web_auth_credentials=(WebAuthCredential(username="jade", password="secret-pass"),),
            web_auth_session_secret="session-secret",
            web_auth_cookie_secure=False,
        )
        client = TestClient(WebServer(agent_graph=DummyGraph(), settings=settings).app)

        response = client.get("/api/conversations")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Authentication required.")

    def test_web_auth_login_session_and_logout_flow(self) -> None:
        settings = replace(
            self.settings,
            web_auth_enabled=True,
            web_auth_credentials=(WebAuthCredential(username="jade", password="secret-pass"),),
            web_auth_session_secret="session-secret",
            web_auth_cookie_secure=False,
        )
        client = TestClient(WebServer(agent_graph=DummyGraph(), settings=settings).app)

        session_before = client.get("/api/auth/session")
        self.assertEqual(
            session_before.json(),
            {"enabled": True, "authenticated": False, "username": ""},
        )

        login_response = client.post(
            "/api/auth/login",
            json={"username": "jade", "password": "secret-pass"},
        )
        self.assertEqual(login_response.status_code, 200)
        self.assertEqual(
            login_response.json(),
            {"enabled": True, "authenticated": True, "username": "jade"},
        )

        session_after = client.get("/api/auth/session")
        self.assertEqual(
            session_after.json(),
            {"enabled": True, "authenticated": True, "username": "jade"},
        )

        index_response = client.get("/")
        self.assertEqual(index_response.status_code, 200)
        self.assertIn("Signed in", index_response.text)

        conversations_response = client.get("/api/conversations")
        self.assertEqual(conversations_response.status_code, 200)

        logout_response = client.post("/api/auth/logout")
        self.assertEqual(logout_response.status_code, 200)
        self.assertEqual(
            logout_response.json(),
            {"enabled": True, "authenticated": False, "username": ""},
        )

        after_logout = client.get("/api/conversations")
        self.assertEqual(after_logout.status_code, 401)

    def test_web_auth_rejects_invalid_credentials(self) -> None:
        settings = replace(
            self.settings,
            web_auth_enabled=True,
            web_auth_credentials=(WebAuthCredential(username="jade", password="secret-pass"),),
            web_auth_session_secret="session-secret",
            web_auth_cookie_secure=False,
        )
        client = TestClient(WebServer(agent_graph=DummyGraph(), settings=settings).app)

        response = client.post(
            "/api/auth/login",
            json={"username": "jade", "password": "wrong-pass"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password.")


if __name__ == "__main__":
    unittest.main()
