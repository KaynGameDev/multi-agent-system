from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.rehydration import RUNTIME_REHYDRATION_METADATA_KEY
from app.config import WebAuthCredential
from interfaces.web.conversations import TRANSCRIPT_TYPE_COMPACT_BOUNDARY
from interfaces.web.server import WebServer
from tests.common import make_settings


class DummyGraph:
    def __init__(self) -> None:
        self.invoke_count = 0

    def invoke(self, initial_state, config=None):
        self.invoke_count += 1
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Dummy route.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content="Dummy reply.")],
        }


class RecordingGraph:
    def __init__(self, reply_text: str = "Recorded reply.") -> None:
        self.reply_text = reply_text
        self.last_state = None
        self.invoke_count = 0

    def invoke(self, initial_state, config=None):
        self.invoke_count += 1
        self.last_state = dict(initial_state)
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Recorded route.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content=self.reply_text)],
        }


class MissingCheckpointStore:
    def has_checkpoint(self, _thread_id: str) -> bool:
        return False

    def delete_thread(self, _thread_id: str) -> None:
        return None


class StickyCheckpointStore:
    def __init__(self) -> None:
        self._has_checkpoint = True
        self.deleted_threads: list[str] = []

    def has_checkpoint(self, _thread_id: str) -> bool:
        return self._has_checkpoint

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)
        self._has_checkpoint = False


class UsageBaselineGraph:
    def __init__(self) -> None:
        self.last_state = None
        self.invoke_count = 0

    def invoke(self, initial_state, config=None):
        self.invoke_count += 1
        self.last_state = dict(initial_state)
        return {
            **initial_state,
            "route": "knowledge_agent",
            "route_reason": "Tool-backed reply.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [
                *initial_state.get("messages", []),
                AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 240, "output_tokens": 30, "total_tokens": 270},
                    id="tool-call-ai",
                ),
                AIMessage(content="Rendered knowledge summary."),
            ],
        }


class SessionMemoryAutoCompactGraph:
    def __init__(self) -> None:
        self.invoke_count = 0
        self.last_state = None

    def invoke(self, initial_state, config=None):
        self.invoke_count += 1
        self.last_state = dict(initial_state)
        if self.invoke_count == 1:
            return {
                **initial_state,
                "route": "general_chat_agent",
                "route_reason": "Built a long assistant reply.",
                "skill_resolution_diagnostics": [],
                "agent_selection_diagnostics": [],
                "selection_warnings": [],
                "messages": [
                    AIMessage(
                        content="Long assistant reply that should seed session memory.",
                        usage_metadata={"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
                    )
                ],
            }
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Continued after a memory-backed compact.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content="Continued after memory compact.")],
        }


class RehydrationStateGraph:
    def __init__(self) -> None:
        self.last_state = None
        self.invoke_count = 0

    def invoke(self, initial_state, config=None):
        self.last_state = dict(initial_state)
        self.invoke_count += 1
        return {
            **initial_state,
            "route": "knowledge_agent",
            "route_reason": "Opened a knowledge document.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "context_paths": ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
            "pending_action": {
                "id": "pending-doc",
                "status": "awaiting_confirmation",
                "summary": "Review the opened knowledge document.",
                "metadata": {"source_tool_id": "knowledge.read_document"},
            },
            "tool_result": {
                "tool_name": "read_knowledge_document",
                "tool_id": "knowledge.read_document",
                "status": "ok",
                "payload": {
                    "ok": True,
                    "document": {
                        "name": "ArchitectureOverview",
                        "title": "Architecture Overview",
                        "path": "knowledge/Docs/00_Shared/ArchitectureOverview.md",
                    },
                    "content": "Architecture excerpt",
                },
            },
            "tool_execution_trace": [
                {
                    "result": {
                        "tool_name": "read_knowledge_document",
                        "tool_id": "knowledge.read_document",
                        "status": "ok",
                        "payload": {
                            "ok": True,
                            "document": {
                                "path": "knowledge/Docs/00_Shared/ArchitectureOverview.md",
                            },
                            "content": "Architecture excerpt",
                        },
                    }
                }
            ],
            "messages": [AIMessage(content="Opened the architecture overview.")],
        }


class PromptTooLongRecoveryGraph:
    def __init__(self, *, fail_count: int = 1, success_reply: str = "Recovered after compaction.") -> None:
        self.fail_count = fail_count
        self.success_reply = success_reply
        self.invoke_count = 0
        self.last_state = None

    def invoke(self, initial_state, config=None):
        self.invoke_count += 1
        self.last_state = dict(initial_state)
        if self.invoke_count <= self.fail_count:
            raise RuntimeError("maximum context length exceeded for this request")
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Recovered after prompt-too-long retry.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content=self.success_reply)],
        }


class MaxOutputRecoveryGraph:
    def __init__(self) -> None:
        self.invoke_count = 0
        self.last_state = None

    def invoke(self, initial_state, config=None):
        self.invoke_count += 1
        self.last_state = dict(initial_state)
        if self.invoke_count == 1:
            return {
                **initial_state,
                "route": "general_chat_agent",
                "route_reason": "First response was truncated.",
                "skill_resolution_diagnostics": [],
                "agent_selection_diagnostics": [],
                "selection_warnings": [],
                "messages": [
                    AIMessage(
                        content="Partial reply",
                        response_metadata={"finish_reason": "length"},
                    )
                ],
            }
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Recovered after output-length retry.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content="Recovered complete reply.")],
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

    def test_checkpoint_rebuild_uses_hidden_boundaries_but_api_response_hides_them(self) -> None:
        graph = RecordingGraph(reply_text="Recovered reply.")
        server = WebServer(
            agent_graph=graph,
            settings=self.settings,
            checkpoint_store=MissingCheckpointStore(),
        )
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Compaction transcript"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="hello")
        server.conversation_store.append_message(conversation_id, role="assistant", markdown="First reply.")
        server.conversation_store.append_compact_boundary(
            conversation_id,
            trigger="checkpoint_missing",
            pre_tokens=4096,
            preserved_tail={"tail_length": 2},
        )

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "continue", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        rebuilt_messages = graph.last_state["messages"]
        self.assertEqual(len(rebuilt_messages), 4)
        self.assertTrue(isinstance(rebuilt_messages[0], HumanMessage))
        self.assertTrue(isinstance(rebuilt_messages[1], AIMessage))
        self.assertTrue(isinstance(rebuilt_messages[2], SystemMessage))
        self.assertTrue(isinstance(rebuilt_messages[3], HumanMessage))
        self.assertEqual(rebuilt_messages[0].content, "hello")
        self.assertEqual(rebuilt_messages[1].content, "First reply.")
        self.assertEqual(rebuilt_messages[3].content, "continue")
        self.assertEqual(
            rebuilt_messages[2].additional_kwargs["transcript_type"],
            TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
        )
        self.assertEqual(
            rebuilt_messages[2].additional_kwargs["metadata"],
            {"trigger": "checkpoint_missing", "preTokens": 4096, "preservedTail": {"tail_length": 2}},
        )

        payload = response.json()
        self.assertTrue(all(message["role"] != "system" for message in payload["messages"]))
        self.assertTrue(all(message["type"] != TRANSCRIPT_TYPE_COMPACT_BOUNDARY for message in payload["messages"]))
        self.assertEqual(
            [(message["role"], message["markdown"]) for message in payload["messages"]],
            [
                ("user", "hello"),
                ("assistant", "First reply."),
                ("user", "continue"),
                ("assistant", "Recovered reply."),
            ],
        )

    def test_web_transcript_persists_usage_baseline_for_synthetic_final_reply(self) -> None:
        graph = UsageBaselineGraph()
        server = WebServer(agent_graph=graph, settings=self.settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Usage baseline"}).json()
        response = client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "tell me more", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        public_payload = response.json()
        self.assertNotIn("metadata", public_payload["assistant_message"])
        self.assertEqual(public_payload["assistant_message"]["markdown"], "Rendered knowledge summary.")
        self.assertEqual(
            public_payload["assistant_message"]["usage"],
            {"input_tokens": 240, "output_tokens": 30, "total_tokens": 270},
        )

        full_conversation = server.conversation_store.get_full_conversation(conversation["conversation_id"])
        assistant_message = full_conversation["messages"][-1]
        self.assertEqual(assistant_message["usage"], {"input_tokens": 240, "output_tokens": 30, "total_tokens": 270})
        self.assertEqual(
            assistant_message["metadata"]["usage_baseline_stage"],
            "before_message",
        )

    def test_web_transcript_persists_runtime_rehydration_state_in_assistant_metadata(self) -> None:
        graph = RehydrationStateGraph()
        server = WebServer(agent_graph=graph, settings=self.settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Rehydration state"}).json()
        response = client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "open the architecture overview", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        full_conversation = server.conversation_store.get_full_conversation(conversation["conversation_id"])
        assistant_message = full_conversation["messages"][-1]
        rehydration_state = assistant_message["metadata"][RUNTIME_REHYDRATION_METADATA_KEY]
        self.assertEqual(
            rehydration_state["context_paths"],
            ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
        )
        self.assertEqual(
            rehydration_state["recent_file_reads"],
            ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
        )
        self.assertEqual(rehydration_state["pending_action"]["id"], "pending-doc")
        self.assertEqual(
            rehydration_state["tool_result"]["payload"]["document"]["path"],
            "knowledge/Docs/00_Shared/ArchitectureOverview.md",
        )

    def test_prompt_too_long_error_triggers_reactive_compaction_and_retry(self) -> None:
        graph = PromptTooLongRecoveryGraph()
        server = WebServer(agent_graph=graph, settings=self.settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Reactive recovery"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="Earlier request")
        server.conversation_store.append_message(conversation_id, role="assistant", markdown="Earlier answer")

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Latest question", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(graph.invoke_count, 2)
        self.assertTrue(payload["limit_recovery"]["attempted"])
        self.assertTrue(payload["limit_recovery"]["recovered"])
        self.assertEqual(payload["limit_recovery"]["reason"], "prompt_too_long")
        self.assertEqual(payload["limit_recovery"]["retry_count"], 1)
        self.assertTrue(payload["context_compaction"]["applied"])
        self.assertEqual(payload["context_compaction"]["trigger"], "reactive_recovery")
        self.assertEqual(payload["context_compaction"]["reason"], "prompt_too_long")
        self.assertEqual(payload["assistant_message"]["markdown"], "Recovered after compaction.")
        self.assertTrue(any("## Continuation Summary" in message["markdown"] for message in payload["messages"]))

    def test_max_output_truncation_triggers_reactive_recovery_retry(self) -> None:
        graph = MaxOutputRecoveryGraph()
        server = WebServer(agent_graph=graph, settings=self.settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Output recovery"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="Earlier request")
        server.conversation_store.append_message(conversation_id, role="assistant", markdown="Earlier answer")

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Need the full answer", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(graph.invoke_count, 2)
        self.assertTrue(payload["limit_recovery"]["attempted"])
        self.assertTrue(payload["limit_recovery"]["recovered"])
        self.assertEqual(payload["limit_recovery"]["reason"], "max_output_tokens")
        self.assertEqual(payload["assistant_message"]["markdown"], "Recovered complete reply.")

    def test_reactive_recovery_blocks_after_single_retry_is_exhausted(self) -> None:
        graph = PromptTooLongRecoveryGraph(fail_count=2)
        server = WebServer(agent_graph=graph, settings=self.settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Retry exhausted"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="Earlier request")
        server.conversation_store.append_message(conversation_id, role="assistant", markdown="Earlier answer")

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Latest question", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 409)
        payload = response.json()
        self.assertTrue(payload["blocked"])
        self.assertTrue(payload["limit_recovery"]["attempted"])
        self.assertFalse(payload["limit_recovery"]["recovered"])
        self.assertEqual(payload["limit_recovery"]["retry_count"], 1)
        self.assertIn("retried once", payload["detail"])
        self.assertEqual(graph.invoke_count, 2)

    def test_auto_compact_uses_session_memory_when_recent_delta_fits_preserved_tail(self) -> None:
        settings = replace(
            self.settings,
            context_window_effective_window=1_000,
            context_window_warning_threshold=600,
            context_window_auto_compact_threshold=700,
            context_window_hard_block_threshold=950,
            context_window_auto_compact_enabled=True,
            context_window_auto_compact_preserved_tail_count=1,
            session_memory_enabled=True,
            session_memory_initialize_threshold_tokens=1,
            session_memory_update_growth_threshold_tokens=1,
        )
        graph = SessionMemoryAutoCompactGraph()
        server = WebServer(agent_graph=graph, settings=settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Session memory auto compact"}).json()
        conversation_id = conversation["conversation_id"]

        first_response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Start a long session", "display_name": "Tester", "email": "tester@example.com"},
        )
        self.assertEqual(first_response.status_code, 200)
        self.assertIsNotNone(server.session_memory_store.get(f"web:{conversation_id}"))

        second_response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "What should we do next?", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(second_response.status_code, 200)
        second_payload = second_response.json()
        self.assertTrue(second_payload["context_compaction"]["applied"])
        self.assertTrue(second_payload["context_compaction"]["used_session_memory"])
        rebuilt_messages = graph.last_state["messages"]
        self.assertEqual(len(rebuilt_messages), 3)
        self.assertTrue(isinstance(rebuilt_messages[0], SystemMessage))
        self.assertTrue(isinstance(rebuilt_messages[1], AIMessage))
        self.assertTrue(isinstance(rebuilt_messages[2], HumanMessage))
        self.assertIn("## Continuation Summary", rebuilt_messages[1].content)
        self.assertEqual(rebuilt_messages[2].content, "What should we do next?")

    def test_auto_compact_rewrites_transcript_before_model_invoke(self) -> None:
        settings = replace(
            self.settings,
            context_window_effective_window=1_000,
            context_window_warning_threshold=600,
            context_window_auto_compact_threshold=700,
            context_window_hard_block_threshold=950,
            context_window_auto_compact_enabled=True,
            context_window_auto_compact_preserved_tail_count=1,
        )
        checkpoint_store = StickyCheckpointStore()
        graph = RecordingGraph(reply_text="After auto compact.")
        server = WebServer(
            agent_graph=graph,
            settings=settings,
            checkpoint_store=checkpoint_store,
        )
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Auto compact"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="Old request")
        server.conversation_store.append_transcript_message(
            conversation_id,
            role="assistant",
            message_type="message",
            markdown="Old answer",
            usage={"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
        )

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Need the next step", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["blocked"])
        self.assertTrue(payload["context_compaction"]["applied"])
        self.assertEqual(checkpoint_store.deleted_threads, [f"web:{conversation_id}"])
        self.assertEqual(graph.invoke_count, 1)

        rebuilt_messages = graph.last_state["messages"]
        self.assertEqual(len(rebuilt_messages), 3)
        self.assertTrue(isinstance(rebuilt_messages[0], SystemMessage))
        self.assertTrue(isinstance(rebuilt_messages[1], AIMessage))
        self.assertTrue(isinstance(rebuilt_messages[2], HumanMessage))
        self.assertIn("## Continuation Summary", rebuilt_messages[1].content)
        self.assertEqual(rebuilt_messages[2].content, "Need the next step")
        self.assertEqual(
            [(message["role"], message["markdown"]) for message in payload["messages"]],
            [
                ("assistant", payload["messages"][0]["markdown"]),
                ("user", "Need the next step"),
                ("assistant", "After auto compact."),
            ],
        )
        self.assertIn("## Continuation Summary", payload["messages"][0]["markdown"])

    def test_blocks_before_hard_limit_when_auto_compact_is_disabled(self) -> None:
        settings = replace(
            self.settings,
            context_window_effective_window=1_000,
            context_window_warning_threshold=600,
            context_window_auto_compact_threshold=700,
            context_window_hard_block_threshold=950,
            context_window_auto_compact_enabled=False,
        )
        graph = RecordingGraph(reply_text="Should not run.")
        server = WebServer(agent_graph=graph, settings=settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Manual compact required"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="Earlier request")
        server.conversation_store.append_transcript_message(
            conversation_id,
            role="assistant",
            message_type="message",
            markdown="Earlier answer",
            usage={"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
        )

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "One more question", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 409)
        payload = response.json()
        self.assertTrue(payload["blocked"])
        self.assertEqual(payload["context_window"]["decision"]["level"], "error")
        self.assertIn("Automatic compaction is disabled", payload["detail"])
        self.assertEqual(graph.invoke_count, 0)
        self.assertEqual(payload["messages"][-1]["markdown"], "One more question")
        self.assertEqual(payload["messages"][-1]["role"], "user")

    def test_repeated_auto_compaction_failures_open_circuit_breaker(self) -> None:
        settings = replace(
            self.settings,
            context_window_effective_window=1_000,
            context_window_warning_threshold=600,
            context_window_auto_compact_threshold=700,
            context_window_hard_block_threshold=980,
            context_window_auto_compact_enabled=True,
            context_window_auto_compact_failure_limit=1,
        )
        graph = RecordingGraph(reply_text="Should not run.")
        server = WebServer(agent_graph=graph, settings=settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Breaker"}).json()
        conversation_id = conversation["conversation_id"]
        server.conversation_store.append_message(conversation_id, role="user", markdown="Earlier request")
        server.conversation_store.append_transcript_message(
            conversation_id,
            role="assistant",
            message_type="message",
            markdown="Earlier answer",
            usage={"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
        )

        with patch("app.compaction.compact_conversation", side_effect=RuntimeError("boom")) as compact_mock:
            first_response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"message": "First blocked try", "display_name": "Tester", "email": "tester@example.com"},
            )
            second_response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                json={"message": "Second blocked try", "display_name": "Tester", "email": "tester@example.com"},
            )

        self.assertEqual(first_response.status_code, 409)
        self.assertEqual(second_response.status_code, 409)
        self.assertEqual(compact_mock.call_count, 1)
        second_payload = second_response.json()
        self.assertTrue(second_payload["blocked"])
        self.assertTrue(second_payload["context_window"]["decision"]["auto_compact_breaker_open"])
        self.assertIn("paused after repeated failures", second_payload["detail"])
        self.assertEqual(graph.invoke_count, 0)


if __name__ == "__main__":
    unittest.main()
