from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.project_task.agent import ProjectTaskAgentNode
from app.checkpoints import build_checkpoint_store
from app.graph import build_graph
from app.main import bootstrap_system
from app.routing.agent_router import AgentRouter
from app.routing.pending_action_router import PendingActionRouter
from interfaces.web.server import WebServer
from tests.common import build_registration, make_settings


class CountingAgentNode:
    def __call__(self, state):
        human_messages = [message for message in state.get("messages", []) if isinstance(message, HumanMessage)]
        latest_human = human_messages[-1].content if human_messages else ""
        return {
            "messages": [
                AIMessage(content=f"human_count={len(human_messages)}; latest={latest_human}")
            ]
        }


class RecordingGraph:
    def __init__(self, reply_text: str = "Recorded reply.") -> None:
        self.reply_text = reply_text
        self.last_state = None

    def invoke(self, initial_state, config=None):
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


class BrokenCheckpointStore:
    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    def has_checkpoint(self, _thread_id: str) -> bool:
        raise RuntimeError("checkpoint corrupted")

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


@tool
def fake_project_tasks() -> dict:
    """Return a deterministic task payload for durable resume tests."""

    return {
        "tasks": [
            {
                "content": "Ship durable memory",
                "project": "Jade",
                "iteration": "Sprint 12",
                "platform": "Web",
                "priority": "P1",
                "assignee": "Tester",
                "end_date": "2026-04-04",
                "due_status": "today",
                "client_owner": "Alice",
                "server_owner": "Bob",
                "test_owner": "Carol",
                "product_owner": "Dana",
            }
        ],
        "match_count": 1,
        "filters": {
            "due_scope": "today",
            "assignee": "Tester",
        },
    }


class FakeTaskLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Here is the short task summary.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fake_project_tasks",
                    "args": {},
                    "id": "call_fake_project_tasks",
                }
            ],
        )


class DummyListener:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class DurableResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.settings = make_settings(self.root / "runtime")
        self._resources_to_close: list[object] = []

    def tearDown(self) -> None:
        for resource in reversed(self._resources_to_close):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self.tempdir.cleanup()

    def _track_resource(self, resource):
        self._resources_to_close.append(resource)
        return resource

    def _build_counting_graph(self, checkpointer):
        registrations = (
            build_registration(
                "general_chat_agent",
                namespace="general_chat",
                is_general_assistant=True,
                selection_order=10,
                build_node=lambda _llm=None, skill_registry=None: CountingAgentNode(),
            ),
        )
        return build_graph(
            None,
            checkpointer=checkpointer,
            agent_registrations=registrations,
            default_route="general_chat_agent",
        )

    def _create_client(self, graph, *, checkpoint_store=None):
        server = WebServer(
            agent_graph=graph,
            settings=self.settings,
            checkpoint_store=checkpoint_store,
        )
        return TestClient(server.app)

    def test_bootstrap_uses_shared_sqlite_checkpointer(self) -> None:
        runtime_dir = self.root / "bootstrap-runtime"
        env = {
            "WEB_ENABLED": "true",
            "SLACK_ENABLED": "false",
            "WEB_HOST": "127.0.0.1",
            "WEB_PORT": "8000",
            "GOOGLE_API_KEY": "test-key",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/credentials.json",
            "JADE_PROJECT_SHEET_ID": "sheet-id",
            "CONVERSION_WORK_DIR": str(runtime_dir),
            "ASSISTANT_REQUEST_PARSER_CONFIDENCE_THRESHOLD": "0.60",
            "PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD": "0.75",
        }
        captured_checkpointers = []
        captured_agent_routers = []
        captured_pending_action_routers = []
        primary_llm = object()
        parser_llm = object()

        def fake_build_agent_graph(_llm, **kwargs):
            captured_checkpointers.append(kwargs["checkpointer"])
            captured_agent_routers.append(kwargs["agent_router"])
            captured_pending_action_routers.append(kwargs["pending_action_router"])
            return object()

        with patch.dict(os.environ, env, clear=False):
            with patch("app.main.build_runtime_llms", return_value=(primary_llm, parser_llm)):
                with patch("app.main.build_agent_graph", side_effect=fake_build_agent_graph):
                    with patch("app.main.build_web_agent_registrations", return_value=()):
                        with patch("app.main.WebServer", side_effect=lambda *args, **kwargs: DummyListener()):
                            listeners = bootstrap_system()

        self.addCleanup(lambda: [getattr(listener, "stop", lambda: None)() for listener in reversed(listeners)])
        self.assertEqual(len(captured_checkpointers), 2)
        self.assertTrue(all(isinstance(checkpointer, SqliteSaver) for checkpointer in captured_checkpointers))
        self.assertIs(captured_checkpointers[0], captured_checkpointers[1])
        self.assertEqual(len(captured_agent_routers), 2)
        self.assertTrue(all(isinstance(router, AgentRouter) for router in captured_agent_routers))
        self.assertIs(captured_agent_routers[0], captured_agent_routers[1])
        self.assertIs(captured_agent_routers[0].parser.llm, parser_llm)
        self.assertEqual(len(captured_pending_action_routers), 2)
        self.assertTrue(all(isinstance(router, PendingActionRouter) for router in captured_pending_action_routers))
        self.assertIs(captured_pending_action_routers[0], captured_pending_action_routers[1])
        self.assertIs(captured_pending_action_routers[0].parser.llm, parser_llm)
        self.assertEqual(captured_agent_routers[0].parser.config.confidence_threshold, 0.60)
        self.assertEqual(captured_pending_action_routers[0].parser.config.confidence_threshold, 0.75)

    def test_web_resume_persists_across_restart(self) -> None:
        checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        graph = self._build_counting_graph(checkpoint_store.saver)
        client = self._create_client(graph, checkpoint_store=checkpoint_store)

        conversation = client.post("/api/conversations", json={"title": "Resume"}).json()
        first_response = client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "first turn", "display_name": "Tester", "email": "tester@example.com"},
        )
        self.assertEqual(first_response.status_code, 200)
        self.assertIn("human_count=1", first_response.json()["assistant_message"]["markdown"])

        checkpoint_store.close()
        self._resources_to_close.remove(checkpoint_store)

        restarted_checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        restarted_graph = self._build_counting_graph(restarted_checkpoint_store.saver)
        restarted_client = self._create_client(restarted_graph, checkpoint_store=restarted_checkpoint_store)

        second_response = restarted_client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "second turn", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(second_response.status_code, 200)
        self.assertIn("human_count=2", second_response.json()["assistant_message"]["markdown"])
        self.assertIn("latest=second turn", second_response.json()["assistant_message"]["markdown"])

    def test_web_rehydrates_transcript_when_checkpoint_missing(self) -> None:
        initial_graph = RecordingGraph(reply_text="First reply.")
        initial_client = self._create_client(initial_graph)
        conversation = initial_client.post("/api/conversations", json={"title": "Transcript"}).json()
        first_response = initial_client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "hello", "display_name": "Tester", "email": "tester@example.com"},
        )
        self.assertEqual(first_response.status_code, 200)

        resumed_graph = RecordingGraph(reply_text="Second reply.")
        resumed_client = self._create_client(
            resumed_graph,
            checkpoint_store=MissingCheckpointStore(),
        )
        second_response = resumed_client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "continue", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(second_response.status_code, 200)
        resumed_messages = resumed_graph.last_state["messages"]
        self.assertEqual(len(resumed_messages), 3)
        self.assertTrue(isinstance(resumed_messages[0], HumanMessage))
        self.assertTrue(isinstance(resumed_messages[1], AIMessage))
        self.assertTrue(isinstance(resumed_messages[2], HumanMessage))
        self.assertEqual(resumed_messages[0].content, "hello")
        self.assertEqual(resumed_messages[1].content, "First reply.")
        self.assertEqual(resumed_messages[2].content, "continue")

    def test_web_clears_broken_checkpoint_and_falls_back_to_transcript(self) -> None:
        initial_graph = RecordingGraph(reply_text="First reply.")
        initial_client = self._create_client(initial_graph)
        conversation = initial_client.post("/api/conversations", json={"title": "Broken checkpoint"}).json()
        first_response = initial_client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "hello", "display_name": "Tester", "email": "tester@example.com"},
        )
        self.assertEqual(first_response.status_code, 200)

        broken_checkpoint_store = BrokenCheckpointStore()
        resumed_graph = RecordingGraph(reply_text="Recovered reply.")
        resumed_client = self._create_client(
            resumed_graph,
            checkpoint_store=broken_checkpoint_store,
        )
        second_response = resumed_client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "continue", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(
            broken_checkpoint_store.deleted_threads,
            [f"web:{conversation['conversation_id']}"],
        )
        self.assertEqual(len(resumed_graph.last_state["messages"]), 3)

    def test_web_recovers_when_checkpoint_write_reports_sqlite_corruption(self) -> None:
        checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        graph = self._build_counting_graph(checkpoint_store.saver)
        client = self._create_client(graph, checkpoint_store=checkpoint_store)
        conversation = client.post("/api/conversations", json={"title": "Corrupted checkpoint write"}).json()

        original_put = SqliteSaver.put
        put_call_count = {"count": 0}

        def flaky_put(self, config, checkpoint, metadata, new_versions):
            if put_call_count["count"] == 0:
                put_call_count["count"] += 1
                raise sqlite3.DatabaseError("database disk image is malformed")
            put_call_count["count"] += 1
            return original_put(self, config, checkpoint, metadata, new_versions)

        with patch.object(SqliteSaver, "put", new=flaky_put):
            response = client.post(
                f"/api/conversations/{conversation['conversation_id']}/messages",
                json={"message": "hello", "display_name": "Tester", "email": "tester@example.com"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("human_count=1", response.json()["assistant_message"]["markdown"])
        self.assertGreaterEqual(put_call_count["count"], 2)
        quarantined = list(
            checkpoint_store.database_path.parent.glob(
                f"{checkpoint_store.database_path.name}.corrupt.*"
            )
        )
        self.assertTrue(quarantined)

    def test_persistent_checkpoint_preserves_tool_state_across_restart(self) -> None:
        checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        registrations = (
            build_registration(
                "project_task_agent",
                namespace="project_task",
                is_general_assistant=True,
                selection_order=10,
                build_node=lambda _llm=None, skill_registry=None: ProjectTaskAgentNode(
                    FakeTaskLLM(),
                    [fake_project_tasks],
                    skill_registry=skill_registry,
                    agent_name="project_task_agent",
                ),
                tools=(fake_project_tasks,),
            ),
        )
        graph = build_graph(
            None,
            checkpointer=checkpoint_store.saver,
            agent_registrations=registrations,
            default_route="project_task_agent",
        )
        thread_config = {"configurable": {"thread_id": "web:tool-state"}}

        first_state = graph.invoke(
            {
                "messages": [HumanMessage(content="show my tasks due today")],
                "requested_agent": "project_task_agent",
                "requested_skill_ids": [],
                "context_paths": [],
                "interface_name": "web",
            },
            config=thread_config,
        )
        self.assertIn("Tasks due today for Tester", first_state["messages"][-1].content)
        self.assertIn("Ship durable memory", first_state["messages"][-1].content)

        checkpoint_store.close()
        self._resources_to_close.remove(checkpoint_store)

        restarted_checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        restarted_graph = build_graph(
            None,
            checkpointer=restarted_checkpoint_store.saver,
            agent_registrations=registrations,
            default_route="project_task_agent",
        )

        second_state = restarted_graph.invoke(
            {
                "messages": [HumanMessage(content="details")],
                "requested_agent": "project_task_agent",
                "requested_skill_ids": [],
                "context_paths": [],
                "interface_name": "web",
            },
            config=thread_config,
        )

        self.assertIn("short task summary", second_state["messages"][-1].content.lower())

    def test_slack_style_thread_id_resumes_after_restart(self) -> None:
        checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        graph = self._build_counting_graph(checkpoint_store.saver)
        thread_config = {"configurable": {"thread_id": "C123456:1712345678.000100"}}

        first_state = graph.invoke(
            {
                "messages": [HumanMessage(content="first slack turn")],
                "requested_agent": "",
                "requested_skill_ids": [],
                "context_paths": [],
            },
            config=thread_config,
        )
        self.assertIn("human_count=1", first_state["messages"][-1].content)

        checkpoint_store.close()
        self._resources_to_close.remove(checkpoint_store)

        restarted_checkpoint_store = self._track_resource(build_checkpoint_store(self.settings))
        restarted_graph = self._build_counting_graph(restarted_checkpoint_store.saver)
        second_state = restarted_graph.invoke(
            {
                "messages": [HumanMessage(content="second slack turn")],
                "requested_agent": "",
                "requested_skill_ids": [],
                "context_paths": [],
            },
            config=thread_config,
        )

        self.assertIn("human_count=2", second_state["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
