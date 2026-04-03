from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.graph import build_graph
from app.skills import SkillRegistry
from gateway.agent import AgentMatchResult
from interfaces.web.server import WebServer
from tests.common import build_registration, make_settings, write_skill


@tool
def echo_tool() -> str:
    """Return a fixed response."""

    return "echo-ok"


class LoopAgentNode:
    def __call__(self, state):
        latest_message = state["messages"][-1]
        if isinstance(latest_message, ToolMessage):
            return {
                "messages": [
                    AIMessage(content=f"Resolved skills: {', '.join(state.get('resolved_skill_ids', []))}")
                ]
            }
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo_tool",
                            "args": {},
                            "id": "call_echo_tool",
                        }
                    ],
                )
            ]
        }


class StaticAgentNode:
    def __call__(self, _state):
        return {"messages": [AIMessage(content="Static response.")]}


def always_match(_state, _latest_user_text: str) -> AgentMatchResult:
    return AgentMatchResult(matched=True, score=100, reasons=("Always matched for integration test.",))


class DeterministicIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_graph_preserves_resolved_skills_through_tool_loop(self) -> None:
        registrations = (
            build_registration(
                "general_chat_agent",
                namespace="general_chat",
                is_general_assistant=True,
                selection_order=20,
                build_node=lambda _llm=None, skill_registry=None: StaticAgentNode(),
            ),
            build_registration(
                "alpha_agent",
                namespace="alpha",
                selection_order=10,
                matcher=always_match,
                build_node=lambda _llm=None, skill_registry=None: LoopAgentNode(),
                tools=(echo_tool,),
            ),
        )
        write_skill(
            self.root,
            ".jade/skills/loop-skill",
            frontmatter={
                "name": "Loop Skill",
                "description": "Integration loop skill.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Loop Skill\n\nIntegration loop skill.",
        )
        registry = SkillRegistry(registrations, project_root=self.root)
        graph = build_graph(
            None,
            agent_registrations=registrations,
            default_route="alpha_agent",
            skill_registry=registry,
        )

        final_state = graph.invoke(
            {
                "messages": [HumanMessage(content="Run the alpha loop.")],
                "requested_agent": "alpha_agent",
                "requested_skill_ids": ["loop-skill"],
            }
        )

        self.assertEqual(final_state["route"], "alpha_agent")
        self.assertEqual(final_state["resolved_skill_ids"], ["loop-skill"])
        self.assertIn("loop-skill", getattr(final_state["messages"][-1], "content", ""))

    def test_web_api_returns_diagnostics_and_transport_does_not_force_route(self) -> None:
        settings = make_settings(self.root / "runtime")

        class FakeGraph:
            def __init__(self) -> None:
                self.last_state = None

            def invoke(self, initial_state, config=None):
                self.last_state = dict(initial_state)
                return {
                    **initial_state,
                    "route": "general_chat_agent",
                    "route_reason": "No specialist matcher applied; used GeneralAssistant fallback.",
                    "skill_resolution_diagnostics": [{"kind": "resolved", "skill_id": "none"}],
                    "agent_selection_diagnostics": [{"kind": "fallback", "selected_agent": "general_chat_agent"}],
                    "selection_warnings": ["test warning"],
                    "messages": [AIMessage(content="Hello from Jade.")],
                }

        fake_graph = FakeGraph()
        server = WebServer(agent_graph=fake_graph, settings=settings)
        client = TestClient(server.app)

        conversation = client.post("/api/conversations", json={"title": "Deterministic"}).json()
        response = client.post(
            f"/api/conversations/{conversation['conversation_id']}/messages",
            json={"message": "Hello there", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("route", fake_graph.last_state)
        self.assertIn("skill_resolution_diagnostics", payload)
        self.assertIn("agent_selection_diagnostics", payload)
        self.assertIn("selection_warnings", payload)
        self.assertEqual(payload["route"], "general_chat_agent")


if __name__ == "__main__":
    unittest.main()
