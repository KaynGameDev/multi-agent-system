from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_default_agent_registrations, build_graph, build_web_agent_registrations
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
            active_skill_ids = [
                str(item.get("skill_id", "")).strip()
                for item in state.get("active_skill_invocation_contracts", [])
                if isinstance(item, dict)
            ]
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Resolved skills: {', '.join(state.get('resolved_skill_ids', []))}; "
                            f"active contracts: {', '.join(active_skill_ids)}"
                        )
                    )
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


def keyword_match(keyword: str, *, score: int = 100):
    def matcher(_state, latest_user_text: str) -> AgentMatchResult:
        if keyword in latest_user_text.lower():
            return AgentMatchResult(matched=True, score=score, reasons=(f"Matched `{keyword}`.",))
        return AgentMatchResult(matched=False, score=0, reasons=())

    return matcher


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
        self.assertEqual(final_state["skill_invocation_contracts"][0]["skill_id"], "loop-skill")
        self.assertEqual(final_state["active_skill_invocation_contracts"][0]["target_agent"], "alpha_agent")
        self.assertEqual(final_state["skill_execution_diagnostics"][0]["executed_by_agent"], "alpha_agent")
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
        self.assertEqual(fake_graph.last_state["requested_agent"], "")
        self.assertEqual(fake_graph.last_state["requested_skill_ids"], [])
        self.assertEqual(fake_graph.last_state["uploaded_files"], [])
        self.assertEqual(fake_graph.last_state["context_paths"], [])
        self.assertEqual(fake_graph.last_state["conversion_session_id"], "")
        self.assertIn("skill_resolution_diagnostics", payload)
        self.assertIn("agent_selection_diagnostics", payload)
        self.assertIn("selection_warnings", payload)
        self.assertEqual(payload["route"], "general_chat_agent")

    def test_checkpointed_graph_does_not_reuse_prior_route_or_requested_skills(self) -> None:
        registrations = (
            build_registration(
                "general_chat_agent",
                namespace="general_chat",
                is_general_assistant=True,
                selection_order=30,
                build_node=lambda _llm=None, skill_registry=None: StaticAgentNode(),
            ),
            build_registration(
                "alpha_agent",
                namespace="alpha",
                selection_order=10,
                matcher=keyword_match("alpha"),
                build_node=lambda _llm=None, skill_registry=None: StaticAgentNode(),
            ),
            build_registration(
                "beta_agent",
                namespace="beta",
                selection_order=20,
                matcher=keyword_match("beta"),
                build_node=lambda _llm=None, skill_registry=None: StaticAgentNode(),
            ),
        )
        write_skill(
            self.root,
            ".jade/skills/alpha-skill",
            frontmatter={
                "name": "Alpha Skill",
                "description": "Integration alpha skill.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Alpha Skill\n\nIntegration alpha skill.",
        )
        registry = SkillRegistry(registrations, project_root=self.root)
        graph = build_graph(
            None,
            checkpointer=InMemorySaver(),
            agent_registrations=registrations,
            default_route="general_chat_agent",
            skill_registry=registry,
        )
        config = {"configurable": {"thread_id": "integration-thread"}}

        first_state = graph.invoke(
            {
                "messages": [HumanMessage(content="please alpha")],
                "requested_agent": "",
                "requested_skill_ids": ["alpha-skill"],
                "context_paths": [],
            },
            config=config,
        )
        second_state = graph.invoke(
            {
                "messages": [HumanMessage(content="please beta")],
                "requested_agent": "",
                "requested_skill_ids": [],
                "context_paths": [],
            },
            config=config,
        )

        self.assertEqual(first_state["route"], "alpha_agent")
        self.assertEqual(first_state["resolved_skill_ids"], ["alpha-skill"])
        self.assertEqual(second_state["route"], "beta_agent")
        self.assertEqual(second_state["requested_skill_ids"], [])
        self.assertEqual(second_state["resolved_skill_ids"], [])
        self.assertNotIn("Explicit requested agent", second_state["route_reason"])

    def test_default_and_web_registrations_include_builder_agent(self) -> None:
        settings = make_settings(self.root / "runtime")

        default_names = {registration.name for registration in build_default_agent_registrations(settings=settings)}
        web_names = {registration.name for registration in build_web_agent_registrations(settings=settings)}

        self.assertIn("knowledge_base_builder_agent", default_names)
        self.assertIn("knowledge_base_builder_agent", web_names)


if __name__ == "__main__":
    unittest.main()
