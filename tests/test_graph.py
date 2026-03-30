from __future__ import annotations

import unittest

from langchain_core.tools import tool

from core.agent_registry import AgentRegistration
from core.graph import (
    build_default_agent_registrations,
    build_web_agent_registrations,
    build_graph,
    normalize_agent_registrations,
    resolve_default_route,
    resolve_gateway_route,
)


class DummyLLM:
    def with_structured_output(self, schema):
        return object()


def noop_agent(_llm):
    def node(_state):
        return {"messages": []}

    return node


@tool
def sample_tool() -> str:
    """Return a placeholder value."""
    return "ok"


class GraphTests(unittest.TestCase):
    def test_default_agent_registrations_include_knowledge_agent(self) -> None:
        registrations = build_default_agent_registrations()
        registration_names = [registration.name for registration in registrations]

        self.assertIn("knowledge_agent", registration_names)
        knowledge_registration = next(
            registration for registration in registrations if registration.name == "knowledge_agent"
        )
        self.assertEqual(len(knowledge_registration.tools), 3)

    def test_web_agent_registrations_include_document_conversion_agent(self) -> None:
        registrations = build_web_agent_registrations()
        registration_names = [registration.name for registration in registrations]

        self.assertEqual(
            registration_names,
            ["general_chat_agent", "knowledge_agent", "project_task_agent", "document_conversion_agent"],
        )

    def test_build_graph_supports_custom_agent_registrations(self) -> None:
        registrations = (
            AgentRegistration(
                name="knowledge_agent",
                description="Handle docs.",
                build_node=noop_agent,
            ),
            AgentRegistration(
                name="planner_agent",
                description="Handle planning with tools.",
                build_node=noop_agent,
                tools=(sample_tool,),
            ),
        )

        graph = build_graph(
            DummyLLM(),
            agent_registrations=registrations,
            default_route="knowledge_agent",
        )

        self.assertIsNotNone(graph)

    def test_normalize_agent_registrations_rejects_duplicate_names(self) -> None:
        registrations = (
            AgentRegistration(name="knowledge_agent", description="Handle docs.", build_node=noop_agent),
            AgentRegistration(name="knowledge_agent", description="Duplicate.", build_node=noop_agent),
        )

        with self.assertRaisesRegex(ValueError, "Duplicate agent registration name"):
            normalize_agent_registrations(registrations)

    def test_resolve_default_route_uses_first_registered_agent_when_unspecified(self) -> None:
        registrations = (
            AgentRegistration(name="knowledge_agent", description="Handle docs.", build_node=noop_agent),
            AgentRegistration(name="planner_agent", description="Handle planning.", build_node=noop_agent),
        )

        self.assertEqual(resolve_default_route(registrations), "knowledge_agent")

    def test_resolve_gateway_route_falls_back_when_state_route_is_unknown(self) -> None:
        route = resolve_gateway_route(
            {"route": "missing_agent"},
            valid_routes={"knowledge_agent", "planner_agent"},
            default_route="knowledge_agent",
        )

        self.assertEqual(route, "knowledge_agent")


if __name__ == "__main__":
    unittest.main()
