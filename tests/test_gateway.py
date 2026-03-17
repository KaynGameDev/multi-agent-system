from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from core.agent_registry import AgentRegistration
from core.gateway import GatewayNode, build_gateway_prompt


def build_agent_registrations() -> tuple[AgentRegistration, ...]:
    return (
        AgentRegistration(
            name="general_chat_agent",
            description="Handle greetings and casual chat.",
            build_node=lambda llm: object(),
        ),
        AgentRegistration(
            name="project_task_agent",
            description="Handle project tracker lookups.",
            build_node=lambda llm: object(),
        ),
        AgentRegistration(
            name="knowledge_agent",
            description="Handle internal documentation questions.",
            build_node=lambda llm: object(),
        ),
    )


class DummyStructuredRouter:
    def __init__(self, schema, response: dict[str, str]) -> None:
        self.schema = schema
        self.response = response
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.schema(**self.response)


class DummyLLM:
    def __init__(self, response: dict[str, str]) -> None:
        self.response = response
        self.router: DummyStructuredRouter | None = None

    def with_structured_output(self, schema):
        self.router = DummyStructuredRouter(schema, self.response)
        return self.router


class GatewayTests(unittest.TestCase):
    def test_build_gateway_prompt_lists_registered_agents(self) -> None:
        prompt = build_gateway_prompt(build_agent_registrations(), default_route="general_chat_agent")

        self.assertIn("knowledge_agent", prompt)
        self.assertIn("Handle internal documentation questions.", prompt)
        self.assertIn("choose general_chat_agent", prompt)

    def test_gateway_defaults_unknown_route_to_registered_default(self) -> None:
        llm = DummyLLM({"route": "nonexistent_agent", "reason": "unknown"})
        node = GatewayNode(
            llm,
            agent_registrations=build_agent_registrations(),
            default_route="general_chat_agent",
        )

        result = node({"messages": [HumanMessage(content="hello")]})

        self.assertEqual(result["route"], "general_chat_agent")
        self.assertEqual(result["route_reason"], "unknown")

    def test_gateway_uses_latest_human_message_not_latest_ai_message(self) -> None:
        llm = DummyLLM({"route": "knowledge_agent", "reason": "docs question"})
        node = GatewayNode(
            llm,
            agent_registrations=build_agent_registrations(),
            default_route="general_chat_agent",
        )

        node(
            {
                "messages": [
                    HumanMessage(content="First user question"),
                    AIMessage(content="assistant reply"),
                    HumanMessage(content="Second user question"),
                ]
            }
        )

        self.assertIsNotNone(llm.router)
        routed_messages = llm.router.invocations[0]
        self.assertEqual(routed_messages[-1].content, "Second user question")


if __name__ == "__main__":
    unittest.main()
