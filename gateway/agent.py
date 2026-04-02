from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field, create_model

from app.agent_registry import AgentRegistration
from app.state import AgentState


def build_gateway_prompt(agent_registrations: Sequence[AgentRegistration], default_route: str) -> str:
    lines = [
        "You are the gateway router for Jade Games Ltd.'s multi-agent system.",
        "Your job is to classify the user's latest message and choose the correct agent.",
        "",
        "Available routes:",
    ]

    for registration in agent_registrations:
        lines.append(f"- {registration.name}: {registration.description}")

    lines.extend(
        [
            "",
            f"If the user's message is ambiguous or no route clearly matches, choose {default_route}.",
            "Return the best route and a short reason.",
        ]
    )
    return "\n".join(lines)


def build_route_decision_model(agent_registrations: Sequence[AgentRegistration]):
    route_names = ", ".join(registration.name for registration in agent_registrations)
    return create_model(
        "RouteDecision",
        route=(
            str,
            Field(description=f"The next agent that should handle the request. Valid routes: {route_names}."),
        ),
        reason=(str, Field(description="Brief reason for the routing decision.")),
    )


class GatewayNode:
    def __init__(
        self,
        llm,
        *,
        agent_registrations: Sequence[AgentRegistration],
        default_route: str,
    ) -> None:
        self.agent_registrations = tuple(agent_registrations)
        self.valid_routes = {registration.name for registration in self.agent_registrations}
        self.default_route = default_route
        self.prompt = build_gateway_prompt(self.agent_registrations, default_route)
        self.route_decision_model = build_route_decision_model(self.agent_registrations)
        self.router = llm.with_structured_output(self.route_decision_model)

    def __call__(self, state: AgentState) -> dict:
        requested_route = str(state.get("route", "")).strip()
        if requested_route in self.valid_routes:
            reason = str(state.get("route_reason", "")).strip() or f"Route override selected {requested_route}."
            return {
                "route": requested_route,
                "route_reason": reason,
            }

        messages = state.get("messages", [])
        latest_user_message = self._get_latest_user_message(messages)

        if not latest_user_message:
            return {
                "route": self.default_route,
                "route_reason": f"No user message found; defaulting to {self.default_route}.",
            }

        decision = self.router.invoke(
            [
                SystemMessage(content=self.prompt),
                HumanMessage(content=latest_user_message),
            ]
        )

        requested_route = self._get_decision_value(decision, "route")
        route = self._normalize_route(requested_route)
        reason = self._get_decision_value(decision, "reason").strip()
        if not reason:
            reason = f"Gateway selected {route}."

        return {
            "route": route,
            "route_reason": reason,
        }

    def _normalize_route(self, route: str) -> str:
        normalized = route.strip()
        if normalized in self.valid_routes:
            return normalized
        return self.default_route

    def _get_decision_value(self, decision, field_name: str) -> str:
        value = getattr(decision, field_name, "")
        if not value and isinstance(decision, dict):
            value = decision.get(field_name, "")
        return str(value) if value is not None else ""

    def _get_latest_user_message(self, messages) -> str:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            text = self._stringify_content(message.content)
            if text:
                return text
        return ""

    def _stringify_content(self, content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()

        return str(content)
