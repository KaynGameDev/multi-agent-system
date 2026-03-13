from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import AgentState, RouteName


class RouteDecision(BaseModel):
    route: RouteName = Field(description="The next agent that should handle the request.")
    reason: str = Field(description="Brief reason for the routing decision.")


GATEWAY_PROMPT = (
    "You are the gateway router for Jade Games Ltd.'s multi-agent system. "
    "Your job is to classify the user's latest message and choose the correct agent.\n\n"
    "Available routes:\n"
    "- project_task_agent: Use for project tracker questions, assignees, deadlines, schedules, priorities, "
    "iterations, project status, or anything that likely requires Google Sheets data.\n"
    "- general_chat_agent: Use for greetings, casual chat, and general knowledge questions that do not require project data.\n\n"
    "Return the best route and a short reason."
)


class GatewayNode:
    def __init__(self, llm) -> None:
        self.router = llm.with_structured_output(RouteDecision)

    def __call__(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        latest_user_message = self._get_latest_user_message(messages)

        if not latest_user_message:
            return {
                "route": "general_chat_agent",
                "route_reason": "No user message found; defaulting to general chat.",
            }

        decision = self.router.invoke(
            [
                SystemMessage(content=GATEWAY_PROMPT),
                HumanMessage(content=latest_user_message),
            ]
        )

        return {
            "route": decision.route,
            "route_reason": decision.reason,
        }

    def _get_latest_user_message(self, messages) -> str:
        for message in reversed(messages):
            content = getattr(message, "content", "")
            text = self._stringify_content(content)
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