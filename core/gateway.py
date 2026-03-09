from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.config import load_settings
from core.state import AgentState, RouteName


class RouteDecision(BaseModel):
    route: Literal["project_task_agent", "general_chat_agent"] = Field(
        description="Which specialist should answer the user's latest request."
    )
    reason: str = Field(
        description="A brief explanation of why this route was chosen."
    )


GATEWAY_PROMPT = (
    "You are the routing gateway for Jade Agent. "
    "Your only job is to choose the best specialist for the user's latest message.\n\n"
    "Available specialists:\n"
    "- project_task_agent: questions about project tasks, owners, assignees, deadlines, milestones, roadmaps, or the Google Sheets tracker.\n"
    "- general_chat_agent: greetings, chit-chat, general questions, writing help, explanations, and anything that does not require the project tracker.\n\n"
    "Always return one route and a short reason."
)


class GatewayNode:
    def __init__(self, llm) -> None:
        self.router = llm.with_structured_output(RouteDecision)
        self.settings = load_settings()

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=GATEWAY_PROMPT), *state["messages"]]

        try:
            decision = self.router.invoke(messages)
            return {
                "route": decision.route,
                "route_reason": decision.reason,
            }
        except Exception:
            fallback_route = classify_with_keywords(state, self.settings.project_lookup_keywords)
            return {
                "route": fallback_route,
                "route_reason": "Fallback keyword routing.",
            }


def classify_with_keywords(
    state: AgentState,
    project_keywords: tuple[str, ...],
) -> RouteName:
    user_text = get_latest_user_text(state).lower()

    for keyword in project_keywords:
        if keyword in user_text:
            return "project_task_agent"

    return "general_chat_agent"


def get_latest_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return stringify_message_content(message.content)
    return ""


def stringify_message_content(content) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
    return " ".join(parts).strip()
