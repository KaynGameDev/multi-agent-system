from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from app.agent_registry import AgentRegistration
from app.messages import stringify_message_content

logger = logging.getLogger(__name__)


class ModelRouteReply(BaseModel):
    selected_agent: str = Field(
        default="",
        description="Exact specialist agent name to handle the turn. Return an empty string when no specialist is clearly needed.",
    )
    reason: str = Field(
        default="",
        description="Short internal routing reason.",
    )


class ModelRouter:
    def __init__(self, llm: Any | None) -> None:
        self.llm = llm
        structured_output = getattr(llm, "with_structured_output", None) if llm is not None else None
        self.response_llm = structured_output(ModelRouteReply) if callable(structured_output) else None
        self.uses_structured_output = self.response_llm is not None

    def select_specialist(
        self,
        *,
        agent_registrations: tuple[AgentRegistration, ...],
        general_assistant_name: str,
        latest_user_text: str,
        state: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]] | None:
        if not self.response_llm:
            return None

        specialists = tuple(
            registration
            for registration in agent_registrations
            if registration.name != general_assistant_name
        )
        if not specialists:
            return None

        messages = [
            SystemMessage(
                content=build_model_router_prompt(
                    specialists=specialists,
                    latest_user_text=latest_user_text,
                    state=state,
                )
            ),
            HumanMessage(content=latest_user_text),
        ]
        response = self._invoke(messages)
        if response is None:
            return None

        selected_agent = extract_model_route_value(response, "selected_agent")
        if selected_agent not in {registration.name for registration in specialists}:
            return None

        reason = extract_model_route_value(response, "reason").strip() or f"Model router selected `{selected_agent}`."
        diagnostics = [
            {
                "kind": "model_router_selected",
                "policy_step": "model_router",
                "selected_agent": selected_agent,
                "reason": reason,
            }
        ]
        return selected_agent, diagnostics

    def _invoke(self, messages: list[Any]) -> Any | None:
        try:
            return self.response_llm.invoke(messages)
        except Exception as exc:
            logger.warning("Model router failed; falling back to general route. error=%s", exc)
            if self.uses_structured_output and is_structured_output_contract_error(exc):
                self.response_llm = None
                self.uses_structured_output = False
            return None


def build_model_router_prompt(
    *,
    specialists: tuple[AgentRegistration, ...],
    latest_user_text: str,
    state: dict[str, Any],
) -> str:
    specialist_lines = "\n".join(
        f"- {registration.name}: {registration.description}"
        for registration in specialists
    )
    recent_context = render_recent_conversation(state)
    interface_name = str(state.get("interface_name", "")).strip().lower() or "unknown"
    uploaded_files = state.get("uploaded_files")
    upload_count = len(uploaded_files) if isinstance(uploaded_files, list) else 0
    conversion_session_active = bool(str(state.get("conversion_session_id", "")).strip())

    return (
        "You are Jade's internal agent router.\n"
        "Select the best specialist agent for the user's request.\n\n"
        "Rules:\n"
        "- Consider only the specialist agents listed below.\n"
        "- Choose a specialist only when it is clearly more appropriate than the general assistant.\n"
        "- If no specialist clearly fits, return an empty selected_agent.\n"
        "- Do not answer the user.\n\n"
        "Available specialist agents:\n"
        f"{specialist_lines}\n\n"
        "Conversation context:\n"
        f"- interface_name: {interface_name}\n"
        f"- uploaded_files_count: {upload_count}\n"
        f"- conversion_session_active: {str(conversion_session_active).lower()}\n"
        f"- latest_user_text: {latest_user_text}\n"
        f"- recent_messages:\n{recent_context}\n"
    )


def render_recent_conversation(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return "  (none)"

    rendered: list[str] = []
    for message in messages[-6:]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = type(message).__name__.replace("Message", "").lower() or "message"
        content = stringify_message_content(getattr(message, "content", "")).strip()
        if not content:
            continue
        rendered.append(f"  - {role}: {content}")
    return "\n".join(rendered) if rendered else "  (none)"


def extract_model_route_value(response: Any, field_name: str) -> str:
    value = getattr(response, field_name, None)
    if isinstance(value, str):
        return value.strip()
    if isinstance(response, dict):
        dict_value = response.get(field_name)
        if isinstance(dict_value, str):
            return dict_value.strip()
    return ""


def is_structured_output_contract_error(exc: Exception) -> bool:
    return isinstance(exc, ValidationError)
