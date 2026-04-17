from __future__ import annotations

from dataclasses import dataclass
import re

KNOWLEDGE_BUILD_MODE = "knowledge_build"
KNOWLEDGE_BUILD_REQUESTED_AGENT = "knowledge_base_builder_agent"
CONVERSATION_MODE_COMMAND_AGENT = "conversation_mode_agent"

_MODE_COMMAND_PATTERN = re.compile(
    r"^/(?P<command>kb|knowledge-build)(?:\s+(?P<action>on|off|enter|exit|start|stop))?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationModeCommand:
    mode: str
    requested_agent: str
    action: str


def conversation_mode_for_requested_agent(requested_agent: str) -> str:
    normalized_agent = str(requested_agent or "").strip()
    if normalized_agent == KNOWLEDGE_BUILD_REQUESTED_AGENT:
        return KNOWLEDGE_BUILD_MODE
    return ""


def parse_conversation_mode_command(
    user_text: str,
    *,
    current_requested_agent: str = "",
) -> ConversationModeCommand | None:
    normalized_text = str(user_text or "").strip()
    match = _MODE_COMMAND_PATTERN.match(normalized_text)
    if match is None:
        return None

    action = str(match.group("action") or "").strip().lower()
    current_mode = conversation_mode_for_requested_agent(current_requested_agent)

    if action in {"on", "enter", "start"}:
        requested_agent = KNOWLEDGE_BUILD_REQUESTED_AGENT
    elif action in {"off", "exit", "stop"}:
        requested_agent = ""
    else:
        requested_agent = (
            ""
            if current_mode == KNOWLEDGE_BUILD_MODE
            else KNOWLEDGE_BUILD_REQUESTED_AGENT
        )

    return ConversationModeCommand(
        mode=KNOWLEDGE_BUILD_MODE,
        requested_agent=requested_agent,
        action=action or "toggle",
    )


def build_conversation_mode_reply(requested_agent: str) -> str:
    if conversation_mode_for_requested_agent(requested_agent) == KNOWLEDGE_BUILD_MODE:
        return (
            "Knowledge Build Mode is on. I’ll keep routing this conversation to the "
            "knowledge builder until you turn it off with `/kb off`."
        )
    return (
        "Knowledge Build Mode is off. New messages will use normal routing again. "
        "Use `/kb` to turn it back on."
    )
