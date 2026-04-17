from __future__ import annotations

from langchain_core.messages import AIMessage

from app.contracts import build_assistant_response
from app.conversation_mode import build_conversation_mode_reply
from app.state import AgentState


class ConversationModeAgentNode:
    def __call__(self, state: AgentState) -> dict:
        content = build_conversation_mode_reply(str(state.get("requested_agent", "")).strip())
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(kind="text", content=content),
        }
