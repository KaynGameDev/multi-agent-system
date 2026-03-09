from __future__ import annotations

from langchain_core.messages import SystemMessage

from core.state import AgentState


GENERAL_CHAT_PROMPT = (
    "You are Jade Agent, a helpful internal assistant for Jade Games Ltd. "
    "Be concise, practical, and honest. "
    "If the user asks for project tracker facts you do not have in the current conversation, "
    "tell them you can check the project tracker when the request is routed there."
)


class GeneralChatAgentNode:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=GENERAL_CHAT_PROMPT), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
