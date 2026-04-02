from __future__ import annotations

from langchain_core.messages import SystemMessage

from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.state import AgentState

PROMPT_PATH = "agents/general_chat/AGENT.md"


class GeneralChatAgentNode:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=build_general_chat_prompt(state)), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_general_chat_prompt(state: AgentState) -> str:
    sections = load_prompt_sections(
        PROMPT_PATH,
        required_sections=(
            "role",
            "responsibilities",
            "boundaries",
            "slack_output",
            "web_output",
            "default_output",
        ),
    )
    interface_name = str(state.get("interface_name", "")).strip().lower()
    format_prompt = sections["default_output"]
    if interface_name == "slack":
        format_prompt = sections["slack_output"]
    elif interface_name == "web":
        format_prompt = sections["web_output"]
    return join_prompt_layers(
        load_shared_instruction_text(),
        sections["role"],
        sections["responsibilities"],
        sections["boundaries"],
        format_prompt,
    )
