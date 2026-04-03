from __future__ import annotations

from langchain_core.messages import SystemMessage

from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skills import SkillRegistry
from app.state import AgentState

PROMPT_PATH = "agents/general_chat/AGENT.md"


class GeneralChatAgentNode:
    def __init__(self, llm, *, skill_registry: SkillRegistry | None = None, agent_name: str = "") -> None:
        self.llm = llm
        self.skill_registry = skill_registry
        self.agent_name = agent_name

    def __call__(self, state: AgentState) -> dict:
        messages = [
            SystemMessage(
                content=build_general_chat_prompt(
                    state,
                    skill_registry=self.skill_registry,
                    agent_name=self.agent_name,
                )
            ),
            *state["messages"],
        ]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_general_chat_prompt(
    state: AgentState,
    *,
    skill_registry: SkillRegistry | None = None,
    agent_name: str = "",
) -> str:
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
    skill_prompt = (
        skill_registry.build_prompt_layers(
            state.get("resolved_skill_ids", []),
            agent_name=agent_name,
            context_paths=state.get("context_paths", []),
        )
        if skill_registry is not None and agent_name
        else ""
    )
    return join_prompt_layers(
        load_shared_instruction_text(),
        sections["role"],
        sections["responsibilities"],
        skill_prompt,
        sections["boundaries"],
        format_prompt,
    )
