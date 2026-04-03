from __future__ import annotations

from langchain_core.messages import SystemMessage

from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skills import SkillRegistry
from app.state import AgentState

PROMPT_PATH = "agents/knowledge_base_builder/AGENT.md"


class KnowledgeBaseBuilderAgentNode:
    def __init__(self, llm, tools: list, *, skill_registry: SkillRegistry | None = None, agent_name: str = "") -> None:
        self.llm = llm.bind_tools(tools)
        self.skill_registry = skill_registry
        self.agent_name = agent_name

    def __call__(self, state: AgentState) -> dict:
        messages = [
            SystemMessage(
                content=build_knowledge_base_builder_prompt(
                    state,
                    skill_registry=self.skill_registry,
                    agent_name=self.agent_name,
                )
            ),
            *state["messages"],
        ]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_knowledge_base_builder_prompt(
    state: AgentState,
    *,
    skill_registry: SkillRegistry | None = None,
    agent_name: str = "",
) -> str:
    sections = load_prompt_sections(
        PROMPT_PATH,
        required_sections=("role", "responsibilities", "tool_usage", "boundaries", "output"),
    )
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
        sections["tool_usage"],
        skill_prompt,
        sections["boundaries"],
        sections["output"],
    )
