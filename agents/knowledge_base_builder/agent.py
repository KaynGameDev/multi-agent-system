from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

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
        rendered_response = build_knowledge_base_builder_response(state)
        if rendered_response is not None:
            return {"messages": [AIMessage(content=rendered_response)]}

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


def build_knowledge_base_builder_response(state: AgentState) -> str | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    payload = get_builder_tool_payload(messages[-1])
    if payload is None:
        return None

    if payload.get("knowledge_mutation") == "write_markdown":
        return render_write_knowledge_payload(payload)

    return None


def get_builder_tool_payload(message) -> dict | None:
    if not isinstance(message, ToolMessage):
        return None
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def render_write_knowledge_payload(payload: dict) -> str:
    relative_path = str(payload.get("relative_path", "")).strip()
    error = str(payload.get("error", "")).strip()

    if payload.get("ok") is True:
        action = "已更新" if payload.get("overwritten") else "已创建"
        return f"{action}知识库草稿：`{relative_path}`"

    if payload.get("requires_confirmation") is True:
        target_exists = payload.get("target_exists") is True
        action_text = "覆盖更新" if target_exists else "创建草稿"
        return (
            "这次知识库文件操作需要你先确认，我还没有实际写入。\n\n"
            f"- 目标路径：`{relative_path}`\n"
            f"- 计划动作：{action_text}\n\n"
            "请直接回复 `approve` / `confirm`，或回复 `批准` / `确认`，我再执行写入。"
        )

    if error:
        return f"知识库写入失败：{error}"
    return "知识库写入未完成。"
