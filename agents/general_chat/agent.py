from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic import ValidationError
from langchain_core.messages import AIMessage, SystemMessage

from app.contracts import build_assistant_response
from app.messages import stringify_message_content
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skill_runtime import build_skill_prompt_context
from app.skills import SkillRegistry
from app.state import AgentState

PROMPT_PATH = "agents/general_chat/AGENT.md"
logger = logging.getLogger(__name__)


class GeneralChatReply(BaseModel):
    final_answer: str = Field(
        default="",
        description="User-facing final reply in Markdown. Return only the answer, with no analysis or role labels.",
    )


class GeneralChatAgentNode:
    def __init__(self, llm, *, skill_registry: SkillRegistry | None = None, agent_name: str = "") -> None:
        self.llm = llm
        structured_output = getattr(llm, "with_structured_output", None)
        self.response_llm = structured_output(GeneralChatReply) if callable(structured_output) else llm
        self.uses_structured_output = self.response_llm is not llm
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
        response = self._invoke_response_model(messages)
        assistant_text = extract_general_chat_reply_text(response)
        return {
            "messages": [AIMessage(content=assistant_text)],
            "assistant_response": build_assistant_response(
                kind="text",
                content=assistant_text,
            ),
        }

    def _invoke_response_model(self, messages):
        try:
            return self.response_llm.invoke(messages)
        except Exception as exc:
            if not self.uses_structured_output or not is_structured_output_contract_error(exc):
                raise
            logger.warning(
                "General chat structured output failed validation; falling back to plain-text invocation. error=%s",
                exc,
            )
            self.response_llm = self.llm
            self.uses_structured_output = False
            return self.llm.invoke(messages)


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
            "output_contract",
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
    skill_prompt = build_skill_prompt_context(
        state,
        skill_registry=skill_registry,
        agent_name=agent_name,
    )
    return join_prompt_layers(
        load_shared_instruction_text(),
        sections["role"],
        sections["responsibilities"],
        skill_prompt,
        sections["boundaries"],
        sections["output_contract"],
        format_prompt,
    )


def is_structured_output_contract_error(exc: Exception) -> bool:
    return isinstance(exc, ValidationError)


def extract_general_chat_reply_text(response) -> str:
    final_answer = getattr(response, "final_answer", None)
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer.strip()
    if isinstance(response, dict):
        dict_answer = response.get("final_answer")
        if isinstance(dict_answer, str) and dict_answer.strip():
            return dict_answer.strip()
    return stringify_message_content(getattr(response, "content", response))
