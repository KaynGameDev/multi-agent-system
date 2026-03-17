from __future__ import annotations

from langchain_core.messages import SystemMessage

from core.state import AgentState


KNOWLEDGE_AGENT_PROMPT = (
    "You are the Knowledge Agent for Jade Games Ltd. "
    "Answer questions about internal documentation, architecture, setup, workflow, and operational guidance that are documented in the repository. "
    "Use the knowledge tools whenever the answer depends on internal docs or project documentation. "
    "Do not invent undocumented behavior. If the documentation is missing or unclear, say so plainly. "
    "Summarize the relevant document content in concise Markdown that stays easy to read in Slack. "
    "When helpful, mention which document you used."
)


class KnowledgeAgentNode:
    def __init__(self, llm, tools: list) -> None:
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=KNOWLEDGE_AGENT_PROMPT), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
