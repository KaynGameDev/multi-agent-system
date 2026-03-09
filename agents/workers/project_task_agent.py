from __future__ import annotations

from langchain_core.messages import SystemMessage

from core.state import AgentState


PROJECT_TASK_PROMPT = (
    "You are the Project Task Agent for Jade Games Ltd. "
    "Answer questions about the Google Sheets project tracker. "
    "Use the project tools whenever the answer depends on task data, assignees, owners, deadlines, or statuses. "
    "Do not invent sheet data. If the sheet does not contain the requested information, say so clearly. "
    "Format the final answer cleanly with short bullets when that improves readability."
)


class ProjectTaskAgentNode:
    def __init__(self, llm, tools: list) -> None:
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=PROJECT_TASK_PROMPT), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
