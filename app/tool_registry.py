from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class ToolMetadata:
    tool_id: str
    runtime_tool_name: str
    display_name: str
    description: str
    tool_family: str
    follow_up_hint: str
    semantic_aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


TOOL_KNOWLEDGE_LIST_DOCUMENTS = "knowledge.list_documents"
TOOL_KNOWLEDGE_SEARCH_DOCUMENTS = "knowledge.search_documents"
TOOL_KNOWLEDGE_READ_DOCUMENT = "knowledge.read_document"
TOOL_KNOWLEDGE_RESOLVE_MARKDOWN_PATH = "knowledge.resolve_markdown_path"
TOOL_KNOWLEDGE_WRITE_MARKDOWN = "knowledge.write_markdown"
TOOL_PROJECT_READ_TASKS = "project.read_tasks"
TOOL_PROJECT_SHEET_OVERVIEW = "project.sheet_overview"

KNOWLEDGE_TOOL_IDS = (
    TOOL_KNOWLEDGE_LIST_DOCUMENTS,
    TOOL_KNOWLEDGE_SEARCH_DOCUMENTS,
    TOOL_KNOWLEDGE_READ_DOCUMENT,
)
KNOWLEDGE_BUILDER_TOOL_IDS = (
    *KNOWLEDGE_TOOL_IDS,
    TOOL_KNOWLEDGE_RESOLVE_MARKDOWN_PATH,
    TOOL_KNOWLEDGE_WRITE_MARKDOWN,
)
PROJECT_TOOL_IDS = (
    TOOL_PROJECT_READ_TASKS,
    TOOL_PROJECT_SHEET_OVERVIEW,
)

_TOOL_REGISTRY = (
    ToolMetadata(
        tool_id=TOOL_KNOWLEDGE_LIST_DOCUMENTS,
        runtime_tool_name="list_knowledge_documents",
        display_name="List Knowledge Documents",
        description="Browse available knowledge-base documents and source files.",
        tool_family="knowledge_read",
        follow_up_hint="selection",
        semantic_aliases=(
            "list docs",
            "list documents",
            "available docs",
            "available documents",
            "what docs",
            "what documents",
            "knowledge docs",
            "knowledge base docs",
            "文档列表",
            "有哪些文档",
            "列出文档",
            "知识库文档",
        ),
        examples=(
            "show me the available docs",
            "can you list the knowledge base documents",
        ),
    ),
    ToolMetadata(
        tool_id=TOOL_KNOWLEDGE_SEARCH_DOCUMENTS,
        runtime_tool_name="search_knowledge_documents",
        display_name="Search Knowledge Documents",
        description="Search the knowledge base for matching documents, sections, and snippets.",
        tool_family="knowledge_read",
        follow_up_hint="selection",
        semantic_aliases=(
            "search docs",
            "search documents",
            "find docs",
            "find documents",
            "search the knowledge base",
            "知识库搜索",
            "搜索文档",
            "查文档",
            "找文档",
        ),
        examples=(
            "find docs about setup",
            "search the knowledge base for tower defense",
        ),
    ),
    ToolMetadata(
        tool_id=TOOL_KNOWLEDGE_READ_DOCUMENT,
        runtime_tool_name="read_knowledge_document",
        display_name="Read Knowledge Document",
        description="Open a knowledge-base document and read its content or a matching section.",
        tool_family="knowledge_read",
        follow_up_hint="selection",
        semantic_aliases=(
            "read doc",
            "read document",
            "open doc",
            "open document",
            "show document",
            "show doc",
            "setup doc",
            "architecture doc",
            "读文档",
            "看文档",
            "打开文档",
            "查看文档",
            "展示文档",
        ),
        examples=(
            "can you read the setup document",
            "open that architecture doc",
        ),
    ),
    ToolMetadata(
        tool_id=TOOL_KNOWLEDGE_RESOLVE_MARKDOWN_PATH,
        runtime_tool_name="resolve_knowledge_markdown_path",
        display_name="Resolve KB Markdown Path",
        description="Resolve the canonical Markdown path for a knowledge-base draft under knowledge/Docs/.",
        tool_family="knowledge_write",
        follow_up_hint="path_preview",
        semantic_aliases=(
            "kb path",
            "knowledge base path",
            "resolve path",
            "suggest path",
            "文档路径",
            "知识库路径",
            "落点路径",
            "建议路径",
        ),
        examples=(
            "where should this kb draft go",
            "resolve the path for this knowledge base document",
        ),
    ),
    ToolMetadata(
        tool_id=TOOL_KNOWLEDGE_WRITE_MARKDOWN,
        runtime_tool_name="write_knowledge_markdown_document",
        display_name="Write KB Markdown Draft",
        description=(
            "Create or update a Markdown draft under knowledge/Docs/ after previewing the target path and "
            "receiving explicit confirmation."
        ),
        tool_family="knowledge_write",
        follow_up_hint="confirmation",
        semantic_aliases=(
            "write files",
            "write file",
            "edit files",
            "edit file",
            "create files",
            "create file",
            "save file",
            "save files",
            "write knowledge base",
            "write to knowledge base",
            "save to knowledge base",
            "update knowledge base",
            "create knowledge doc",
            "update knowledge doc",
            "kb draft",
            "markdown draft",
            "save our discussion",
            "save this discussion",
            "save this chat",
            "discussion to the knowledge base",
            "写文件",
            "改文件",
            "创建文件",
            "保存文件",
            "写入知识库",
            "保存到知识库",
            "更新知识库",
            "创建知识库文档",
            "更新知识库文档",
            "把讨论保存到知识库",
            "把聊天记录存到知识库",
            "沉淀成知识库文档",
            "落成草稿",
        ),
        examples=(
            "can you write files",
            "can you save our discussion to the knowledge base",
        ),
    ),
    ToolMetadata(
        tool_id=TOOL_PROJECT_READ_TASKS,
        runtime_tool_name="read_project_tasks",
        display_name="Read Project Tasks",
        description="Query the project tracker for tasks, deadlines, assignees, priorities, and due scopes.",
        tool_family="project_tracker",
        follow_up_hint="selection",
        semantic_aliases=(
            "project tracker",
            "task list",
            "tasks",
            "deadlines",
            "my work",
            "my tasks",
            "what am i doing",
            "due today",
            "due this week",
            "任务",
            "任务列表",
            "截止",
            "逾期任务",
            "今天到期",
            "本周任务",
        ),
        examples=(
            "show my tasks due today",
            "what is Alice working on",
        ),
    ),
    ToolMetadata(
        tool_id=TOOL_PROJECT_SHEET_OVERVIEW,
        runtime_tool_name="get_project_sheet_overview",
        display_name="Project Tracker Overview",
        description="Return a structured overview and preview of the project tracker sheet.",
        tool_family="project_tracker",
        follow_up_hint="selection",
        semantic_aliases=(
            "project overview",
            "tracker overview",
            "sheet overview",
            "task overview",
            "任务总览",
            "看板概览",
            "项目概览",
        ),
        examples=(
            "show me the tracker overview",
            "give me a quick project sheet preview",
        ),
    ),
)

_TOOL_METADATA_BY_ID = {metadata.tool_id: metadata for metadata in _TOOL_REGISTRY}
_TOOL_METADATA_BY_RUNTIME_NAME = {metadata.runtime_tool_name: metadata for metadata in _TOOL_REGISTRY}


def list_tool_metadata(tool_ids: Iterable[str] | None = None) -> tuple[ToolMetadata, ...]:
    if tool_ids is None:
        return _TOOL_REGISTRY
    return tuple(get_tool_metadata(tool_id) for tool_id in tool_ids)


def get_tool_metadata(tool_id: str) -> ToolMetadata:
    try:
        return _TOOL_METADATA_BY_ID[tool_id]
    except KeyError as exc:
        raise KeyError(f"Unknown tool id: {tool_id}") from exc


def get_tool_metadata_by_runtime_name(runtime_tool_name: str) -> ToolMetadata:
    try:
        return _TOOL_METADATA_BY_RUNTIME_NAME[runtime_tool_name]
    except KeyError as exc:
        raise KeyError(f"Runtime tool `{runtime_tool_name}` is missing metadata.") from exc


def resolve_tool_ids_for_runtime_tools(tools: Iterable[BaseTool]) -> tuple[str, ...]:
    return tuple(get_tool_metadata_by_runtime_name(tool.name).tool_id for tool in tools)


def build_agent_tool_prompt(tool_ids: Iterable[str]) -> str:
    metadata_items = list_tool_metadata(tool_ids)
    if not metadata_items:
        return ""

    lines = [
        "# Available Tools",
        "Answer questions about what you can or cannot do based on the tools below. "
        "Do not claim generic access outside this list.",
    ]
    for metadata in metadata_items:
        lines.append(f"- {metadata.display_name} (`{metadata.tool_id}`): {metadata.description}")
    return "\n".join(lines).strip()
