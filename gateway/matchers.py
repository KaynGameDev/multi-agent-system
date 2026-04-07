from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage

from app.agent_registry import AgentRegistration
from app.contracts import normalize_tool_result_envelope
from gateway.text_utils import is_casual_greeting, normalize_text
from tools.conversion_google_sources import extract_google_document_references

UPLOAD_PATTERN = re.compile(r"\b(upload|attach|attachment|file)\b", re.IGNORECASE)
CONVERSION_PATTERN = re.compile(r"\b(convert|conversion|knowledge package)\b", re.IGNORECASE)
KNOWLEDGE_KEYWORDS = (
    "architecture",
    "setup",
    "repository",
    "repo",
    "documentation",
    "document",
    "docs",
    "workflow",
    "process",
    "readme",
    "guide",
    "knowledge base",
    "知识库",
    "文档",
    "架构",
    "流程",
    "仓库",
)
KB_BUILDER_ELICITATION_PHRASES = (
    "一步步提问",
    "逐步提问",
    "分步提问",
    "知识抽取",
    "抽取知识",
    "主持这轮讨论",
    "主持这轮梳理",
    "引导我们梳理",
    "帮我们梳理",
    "实时总结",
    "实时归纳",
    "文档骨架",
    "feature spec 骨架",
    "master gdd 骨架",
    "沉淀成文档",
    "沉淀为文档",
)
KB_BUILDER_REVIEW_PHRASES = (
    "review 这份 kb 文档",
    "review 这份文档",
    "review kb doc",
    "审查这份文档",
    "检查 metadata",
    "metadata 和层级",
    "层级归属",
    "可批准状态",
    "建议状态",
    "review 某份",
)
KB_BUILDER_TRACKING_PHRASES = (
    "kb v1",
    "milestone",
    "当前阶段",
    "推进到哪",
    "推进到哪里",
    "阻塞项",
    "下一步动作",
    "下一步建议",
    "status",
)
KB_BUILDER_LAYER_PHRASES = (
    "shared",
    "game line",
    "gameline",
    "deployment",
    "legacy",
    "shared 层",
    "gameline 层",
    "deployment 层",
    "legacy 层",
    "四层",
    "知识拓扑",
)
KB_BUILDER_LAYER_DECISION_PHRASES = (
    "归属",
    "落点",
    "边界",
    "区别",
    "哪一层",
    "放在哪层",
    "放到哪层",
)
KB_BUILDER_FILE_WRITE_PHRASES = (
    "写入文件",
    "写入知识库",
    "写到知识库",
    "存入知识库",
    "更新知识库",
    "更新到知识库",
    "保存到知识库",
    "创建知识库文档",
    "更新知识库文档",
    "修改知识库文档",
    "写文件",
    "改文件",
    "创建文件",
    "保存文件",
    "删除文件",
    "能写文件吗",
    "可以写文件吗",
    "你能写文件吗",
    "能改文件吗",
    "可以改文件吗",
    "能创建文件吗",
    "可以创建文件吗",
    "有写入权限吗",
    "有修改权限吗",
    "can you write files",
    "can you write file",
    "can you edit files",
    "can you edit file",
    "can you create files",
    "can you create file",
    "can you save files",
    "can you save file",
    "can you delete files",
    "can you modify files",
    "write to the knowledge base",
    "write knowledge base files",
    "edit knowledge base files",
)
PROJECT_PATTERNS = (
    "my tasks",
    "my work",
    "my deadlines",
    "what am i doing",
    "what is ",
    "who is working on",
    "who owns",
    "due this week",
    "due today",
    "逾期任务",
    "任务",
    "截止",
    "负责人",
)


@dataclass(frozen=True)
class AgentMatchResult:
    matched: bool
    score: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate_agent_match(registration: AgentRegistration, state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    matcher = registration.matcher
    if callable(matcher):
        result = matcher(state, latest_user_text)
        if isinstance(result, AgentMatchResult):
            return result
    return AgentMatchResult(matched=False, score=0, reasons=())


def document_conversion_matcher(state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    uploaded_files = state.get("uploaded_files")
    if isinstance(uploaded_files, list) and uploaded_files:
        return AgentMatchResult(matched=True, score=100, reasons=("Uploaded files require document conversion.",))

    if extract_google_document_references(latest_user_text):
        return AgentMatchResult(matched=True, score=95, reasons=("Google Docs or Sheets links trigger document conversion.",))

    conversion_session_id = str(state.get("conversion_session_id", "")).strip()
    if conversion_session_id and not is_casual_greeting(latest_user_text):
        return AgentMatchResult(matched=True, score=90, reasons=("Active conversion session should continue in document conversion.",))

    interface_name = str(state.get("interface_name", "")).strip().lower()
    if interface_name == "web" and UPLOAD_PATTERN.search(latest_user_text) and CONVERSION_PATTERN.search(latest_user_text):
        return AgentMatchResult(
            matched=True,
            score=80,
            reasons=("Web conversion request mentioned uploads; route to document conversion for centralized fallback handling.",),
        )

    return AgentMatchResult(matched=False, score=0, reasons=())


def project_task_matcher_factory(project_keywords: tuple[str, ...]):
    normalized_keywords = tuple(keyword.casefold() for keyword in project_keywords if keyword)

    def matcher(_state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
        normalized = normalize_text(latest_user_text)
        hits = [keyword for keyword in normalized_keywords if keyword and keyword in normalized]
        pattern_hits = [pattern for pattern in PROJECT_PATTERNS if pattern in normalized]
        total_hits = list(dict.fromkeys(hits + pattern_hits))
        if not total_hits:
            return AgentMatchResult(matched=False, score=0, reasons=())
        return AgentMatchResult(
            matched=True,
            score=70 + len(total_hits),
            reasons=(f"Project/task signals matched: {', '.join(total_hits[:5])}.",),
        )

    return matcher


def knowledge_matcher(_state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    normalized = normalize_text(latest_user_text)
    hits = [keyword for keyword in KNOWLEDGE_KEYWORDS if keyword in normalized]
    if not hits:
        return AgentMatchResult(matched=False, score=0, reasons=())
    return AgentMatchResult(
        matched=True,
        score=60 + len(hits),
        reasons=(f"Knowledge/documentation signals matched: {', '.join(hits[:5])}.",),
    )


def knowledge_base_builder_matcher(state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    pending_mutation_payload = get_latest_knowledge_mutation_payload(state)
    if pending_mutation_payload and pending_mutation_payload.get("requires_confirmation") is True and not is_casual_greeting(latest_user_text):
        return AgentMatchResult(
            matched=True,
            score=92,
            reasons=("Pending knowledge-base file confirmation should continue in knowledge_base_builder_agent.",),
        )

    normalized = normalize_text(latest_user_text)
    reasons: list[str] = []
    score = 0

    file_write_hits = [phrase for phrase in KB_BUILDER_FILE_WRITE_PHRASES if phrase in normalized]
    if file_write_hits:
        score = max(score, 89 + min(len(file_write_hits), 5))
        reasons.append(f"Knowledge-base file write signals matched: {', '.join(file_write_hits[:4])}.")

    if (
        "知识库" in normalized
        and any(keyword in normalized for keyword in ("写入", "更新", "修改", "创建", "保存", "删除"))
    ):
        score = max(score, 90)
        reasons.append("Knowledge-base update intent matched via `知识库` + mutation verb.")

    if "文件" in normalized and any(keyword in normalized for keyword in ("写入", "写", "修改", "创建", "保存", "删除")):
        score = max(score, 88)
        reasons.append("File write intent matched via `文件` + mutation verb.")

    elicitation_hits = [phrase for phrase in KB_BUILDER_ELICITATION_PHRASES if phrase in normalized]
    if elicitation_hits:
        score = max(score, 90 + min(len(elicitation_hits), 5))
        reasons.append(f"Knowledge elicitation signals matched: {', '.join(elicitation_hits[:4])}.")

    review_hits = [phrase for phrase in KB_BUILDER_REVIEW_PHRASES if phrase in normalized]
    if review_hits:
        score = max(score, 88 + min(len(review_hits), 5))
        reasons.append(f"KB review signals matched: {', '.join(review_hits[:4])}.")

    tracking_hits = [phrase for phrase in KB_BUILDER_TRACKING_PHRASES if phrase in normalized]
    if tracking_hits:
        score = max(score, 86 + min(len(tracking_hits), 5))
        reasons.append(f"KB execution tracking signals matched: {', '.join(tracking_hits[:4])}.")

    layer_hits = [phrase for phrase in KB_BUILDER_LAYER_PHRASES if phrase in normalized]
    decision_hits = [phrase for phrase in KB_BUILDER_LAYER_DECISION_PHRASES if phrase in normalized]
    if layer_hits and decision_hits:
        score = max(score, 84 + min(len(layer_hits) + len(decision_hits), 5))
        reasons.append(f"Knowledge layer placement signals matched: {', '.join((layer_hits + decision_hits)[:5])}.")

    if not reasons:
        return AgentMatchResult(matched=False, score=0, reasons=())
    return AgentMatchResult(matched=True, score=score, reasons=tuple(reasons))


def get_latest_knowledge_mutation_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(state.get("messages") or []):
        if not isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = normalize_tool_result_envelope(
                json.loads(content),
                tool_name="knowledge_base",
            )
        except Exception:
            continue
        inner_payload = payload.get("payload") if isinstance(payload, dict) else None
        if isinstance(inner_payload, dict) and str(inner_payload.get("knowledge_mutation", "")).strip():
            return inner_payload
    return None


def general_chat_matcher(_state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    if is_casual_greeting(latest_user_text):
        return AgentMatchResult(matched=True, score=40, reasons=("Casual greeting matched GeneralAssistant.",))
    return AgentMatchResult(matched=False, score=0, reasons=())
