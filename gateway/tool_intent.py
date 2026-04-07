from __future__ import annotations

import re
from typing import Any

from app.agent_registry import AgentRegistration
from app.tool_registry import ToolMetadata, get_tool_metadata
from gateway.matchers import evaluate_agent_match
from gateway.text_utils import normalize_text

TOOL_AVAILABILITY_PATTERNS = (
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bdo you have\b",
    r"\bdo you support\b",
    r"\bare you able to\b",
    "你能",
    "你可以",
    "能不能",
    "可以吗",
    "能否",
    "会不会",
    "支持",
)
TOOL_ACTION_PATTERNS = (
    "help me",
    "please",
    "save",
    "write",
    "create",
    "edit",
    "update",
    "read",
    "open",
    "list",
    "search",
    "find",
    "show",
    "帮我",
    "请",
    "保存",
    "写",
    "创建",
    "修改",
    "更新",
    "读取",
    "读",
    "打开",
    "列出",
    "搜索",
    "查",
    "找",
    "展示",
)
TOOL_ACTION_TARGET_HINTS = (
    "discussion",
    "chat",
    "summary",
    "content",
    "note",
    "notes",
    "document",
    "doc",
    "draft",
    "task",
    "tasks",
    "kb",
    "knowledge base",
    "讨论",
    "聊天",
    "对话",
    "总结",
    "内容",
    "文档",
    "草稿",
    "任务",
    "知识库",
)


def select_agent_from_tool_intent(
    agent_registrations: tuple[AgentRegistration, ...],
    state: dict[str, Any],
    latest_user_text: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    normalized = normalize_text(latest_user_text)
    if not normalized:
        return None

    candidate_rows: list[tuple[AgentRegistration, int, Any, list[dict[str, Any]]]] = []
    matched_tool_ids: list[str] = []
    for registration in agent_registrations:
        if not registration.tool_ids:
            continue
        tool_matches = collect_tool_matches(registration.tool_ids, normalized)
        tool_score = sum(int(item["score"]) for item in tool_matches)
        if tool_score <= 0:
            continue
        matcher_result = evaluate_agent_match(registration, state, latest_user_text)
        matched_tool_ids.extend(str(item["tool_id"]) for item in tool_matches)
        candidate_rows.append((registration, tool_score, matcher_result, tool_matches))

    if not candidate_rows:
        return None

    intent_kind = classify_tool_intent(normalized)
    if intent_kind == "plain_query":
        return None

    diagnostics: list[dict[str, Any]] = [
        {
            "kind": "tool_intent_detected",
            "policy_step": "tool_intent",
            "intent_kind": intent_kind,
            "matched_tool_ids": list(dict.fromkeys(matched_tool_ids)),
            "reason": f"Detected `{intent_kind}` from tool metadata overlap.",
        }
    ]
    for registration, tool_score, matcher_result, tool_matches in candidate_rows:
        diagnostics.append(
            {
                "kind": "tool_intent_candidate",
                "policy_step": "tool_intent",
                "agent": registration.name,
                "tool_score": tool_score,
                "domain_matched": matcher_result.matched and matcher_result.score > 0,
                "declared_tool_count": len(registration.tool_ids),
                "tool_matches": tool_matches,
            }
        )

    selected_registration, _, selected_matcher_result, selected_tool_matches = sorted(
        candidate_rows,
        key=lambda item: (
            -item[1],
            -int(item[2].matched and item[2].score > 0),
            len(item[0].tool_ids),
            item[0].selection_order,
            item[0].name,
        ),
    )[0]
    matched_tools = ", ".join(str(item["tool_id"]) for item in selected_tool_matches[:3])
    reason_parts = [f"Tool intent `{intent_kind}` matched `{selected_registration.name}` via {matched_tools}."]
    if selected_matcher_result.matched and selected_matcher_result.score > 0:
        reason_parts.append("Existing domain matcher also matched.")
    diagnostics.append(
        {
            "kind": "selected",
            "policy_step": "tool_intent",
            "selected_agent": selected_registration.name,
            "reason": " ".join(reason_parts),
        }
    )
    return selected_registration.name, diagnostics


def collect_tool_matches(tool_ids: tuple[str, ...], normalized_text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for tool_id in tool_ids:
        metadata = get_tool_metadata(tool_id)
        hits = find_tool_phrase_hits(metadata, normalized_text)
        if not hits:
            continue
        matches.append(
            {
                "tool_id": tool_id,
                "score": score_tool_metadata_hits(hits),
                "hits": hits,
            }
        )
    return matches


def find_tool_phrase_hits(metadata: ToolMetadata, normalized_text: str) -> list[str]:
    phrases = []
    for phrase in [*metadata.semantic_aliases, *metadata.examples]:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            phrases.append(normalized_phrase)
    return list(dict.fromkeys(phrases))


def score_tool_metadata_hits(hits: list[str]) -> int:
    return sum(5 + min(len(hit), 20) for hit in hits)


def classify_tool_intent(normalized_text: str) -> str:
    if not normalized_text:
        return "plain_query"

    availability = any(
        (pattern in normalized_text if not pattern.startswith(r"\b") else re.search(pattern, normalized_text))
        for pattern in TOOL_AVAILABILITY_PATTERNS
    )
    action = any(pattern in normalized_text for pattern in TOOL_ACTION_PATTERNS)
    action_target = any(pattern in normalized_text for pattern in TOOL_ACTION_TARGET_HINTS)

    if availability and action_target:
        return "tool_action_request"
    if availability:
        return "tool_availability_question"
    if action:
        return "tool_action_request"
    return "plain_query"
