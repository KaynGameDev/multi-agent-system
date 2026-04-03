from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.agent_registry import AgentRegistration
from app.messages import extract_latest_human_text
from app.skills import SkillDefinition, SkillRegistry, normalize_skill_id
from app.state import AgentState
from tools.conversion_google_sources import extract_google_document_references

logger = logging.getLogger(__name__)

CASUAL_GREETING_NORMALIZED_TEXTS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "早上好",
    "下午好",
    "晚上好",
}
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
GENERAL_ASSISTANT_ALIAS = "GeneralAssistant"


@dataclass(frozen=True)
class AgentMatchResult:
    matched: bool
    score: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)


class GatewayNode:
    def __init__(
        self,
        _llm=None,
        *,
        agent_registrations: tuple[AgentRegistration, ...],
        default_route: str,
        skill_registry: SkillRegistry,
    ) -> None:
        self.agent_registrations = tuple(agent_registrations)
        self.default_route = default_route
        self.skill_registry = skill_registry
        self.registrations_by_name = {registration.name: registration for registration in self.agent_registrations}
        self.general_assistant_name = self._resolve_general_assistant_name()

    def __call__(self, state: AgentState) -> dict[str, Any]:
        latest_user_text = extract_latest_human_text(state)
        context_paths = normalize_context_paths(state)
        requested_skill_ids = normalize_requested_skill_ids(state)
        requested_agent = self._normalize_agent_name(str(state.get("requested_agent") or "").strip())

        warnings: list[str] = []
        skill_diagnostics: list[dict[str, Any]] = list(self.skill_registry.discovery_diagnostics)

        explicit_skill_definitions: list[SkillDefinition] = []
        for skill_id in requested_skill_ids:
            resolution = self.skill_registry.resolve_skill(skill_id, context_paths=context_paths)
            skill_diagnostics.extend(resolution.diagnostics)
            if resolution.effective_definition is None:
                warnings.append(f"Requested skill `{skill_id}` could not be resolved from the shared registry.")
                continue
            explicit_skill_definitions.append(resolution.effective_definition)
            skill_diagnostics.append(
                {
                    "kind": "explicit_request",
                    "skill_id": resolution.effective_definition.skill_id,
                    "selected": True,
                    "reason": "Skill was explicitly requested in state.",
                }
            )

        selected_agent, agent_diagnostics, warnings = self._select_agent(
            state,
            latest_user_text,
            requested_agent=requested_agent,
            explicit_skill_definitions=tuple(explicit_skill_definitions),
            warnings=warnings,
        )

        resolved_skill_ids, selected_skill_diagnostics, warnings = self._select_skills_for_agent(
            selected_agent,
            latest_user_text,
            context_paths=context_paths,
            explicit_skill_definitions=tuple(explicit_skill_definitions),
            warnings=warnings,
        )
        skill_diagnostics.extend(selected_skill_diagnostics)

        route_reason = build_route_reason(agent_diagnostics, selected_agent)
        logger.info(
            "Gateway selected route=%s requested_agent=%s requested_skills=%s resolved_skills=%s warnings=%s",
            selected_agent,
            requested_agent,
            requested_skill_ids,
            resolved_skill_ids,
            warnings,
        )

        return {
            "route": selected_agent,
            "route_reason": route_reason,
            "requested_skill_ids": list(requested_skill_ids),
            "resolved_skill_ids": list(resolved_skill_ids),
            "context_paths": list(context_paths),
            "skill_resolution_diagnostics": skill_diagnostics,
            "agent_selection_diagnostics": agent_diagnostics,
            "selection_warnings": warnings,
        }

    def _select_agent(
        self,
        state: AgentState,
        latest_user_text: str,
        *,
        requested_agent: str,
        explicit_skill_definitions: tuple[SkillDefinition, ...],
        warnings: list[str],
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        diagnostics: list[dict[str, Any]] = []

        if requested_agent:
            if requested_agent in self.registrations_by_name:
                diagnostics.append(
                    {
                        "kind": "requested_agent",
                        "selected_agent": requested_agent,
                        "reason": f"Explicit requested agent `{requested_agent}` was honored.",
                    }
                )
                return requested_agent, diagnostics, warnings
            warnings.append(f"Requested agent `{requested_agent}` is not active; falling back to gateway policy.")

        explicit_delegate_agents = [
            self._normalize_agent_name(definition.delegate_agent)
            for definition in explicit_skill_definitions
            if definition.execution_mode == "forked" and definition.delegate_agent
        ]
        explicit_delegate_agents = [name for name in explicit_delegate_agents if name in self.registrations_by_name]
        if explicit_delegate_agents:
            selected = self._choose_from_candidates(
                tuple(dict.fromkeys(explicit_delegate_agents)),
                state,
                latest_user_text,
                diagnostics,
                reason_prefix="Selected from explicit forked skill delegates",
            )
            return selected, diagnostics, warnings

        if any(definition.execution_mode == "forked" for definition in explicit_skill_definitions):
            selected, warnings = self._fallback_general_assistant(
                warnings,
                "Forked skill omitted `delegate_agent`; routed to GeneralAssistant.",
            )
            diagnostics.append(
                {
                    "kind": "forked_skill_fallback",
                    "selected_agent": selected,
                    "reason": "Forked skill omitted `delegate_agent`; used GeneralAssistant fallback.",
                }
            )
            return selected, diagnostics, warnings

        explicit_inline_candidates: list[str] = []
        for definition in explicit_skill_definitions:
            if definition.execution_mode != "inline":
                continue
            explicit_inline_candidates.extend(
                self._normalize_agent_name(agent_name)
                for agent_name in definition.available_to_agents
                if self._normalize_agent_name(agent_name) in self.registrations_by_name
            )
        explicit_inline_candidates = list(dict.fromkeys(explicit_inline_candidates))
        if explicit_inline_candidates:
            selected = self._choose_from_candidates(
                tuple(explicit_inline_candidates),
                state,
                latest_user_text,
                diagnostics,
                reason_prefix="Selected from explicit inline skill compatibility",
            )
            return selected, diagnostics, warnings

        candidate_results: list[tuple[AgentRegistration, AgentMatchResult]] = []
        for registration in self.agent_registrations:
            result = evaluate_agent_match(registration, state, latest_user_text)
            candidate_results.append((registration, result))
            diagnostics.append(
                {
                    "kind": "matcher_result",
                    "agent": registration.name,
                    "matched": result.matched,
                    "score": result.score,
                    "reasons": list(result.reasons),
                }
            )

        matched_candidates = [
            (registration, result)
            for registration, result in candidate_results
            if result.matched and result.score > 0
        ]
        if matched_candidates:
            selected_registration, selected_result = sorted(
                matched_candidates,
                key=lambda item: (
                    -item[1].score,
                    item[0].selection_order,
                    item[0].name,
                ),
            )[0]
            diagnostics.append(
                {
                    "kind": "selected",
                    "selected_agent": selected_registration.name,
                    "reason": "; ".join(selected_result.reasons)
                    or f"Selected `{selected_registration.name}` from deterministic matcher policy.",
                }
            )
            return selected_registration.name, diagnostics, warnings

        selected, warnings = self._fallback_general_assistant(
            warnings,
            "No specialist matcher applied; routed to GeneralAssistant.",
        )
        diagnostics.append(
            {
                "kind": "fallback",
                "selected_agent": selected,
                "reason": "No specialist matcher applied; used GeneralAssistant fallback.",
            }
        )
        return selected, diagnostics, warnings

    def _select_skills_for_agent(
        self,
        agent_name: str,
        latest_user_text: str,
        *,
        context_paths: tuple[str, ...],
        explicit_skill_definitions: tuple[SkillDefinition, ...],
        warnings: list[str],
    ) -> tuple[tuple[str, ...], list[dict[str, Any]], list[str]]:
        diagnostics: list[dict[str, Any]] = []
        resolved_skill_ids: list[str] = []
        selected_ids: set[str] = set()

        for definition in explicit_skill_definitions:
            eligible = is_skill_eligible_for_agent(
                definition,
                target_agent=agent_name,
                general_assistant_name=self.general_assistant_name,
            )
            diagnostics.append(
                {
                    "kind": "explicit_skill_selection",
                    "skill_id": definition.skill_id,
                    "selected": eligible,
                    "reason": (
                        f"Explicit skill applies to `{agent_name}`."
                        if eligible
                        else f"Explicit skill does not apply to `{agent_name}`."
                    ),
                }
            )
            if not eligible:
                warnings.append(
                    f"Explicit skill `{definition.skill_id}` was skipped because it does not apply to `{agent_name}`."
                )
                continue
            if definition.skill_id in selected_ids:
                continue
            resolved_skill_ids.append(definition.skill_id)
            selected_ids.add(definition.skill_id)

        scored_auto_matches: list[tuple[int, SkillDefinition, dict[str, Any]]] = []
        for skill_id in self.skill_registry.list_skill_ids():
            resolution = self.skill_registry.resolve_skill(skill_id, context_paths=context_paths)
            diagnostics.extend(resolution.diagnostics)
            definition = resolution.effective_definition
            if definition is None:
                continue
            if definition.skill_id in selected_ids:
                continue
            if definition.execution_mode != "inline":
                continue
            if not is_skill_eligible_for_agent(
                definition,
                target_agent=agent_name,
                general_assistant_name=self.general_assistant_name,
            ):
                diagnostics.append(
                    {
                        "kind": "auto_skill_ineligible",
                        "skill_id": definition.skill_id,
                        "selected": False,
                        "reason": f"Effective skill does not apply to `{agent_name}`.",
                    }
                )
                continue
            score, reason = score_skill_match(definition, latest_user_text)
            diagnostics.append(
                {
                    "kind": "auto_skill_match",
                    "skill_id": definition.skill_id,
                    "selected": score > 0,
                    "score": score,
                    "reason": reason,
                }
            )
            if score <= 0:
                continue
            scored_auto_matches.append((score, definition, {"reason": reason}))

        for _, definition, _metadata in sorted(
            scored_auto_matches,
            key=lambda item: (-item[0], item[1].skill_id),
        )[:3]:
            resolved_skill_ids.append(definition.skill_id)
            selected_ids.add(definition.skill_id)

        return tuple(resolved_skill_ids), diagnostics, warnings

    def _resolve_general_assistant_name(self) -> str:
        for registration in self.agent_registrations:
            if registration.is_general_assistant:
                return registration.name
        if "general_chat_agent" in self.registrations_by_name:
            return "general_chat_agent"
        return ""

    def _choose_from_candidates(
        self,
        candidate_agent_names: tuple[str, ...],
        state: AgentState,
        latest_user_text: str,
        diagnostics: list[dict[str, Any]],
        *,
        reason_prefix: str,
    ) -> str:
        candidates = [self.registrations_by_name[name] for name in candidate_agent_names if name in self.registrations_by_name]
        if not candidates:
            selected, _warnings = self._fallback_general_assistant([], f"{reason_prefix}; falling back to GeneralAssistant.")
            diagnostics.append(
                {
                    "kind": "candidate_fallback",
                    "selected_agent": selected,
                    "reason": f"{reason_prefix}; no valid candidate remained.",
                }
            )
            return selected

        scored_candidates = []
        for registration in candidates:
            result = evaluate_agent_match(registration, state, latest_user_text)
            diagnostics.append(
                {
                    "kind": "candidate_matcher_result",
                    "agent": registration.name,
                    "matched": result.matched,
                    "score": result.score,
                    "reasons": list(result.reasons),
                }
            )
            scored_candidates.append((registration, result))

        positively_matched = [item for item in scored_candidates if item[1].matched and item[1].score > 0]
        ordered_candidates = positively_matched or scored_candidates
        selected_registration, selected_result = sorted(
            ordered_candidates,
            key=lambda item: (
                -item[1].score,
                item[0].selection_order,
                item[0].name,
            ),
        )[0]
        diagnostics.append(
            {
                "kind": "candidate_selected",
                "selected_agent": selected_registration.name,
                "reason": "; ".join(selected_result.reasons) or reason_prefix,
            }
        )
        return selected_registration.name

    def _fallback_general_assistant(
        self,
        warnings: list[str],
        warning_text: str,
    ) -> tuple[str, list[str]]:
        if self.general_assistant_name and self.general_assistant_name in self.registrations_by_name:
            return self.general_assistant_name, warnings

        ordered_registrations = sorted(
            self.agent_registrations,
            key=lambda item: (item.selection_order, item.name),
        )
        if not ordered_registrations:
            return self.default_route, warnings

        warnings = list(warnings)
        warnings.append(f"{warning_text} GeneralAssistant is unavailable; used `{ordered_registrations[0].name}` instead.")
        return ordered_registrations[0].name, warnings

    def _normalize_agent_name(self, raw_value: str) -> str:
        if not raw_value:
            return ""
        if raw_value == GENERAL_ASSISTANT_ALIAS:
            return self.general_assistant_name or raw_value
        return raw_value


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


def knowledge_base_builder_matcher(_state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    normalized = normalize_text(latest_user_text)
    reasons: list[str] = []
    score = 0

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


def general_chat_matcher(_state: dict[str, Any], latest_user_text: str) -> AgentMatchResult:
    if is_casual_greeting(latest_user_text):
        return AgentMatchResult(matched=True, score=40, reasons=("Casual greeting matched GeneralAssistant.",))
    return AgentMatchResult(matched=False, score=0, reasons=())


def is_skill_eligible_for_agent(
    definition: SkillDefinition,
    *,
    target_agent: str,
    general_assistant_name: str,
) -> bool:
    if definition.execution_mode == "inline":
        return target_agent in definition.available_to_agents
    if definition.delegate_agent:
        return target_agent == definition.delegate_agent
    return target_agent == general_assistant_name


def score_skill_match(definition: SkillDefinition, latest_user_text: str) -> tuple[int, str]:
    normalized_text = normalize_text(latest_user_text)
    if not normalized_text:
        return 0, "No user text available for deterministic skill matching."

    text_tokens = tokenize_text(normalized_text)
    skill_tokens = tokenize_text(
        normalize_text(" ".join([definition.skill_id, definition.name, definition.description]))
    )
    overlaps = sorted(text_tokens & skill_tokens)
    if len(overlaps) >= 2:
        return 20 + len(overlaps), f"Token overlap matched: {', '.join(overlaps[:5])}."

    normalized_skill_id = definition.skill_id.replace("-", " ")
    if normalized_skill_id and normalized_skill_id in normalized_text:
        return 15, f"Skill id phrase `{normalized_skill_id}` appeared in the request."

    return 0, "No deterministic metadata overlap matched."


def build_route_reason(agent_diagnostics: list[dict[str, Any]], route: str) -> str:
    for item in reversed(agent_diagnostics):
        reason = str(item.get("reason", "")).strip()
        if reason:
            return reason
    return f"Selected `{route}` from deterministic gateway policy."


def normalize_requested_skill_ids(state: dict[str, Any]) -> tuple[str, ...]:
    raw_value = state.get("requested_skill_ids") or []
    if isinstance(raw_value, str):
        raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        raw_items = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        raw_items = []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        skill_id = normalize_skill_id(item)
        if not skill_id or skill_id in seen:
            continue
        normalized.append(skill_id)
        seen.add(skill_id)
    return tuple(normalized)


def normalize_context_paths(state: dict[str, Any]) -> tuple[str, ...]:
    raw_value = state.get("context_paths") or []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        values = []
    return tuple(values)


def tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.casefold())
        if token not in {"the", "and", "for", "with", "that", "this", "what", "when", "where"}
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def is_casual_greeting(text: str) -> bool:
    normalized = re.sub(r"[!?,.，。！？\s]+", " ", (text or "").strip().lower()).strip()
    if not normalized:
        return False
    return normalized in CASUAL_GREETING_NORMALIZED_TEXTS
