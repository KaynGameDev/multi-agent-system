from __future__ import annotations

from typing import Any

from app.contracts import AssistantRequestDomain

DOMAIN_AGENT_MAP: dict[AssistantRequestDomain, str] = {
    "general": "general_chat_agent",
    "knowledge": "knowledge_agent",
    "project_task": "project_task_agent",
    "knowledge_base_builder": "knowledge_base_builder_agent",
    "document_conversion": "document_conversion_agent",
}


def resolve_agent_for_domain(
    domain: str,
    *,
    registrations_by_name: dict[str, Any],
    default_agent: str = "general_chat_agent",
) -> tuple[str, bool, str]:
    normalized_domain = str(domain or "").strip().lower()
    if not normalized_domain:
        return _fallback(default_agent, registrations_by_name, "AssistantRequest did not include a likely_domain.")

    mapped_agent = DOMAIN_AGENT_MAP.get(normalized_domain)
    if not mapped_agent:
        return _fallback(
            default_agent,
            registrations_by_name,
            f"AssistantRequest returned unsupported domain `{normalized_domain}`.",
        )

    if mapped_agent not in registrations_by_name:
        return _fallback(
            default_agent,
            registrations_by_name,
            f"AssistantRequest mapped `{normalized_domain}` to unavailable agent `{mapped_agent}`.",
        )

    return mapped_agent, False, ""


def _fallback(
    default_agent: str,
    registrations_by_name: dict[str, Any],
    reason: str,
) -> tuple[str, bool, str]:
    selected_agent = default_agent if default_agent in registrations_by_name else next(iter(registrations_by_name), default_agent)
    return selected_agent, True, reason
