from __future__ import annotations

import re
from typing import Any, cast

from typing_extensions import TypedDict


DEFAULT_ACCEPT_REPLIES = ("ok", "yes", "approve", "confirm", "批准", "确认")
DEFAULT_CANCEL_REPLIES = ("cancel", "stop", "never mind", "取消", "不用了", "算了")


class PendingInteractionOption(TypedDict, total=False):
    label: str
    aliases: list[str]
    value: str
    payload: dict[str, Any]


class PendingInteraction(TypedDict, total=False):
    kind: str
    owner_agent: str
    source_tool_id: str
    status: str
    prompt_context: str
    options: list[PendingInteractionOption]
    accepted_replies: list[str]
    cancel_replies: list[str]
    payload: dict[str, Any]


def build_confirmation_interaction(
    *,
    owner_agent: str,
    source_tool_id: str,
    prompt_context: str,
    payload: dict[str, Any] | None = None,
    accepted_replies: tuple[str, ...] | list[str] = DEFAULT_ACCEPT_REPLIES,
    cancel_replies: tuple[str, ...] | list[str] = DEFAULT_CANCEL_REPLIES,
) -> PendingInteraction:
    return PendingInteraction(
        kind="confirmation",
        owner_agent=owner_agent,
        source_tool_id=source_tool_id,
        status="awaiting_reply",
        prompt_context=prompt_context,
        options=[],
        accepted_replies=list(accepted_replies),
        cancel_replies=list(cancel_replies),
        payload=dict(payload or {}),
    )


def build_selection_interaction(
    *,
    owner_agent: str,
    source_tool_id: str,
    prompt_context: str,
    options: list[PendingInteractionOption],
    payload: dict[str, Any] | None = None,
    cancel_replies: tuple[str, ...] | list[str] = DEFAULT_CANCEL_REPLIES,
) -> PendingInteraction:
    return PendingInteraction(
        kind="selection",
        owner_agent=owner_agent,
        source_tool_id=source_tool_id,
        status="awaiting_reply",
        prompt_context=prompt_context,
        options=list(options),
        accepted_replies=[],
        cancel_replies=list(cancel_replies),
        payload=dict(payload or {}),
    )


def get_pending_interaction(state: dict[str, Any]) -> PendingInteraction | None:
    interaction = state.get("pending_interaction")
    if isinstance(interaction, dict) and str(interaction.get("status", "")).strip():
        return cast(PendingInteraction, interaction)
    return None


def is_pending_interaction_active(interaction: PendingInteraction | None) -> bool:
    return bool(interaction and str(interaction.get("status", "")).strip().lower() == "awaiting_reply")


def match_pending_interaction_reply(interaction: PendingInteraction | None, user_text: str) -> dict[str, Any] | None:
    if not is_pending_interaction_active(interaction):
        return None

    normalized_text = normalize_interaction_text(user_text)
    if not normalized_text:
        return None

    cancel_replies = {
        normalize_interaction_text(reply)
        for reply in interaction.get("cancel_replies", [])
        if normalize_interaction_text(reply)
    }
    if normalized_text in cancel_replies:
        return {"action": "cancel"}

    if interaction.get("kind") == "confirmation":
        accepted_replies = {
            normalize_interaction_text(reply)
            for reply in interaction.get("accepted_replies", [])
            if normalize_interaction_text(reply)
        }
        if normalized_text in accepted_replies:
            return {"action": "accept"}
        return None

    if interaction.get("kind") == "selection":
        option_match = match_selection_option(interaction.get("options", []), normalized_text)
        if option_match is not None:
            return {"action": "select", **option_match}
    return None


def match_selection_option(options: list[PendingInteractionOption], normalized_text: str) -> dict[str, Any] | None:
    selected_index = extract_selection_index(normalized_text)
    if selected_index is not None and 0 <= selected_index < len(options):
        option = options[selected_index]
        return {"option": option, "index": selected_index}

    for index, option in enumerate(options):
        labels = [option.get("label", ""), *option.get("aliases", [])]
        normalized_labels = {normalize_interaction_text(label) for label in labels if normalize_interaction_text(label)}
        if normalized_text in normalized_labels:
            return {"option": option, "index": index}
    return None


def extract_selection_index(normalized_text: str) -> int | None:
    if normalized_text.isdigit():
        return max(int(normalized_text) - 1, 0)

    patterns = (
        r"^第\s*(\d+)\s*(?:个|篇|份|条|项)?$",
        r"^(?:option|doc|document|task)\s*(\d+)$",
        r"^(\d+)[\).]$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized_text)
        if match:
            return max(int(match.group(1)) - 1, 0)
    return None


def normalize_interaction_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().casefold())
    normalized = normalized.strip("`'\"“”‘’.,!?，。！？")
    return normalized
