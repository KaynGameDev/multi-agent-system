from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_PROJECT_KEYWORDS = (
    "project",
    "task",
    "tasks",
    "status",
    "deadline",
    "milestone",
    "roadmap",
    "sheet",
    "tracker",
    "assignee",
    "owner",
    "who is doing",
    "feature",
    "sprint",
)

DEFAULT_KNOWLEDGE_BASE_DIR = "data/knowledge"
DEFAULT_KNOWLEDGE_FILE_TYPES = (
    ".md",
    ".txt",
    ".rst",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xlsm",
)


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str
    slack_app_token: str
    telegram_bot_token: str
    telegram_allowed_chat_ids: tuple[str, ...]
    google_api_key: str
    gemini_model: str
    gemini_temperature: float
    google_application_credentials: str
    jade_project_sheet_id: str
    project_sheet_range: str
    project_sheet_cache_ttl_seconds: int
    slack_thinking_reaction: str
    project_lookup_keywords: tuple[str, ...]
    knowledge_base_dir: str
    knowledge_file_types: tuple[str, ...]


_cached_settings: Settings | None = None


def load_settings(force_reload: bool = False) -> Settings:
    global _cached_settings

    if _cached_settings is not None and not force_reload:
        return _cached_settings

    keywords_value = os.getenv("PROJECT_LOOKUP_KEYWORDS", "")
    keywords = tuple(
        item.strip().lower()
        for item in keywords_value.split(",")
        if item.strip()
    ) or DEFAULT_PROJECT_KEYWORDS
    telegram_allowed_chat_ids_value = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    telegram_allowed_chat_ids = tuple(
        item.strip()
        for item in telegram_allowed_chat_ids_value.split(",")
        if item.strip()
    )
    knowledge_file_types_value = os.getenv("KNOWLEDGE_FILE_TYPES", "")
    knowledge_file_types = tuple(
        normalize_knowledge_file_type(item)
        for item in knowledge_file_types_value.split(",")
        if item.strip()
    ) or DEFAULT_KNOWLEDGE_FILE_TYPES

    _cached_settings = Settings(
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_allowed_chat_ids=telegram_allowed_chat_ids,
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        jade_project_sheet_id=os.getenv("JADE_PROJECT_SHEET_ID", ""),
        project_sheet_range=os.getenv("PROJECT_SHEET_RANGE", "Tasks!A1:Z"),
        project_sheet_cache_ttl_seconds=int(os.getenv("PROJECT_SHEET_CACHE_TTL_SECONDS", "30")),
        slack_thinking_reaction=os.getenv("SLACK_THINKING_REACTION", "eyes"),
        project_lookup_keywords=keywords,
        knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", DEFAULT_KNOWLEDGE_BASE_DIR).strip() or DEFAULT_KNOWLEDGE_BASE_DIR,
        knowledge_file_types=knowledge_file_types,
    )
    return _cached_settings


def validate_bootstrap_settings(settings: Settings) -> None:
    missing: list[str] = []
    slack_configured = bool(settings.slack_bot_token and settings.slack_app_token)
    partial_slack_config = bool(settings.slack_bot_token or settings.slack_app_token) and not slack_configured

    if partial_slack_config and not settings.slack_bot_token:
        missing.append("SLACK_BOT_TOKEN")
    if partial_slack_config and not settings.slack_app_token:
        missing.append("SLACK_APP_TOKEN")
    if not settings.google_api_key:
        missing.append("GOOGLE_API_KEY")
    if not settings.google_application_credentials:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if not settings.jade_project_sheet_id:
        missing.append("JADE_PROJECT_SHEET_ID")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")

    if not slack_configured and not settings.telegram_bot_token:
        raise RuntimeError(
            "At least one interface must be configured: Slack "
            "(SLACK_BOT_TOKEN + SLACK_APP_TOKEN) or Telegram (TELEGRAM_BOT_TOKEN)."
        )


def normalize_knowledge_file_type(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned
