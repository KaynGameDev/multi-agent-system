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


@dataclass(frozen=True)
class Settings:
    slack_bot_token: str
    slack_app_token: str
    google_api_key: str
    gemini_model: str
    gemini_temperature: float
    google_application_credentials: str
    jade_project_sheet_id: str
    project_sheet_range: str
    project_sheet_cache_ttl_seconds: int
    slack_thinking_reaction: str
    project_lookup_keywords: tuple[str, ...]


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

    _cached_settings = Settings(
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        jade_project_sheet_id=os.getenv("JADE_PROJECT_SHEET_ID", ""),
        project_sheet_range=os.getenv("PROJECT_SHEET_RANGE", "Tasks!A1:Z"),
        project_sheet_cache_ttl_seconds=int(os.getenv("PROJECT_SHEET_CACHE_TTL_SECONDS", "30")),
        slack_thinking_reaction=os.getenv("SLACK_THINKING_REACTION", "eyes"),
        project_lookup_keywords=keywords,
    )
    return _cached_settings


def validate_bootstrap_settings(settings: Settings) -> None:
    missing: list[str] = []

    if not settings.slack_bot_token:
        missing.append("SLACK_BOT_TOKEN")
    if not settings.slack_app_token:
        missing.append("SLACK_APP_TOKEN")
    if not settings.google_api_key:
        missing.append("GOOGLE_API_KEY")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")
