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

DEFAULT_KNOWLEDGE_BASE_DIR = "knowledge"
DEFAULT_KNOWLEDGE_FILE_TYPES = (
    ".md",
    ".txt",
    ".rst",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xlsm",
)
DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH = "knowledge/AI/Rules/google_sheets_catalog.json"
DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS = 120
DEFAULT_CONVERSION_WORK_DIR = "runtime/conversion"
DEFAULT_JADE_PROJECT_SKILLS_DIR = ".jade/skills"
DEFAULT_TAX_MONITOR_URL = "https://ghv2.rydgames.com:62933/Page/index.html"
DEFAULT_TAX_MONITOR_STATE_PATH = "runtime/monitoring/tax_monitor_state.json"
DEFAULT_TAX_MONITOR_NAVIGATION_PATH = ("税收调控管理", "税收详情（新）")


@dataclass(frozen=True)
class Settings:
    slack_enabled: bool
    slack_bot_token: str
    slack_app_token: str
    web_enabled: bool
    web_host: str
    web_port: int
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
    jade_project_skills_dir: str = DEFAULT_JADE_PROJECT_SKILLS_DIR
    knowledge_google_sheets_catalog_path: str = DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH
    knowledge_google_sheets_cache_ttl_seconds: int = DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS
    conversion_work_dir: str = DEFAULT_CONVERSION_WORK_DIR
    gemini_http_trust_env: bool = False
    tax_monitor_enabled: bool = False
    tax_monitor_url: str = DEFAULT_TAX_MONITOR_URL
    tax_monitor_username: str = ""
    tax_monitor_password: str = ""
    tax_monitor_token: str = ""
    tax_monitor_capture_group: str = ""
    tax_monitor_slack_channel: str = ""
    tax_monitor_poll_interval_seconds: int = 300
    tax_monitor_alert_cooldown_seconds: int = 7200
    tax_monitor_error_cooldown_seconds: int = 1800
    tax_monitor_state_path: str = DEFAULT_TAX_MONITOR_STATE_PATH
    tax_monitor_browser_timeout_seconds: int = 45
    tax_monitor_headless: bool = True
    tax_monitor_navigation_path: tuple[str, ...] = DEFAULT_TAX_MONITOR_NAVIGATION_PATH
    tax_monitor_username_selector: str = ""
    tax_monitor_password_selector: str = ""
    tax_monitor_token_selector: str = ""
    tax_monitor_login_button_selector: str = ""
    tax_monitor_capture_group_selector: str = ""
    tax_monitor_query_button_selector: str = ""


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
    knowledge_file_types_value = os.getenv("KNOWLEDGE_FILE_TYPES", "")
    knowledge_file_types = tuple(
        normalize_knowledge_file_type(item)
        for item in knowledge_file_types_value.split(",")
        if item.strip()
    ) or DEFAULT_KNOWLEDGE_FILE_TYPES
    tax_navigation_path_value = os.getenv("TAX_MONITOR_NAVIGATION_PATH", "")
    tax_navigation_path = tuple(
        item.strip()
        for item in tax_navigation_path_value.split(",")
        if item.strip()
    ) or DEFAULT_TAX_MONITOR_NAVIGATION_PATH

    _cached_settings = Settings(
        slack_enabled=parse_bool_env("SLACK_ENABLED", True),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
        web_enabled=parse_bool_env("WEB_ENABLED", False),
        web_host=os.getenv("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        web_port=int(os.getenv("WEB_PORT", "8000")),
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
        jade_project_skills_dir=(
            os.getenv("JADE_PROJECT_SKILLS_DIR", DEFAULT_JADE_PROJECT_SKILLS_DIR).strip()
            or DEFAULT_JADE_PROJECT_SKILLS_DIR
        ),
        knowledge_google_sheets_catalog_path=(
            os.getenv(
                "KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH",
                DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH,
            ).strip()
            or DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH
        ),
        knowledge_google_sheets_cache_ttl_seconds=int(
            os.getenv(
                "KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS",
                str(DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS),
            )
        ),
        conversion_work_dir=(
            os.getenv("CONVERSION_WORK_DIR", DEFAULT_CONVERSION_WORK_DIR).strip()
            or DEFAULT_CONVERSION_WORK_DIR
        ),
        gemini_http_trust_env=parse_bool_env("GEMINI_HTTP_TRUST_ENV", False),
        tax_monitor_enabled=parse_bool_env("TAX_MONITOR_ENABLED", False),
        tax_monitor_url=os.getenv("TAX_MONITOR_URL", DEFAULT_TAX_MONITOR_URL).strip() or DEFAULT_TAX_MONITOR_URL,
        tax_monitor_username=os.getenv("TAX_MONITOR_USERNAME", "").strip(),
        tax_monitor_password=os.getenv("TAX_MONITOR_PASSWORD", ""),
        tax_monitor_token=os.getenv("TAX_MONITOR_TOKEN", ""),
        tax_monitor_capture_group=os.getenv("TAX_MONITOR_CAPTURE_GROUP", "").strip(),
        tax_monitor_slack_channel=os.getenv("TAX_MONITOR_SLACK_CHANNEL", "").strip(),
        tax_monitor_poll_interval_seconds=int(os.getenv("TAX_MONITOR_POLL_INTERVAL_SECONDS", "300")),
        tax_monitor_alert_cooldown_seconds=int(os.getenv("TAX_MONITOR_ALERT_COOLDOWN_SECONDS", "7200")),
        tax_monitor_error_cooldown_seconds=int(os.getenv("TAX_MONITOR_ERROR_COOLDOWN_SECONDS", "1800")),
        tax_monitor_state_path=(
            os.getenv("TAX_MONITOR_STATE_PATH", DEFAULT_TAX_MONITOR_STATE_PATH).strip()
            or DEFAULT_TAX_MONITOR_STATE_PATH
        ),
        tax_monitor_browser_timeout_seconds=int(os.getenv("TAX_MONITOR_BROWSER_TIMEOUT_SECONDS", "45")),
        tax_monitor_headless=parse_bool_env("TAX_MONITOR_HEADLESS", True),
        tax_monitor_navigation_path=tax_navigation_path,
        tax_monitor_username_selector=os.getenv("TAX_MONITOR_USERNAME_SELECTOR", "").strip(),
        tax_monitor_password_selector=os.getenv("TAX_MONITOR_PASSWORD_SELECTOR", "").strip(),
        tax_monitor_token_selector=os.getenv("TAX_MONITOR_TOKEN_SELECTOR", "").strip(),
        tax_monitor_login_button_selector=os.getenv("TAX_MONITOR_LOGIN_BUTTON_SELECTOR", "").strip(),
        tax_monitor_capture_group_selector=os.getenv("TAX_MONITOR_CAPTURE_GROUP_SELECTOR", "").strip(),
        tax_monitor_query_button_selector=os.getenv("TAX_MONITOR_QUERY_BUTTON_SELECTOR", "").strip(),
    )
    return _cached_settings


def validate_bootstrap_settings(settings: Settings) -> None:
    validate_core_settings(settings)
    validate_interface_settings(settings)


def validate_core_settings(settings: Settings) -> None:
    if settings.tax_monitor_enabled:
        validate_tax_monitor_settings(settings)

    if not is_agent_runtime_enabled(settings):
        return

    missing: list[str] = []
    if not settings.google_api_key:
        missing.append("GOOGLE_API_KEY")
    if not settings.google_application_credentials:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if not settings.jade_project_sheet_id:
        missing.append("JADE_PROJECT_SHEET_ID")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")


def validate_interface_settings(settings: Settings) -> None:
    if not (is_agent_runtime_enabled(settings) or settings.tax_monitor_enabled):
        raise RuntimeError("No communication interface is configured.")

    if settings.slack_enabled:
        if settings.slack_bot_token and not settings.slack_app_token:
            raise RuntimeError("Missing required environment variables: SLACK_APP_TOKEN")
        if settings.slack_app_token and not settings.slack_bot_token:
            raise RuntimeError("Missing required environment variables: SLACK_BOT_TOKEN")

    if settings.web_enabled and settings.web_port <= 0:
        raise RuntimeError("WEB_PORT must be a positive integer.")


def is_slack_enabled(settings: Settings) -> bool:
    return bool(settings.slack_enabled and settings.slack_bot_token and settings.slack_app_token)


def is_agent_runtime_enabled(settings: Settings) -> bool:
    return bool(is_slack_enabled(settings) or settings.web_enabled)


def validate_tax_monitor_settings(settings: Settings) -> None:
    missing: list[str] = []
    if not settings.tax_monitor_url:
        missing.append("TAX_MONITOR_URL")
    if not settings.tax_monitor_username:
        missing.append("TAX_MONITOR_USERNAME")
    if not settings.tax_monitor_password:
        missing.append("TAX_MONITOR_PASSWORD")
    if not settings.slack_bot_token:
        missing.append("SLACK_BOT_TOKEN")
    if not settings.tax_monitor_slack_channel:
        missing.append("TAX_MONITOR_SLACK_CHANNEL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def normalize_knowledge_file_type(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


def parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
