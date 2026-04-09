from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_LLM_PROVIDER = "google"
SUPPORTED_LLM_PROVIDERS = ("google", "minimax", "openai")
DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7-highspeed"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

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


@dataclass(frozen=True)
class Settings:
    slack_enabled: bool
    slack_bot_token: str
    slack_app_token: str
    web_enabled: bool
    web_host: str
    web_port: int
    llm_provider: str
    llm_model: str
    llm_temperature: float
    google_api_key: str
    minimax_api_key: str
    minimax_base_url: str
    openai_api_key: str
    openai_base_url: str
    pending_action_parser_model: str
    pending_action_parser_temperature: float
    pending_action_parser_confidence_threshold: float
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
    langgraph_checkpoint_db_path: str = ""
    llm_http_trust_env: bool = False


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
    llm_provider = normalize_llm_provider(os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER))
    llm_model = resolve_llm_model(llm_provider)
    llm_temperature = resolve_llm_temperature()
    pending_action_parser_model = (
        os.getenv("PENDING_ACTION_PARSER_MODEL", "").strip()
        or llm_model
    )
    pending_action_parser_temperature = resolve_pending_action_parser_temperature(llm_provider)
    llm_http_trust_env = resolve_llm_http_trust_env()

    _cached_settings = Settings(
        slack_enabled=parse_bool_env("SLACK_ENABLED", True),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
        web_enabled=parse_bool_env("WEB_ENABLED", False),
        web_host=os.getenv("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        web_port=int(os.getenv("WEB_PORT", "8000")),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
        minimax_base_url=(
            os.getenv("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL).strip()
            or DEFAULT_MINIMAX_BASE_URL
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=(
            os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).strip()
            or DEFAULT_OPENAI_BASE_URL
        ),
        pending_action_parser_model=pending_action_parser_model,
        pending_action_parser_temperature=pending_action_parser_temperature,
        pending_action_parser_confidence_threshold=float(
            os.getenv("PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD", "0.75")
        ),
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
        langgraph_checkpoint_db_path=os.getenv("LANGGRAPH_CHECKPOINT_DB_PATH", "").strip(),
        llm_http_trust_env=llm_http_trust_env,
    )
    return _cached_settings


def validate_bootstrap_settings(settings: Settings) -> None:
    validate_core_settings(settings)
    validate_interface_settings(settings)


def validate_core_settings(settings: Settings) -> None:
    if not is_agent_runtime_enabled(settings):
        return

    if settings.llm_provider not in SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(SUPPORTED_LLM_PROVIDERS)
        raise RuntimeError(f"Invalid LLM_PROVIDER: {settings.llm_provider}. Supported values: {supported}")

    missing: list[str] = []
    if settings.llm_provider == "google" and not settings.google_api_key:
        missing.append("GOOGLE_API_KEY")
    if settings.llm_provider == "minimax" and not settings.minimax_api_key:
        missing.append("MINIMAX_API_KEY")
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.google_application_credentials:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if not settings.jade_project_sheet_id:
        missing.append("JADE_PROJECT_SHEET_ID")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")

    parser_temperature_raw = os.getenv("PENDING_ACTION_PARSER_TEMPERATURE")
    if (
        settings.llm_provider == "minimax"
        and parser_temperature_raw is not None
        and parser_temperature_raw.strip()
        and settings.pending_action_parser_temperature <= 0
    ):
        raise RuntimeError("PENDING_ACTION_PARSER_TEMPERATURE must be greater than 0 when LLM_PROVIDER=minimax.")


def validate_interface_settings(settings: Settings) -> None:
    if not is_agent_runtime_enabled(settings):
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


def normalize_knowledge_file_type(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.startswith("."):
        cleaned = f".{cleaned}"
    return cleaned


def normalize_llm_provider(value: str) -> str:
    cleaned = value.strip().lower()
    return cleaned or DEFAULT_LLM_PROVIDER


def default_llm_model_for_provider(provider: str) -> str:
    normalized = normalize_llm_provider(provider)
    if normalized == "minimax":
        return DEFAULT_MINIMAX_MODEL
    if normalized == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_GOOGLE_MODEL


def default_pending_action_parser_temperature(provider: str) -> float:
    normalized = normalize_llm_provider(provider)
    if normalized == "minimax":
        return 0.01
    return 0.0


def resolve_llm_model(provider: str) -> str:
    generic_model = os.getenv("LLM_MODEL", "").strip()
    return generic_model or default_llm_model_for_provider(provider)


def resolve_llm_temperature() -> float:
    generic_temperature = os.getenv("LLM_TEMPERATURE", "").strip()
    return float(generic_temperature or str(DEFAULT_LLM_TEMPERATURE))


def resolve_pending_action_parser_temperature(provider: str) -> float:
    raw_value = os.getenv("PENDING_ACTION_PARSER_TEMPERATURE")
    if raw_value is None or not raw_value.strip():
        return default_pending_action_parser_temperature(provider)
    return float(raw_value)


def resolve_llm_http_trust_env() -> bool:
    return parse_bool_env("LLM_HTTP_TRUST_ENV", False)


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
