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
DEFAULT_ASSISTANT_REQUEST_PARSER_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD = 0.75

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
DEFAULT_MEMORY_WORK_DIR = "runtime/memory"
DEFAULT_JADE_PROJECT_SKILLS_DIR = ".jade/skills"
DEFAULT_WEB_AUTH_SESSION_MAX_AGE_SECONDS = 60 * 60 * 12
DEFAULT_CONTEXT_WINDOW_AUTO_COMPACT_ENABLED = True
DEFAULT_CONTEXT_WINDOW_AUTO_COMPACT_PRESERVED_TAIL_COUNT = 1
DEFAULT_CONTEXT_WINDOW_AUTO_COMPACT_FAILURE_LIMIT = 3
DEFAULT_SESSION_MEMORY_ENABLED = True
DEFAULT_SESSION_MEMORY_INITIALIZE_THRESHOLD_TOKENS = 2_048
DEFAULT_SESSION_MEMORY_UPDATE_GROWTH_THRESHOLD_TOKENS = 768
DEFAULT_LONG_TERM_MEMORY_ENABLED = False
DEFAULT_MEMORY_RETRIEVAL_ENABLED = False
DEFAULT_MEMORY_RETRIEVAL_DEFAULT_LIMIT = 8


@dataclass(frozen=True)
class WebAuthCredential:
    username: str
    password: str


@dataclass(frozen=True)
class ResolvedLLMConfig:
    provider: str
    model: str
    temperature: float
    http_trust_env: bool
    google_api_key: str
    minimax_api_key: str
    minimax_base_url: str
    openai_api_key: str
    openai_base_url: str


@dataclass(frozen=True)
class Settings:
    slack_enabled: bool
    slack_bot_token: str
    slack_app_token: str
    web_enabled: bool
    web_host: str
    web_port: int
    web_allowed_hosts: tuple[str, ...]
    web_auth_enabled: bool
    web_auth_credentials: tuple[WebAuthCredential, ...]
    web_auth_session_secret: str
    web_auth_cookie_secure: bool
    web_auth_session_max_age_seconds: int
    llm_provider: str
    llm_model: str
    llm_temperature: float
    google_api_key: str
    minimax_api_key: str
    minimax_base_url: str
    openai_api_key: str
    openai_base_url: str
    llm_http_trust_env: bool
    routing_llm_provider: str
    routing_llm_model: str
    routing_llm_temperature: float
    routing_google_api_key: str
    routing_minimax_api_key: str
    routing_minimax_base_url: str
    routing_openai_api_key: str
    routing_openai_base_url: str
    routing_llm_http_trust_env: bool
    routing_llm_overrides_present: bool
    assistant_request_parser_confidence_threshold: float
    pending_action_parser_confidence_threshold: float
    google_application_credentials: str
    jade_project_sheet_id: str
    project_sheet_range: str
    project_sheet_cache_ttl_seconds: int
    slack_thinking_reaction: str
    project_lookup_keywords: tuple[str, ...]
    knowledge_base_dir: str
    knowledge_file_types: tuple[str, ...]
    context_window_effective_window: int | None
    context_window_warning_threshold: int | None
    context_window_auto_compact_threshold: int | None
    context_window_hard_block_threshold: int | None
    context_window_auto_compact_enabled: bool
    context_window_auto_compact_preserved_tail_count: int
    context_window_auto_compact_failure_limit: int
    session_memory_enabled: bool
    session_memory_initialize_threshold_tokens: int
    session_memory_update_growth_threshold_tokens: int
    jade_project_skills_dir: str = DEFAULT_JADE_PROJECT_SKILLS_DIR
    knowledge_google_sheets_catalog_path: str = DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH
    knowledge_google_sheets_cache_ttl_seconds: int = DEFAULT_KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS
    conversion_work_dir: str = DEFAULT_CONVERSION_WORK_DIR
    langgraph_checkpoint_db_path: str = ""
    memory_work_dir: str = DEFAULT_MEMORY_WORK_DIR
    long_term_memory_enabled: bool = DEFAULT_LONG_TERM_MEMORY_ENABLED
    memory_retrieval_enabled: bool = DEFAULT_MEMORY_RETRIEVAL_ENABLED
    memory_retrieval_default_limit: int = DEFAULT_MEMORY_RETRIEVAL_DEFAULT_LIMIT


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
    llm_http_trust_env = resolve_llm_http_trust_env()
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    minimax_api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    minimax_base_url = (
        os.getenv("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL).strip()
        or DEFAULT_MINIMAX_BASE_URL
    )
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = (
        os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).strip()
        or DEFAULT_OPENAI_BASE_URL
    )
    routing_llm_overrides_present = resolve_routing_llm_overrides_present()
    routing_llm_provider = resolve_routing_llm_provider(
        llm_provider,
        overrides_present=routing_llm_overrides_present,
    )
    routing_llm_model = resolve_routing_llm_model(
        routing_llm_provider,
        main_provider=llm_provider,
        main_model=llm_model,
        overrides_present=routing_llm_overrides_present,
    )
    routing_llm_temperature = resolve_routing_llm_temperature(llm_temperature)
    routing_llm_http_trust_env = resolve_routing_llm_http_trust_env(llm_http_trust_env)
    routing_google_api_key = resolve_routing_env_value("ROUTING_GOOGLE_API_KEY", google_api_key)
    routing_minimax_api_key = resolve_routing_env_value("ROUTING_MINIMAX_API_KEY", minimax_api_key)
    routing_minimax_base_url = resolve_routing_env_value("ROUTING_MINIMAX_BASE_URL", minimax_base_url)
    routing_openai_api_key = resolve_routing_env_value("ROUTING_OPENAI_API_KEY", openai_api_key)
    routing_openai_base_url = resolve_routing_env_value("ROUTING_OPENAI_BASE_URL", openai_base_url)
    web_allowed_hosts = parse_csv_env("WEB_ALLOWED_HOSTS")
    web_auth_enabled = parse_bool_env("WEB_AUTH_ENABLED", False)
    web_auth_credentials = resolve_web_auth_credentials() if web_auth_enabled else ()

    _cached_settings = Settings(
        slack_enabled=parse_bool_env("SLACK_ENABLED", True),
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN", ""),
        slack_app_token=os.getenv("SLACK_APP_TOKEN", ""),
        web_enabled=parse_bool_env("WEB_ENABLED", False),
        web_host=os.getenv("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        web_port=int(os.getenv("WEB_PORT", "8000")),
        web_allowed_hosts=web_allowed_hosts,
        web_auth_enabled=web_auth_enabled,
        web_auth_credentials=web_auth_credentials,
        web_auth_session_secret=os.getenv("WEB_AUTH_SESSION_SECRET", "").strip(),
        web_auth_cookie_secure=parse_bool_env("WEB_AUTH_COOKIE_SECURE", True),
        web_auth_session_max_age_seconds=int(
            os.getenv(
                "WEB_AUTH_SESSION_MAX_AGE_SECONDS",
                str(DEFAULT_WEB_AUTH_SESSION_MAX_AGE_SECONDS),
            )
        ),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        google_api_key=google_api_key,
        minimax_api_key=minimax_api_key,
        minimax_base_url=minimax_base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        llm_http_trust_env=llm_http_trust_env,
        routing_llm_provider=routing_llm_provider,
        routing_llm_model=routing_llm_model,
        routing_llm_temperature=routing_llm_temperature,
        routing_google_api_key=routing_google_api_key,
        routing_minimax_api_key=routing_minimax_api_key,
        routing_minimax_base_url=routing_minimax_base_url,
        routing_openai_api_key=routing_openai_api_key,
        routing_openai_base_url=routing_openai_base_url,
        routing_llm_http_trust_env=routing_llm_http_trust_env,
        routing_llm_overrides_present=routing_llm_overrides_present,
        assistant_request_parser_confidence_threshold=float(
            os.getenv(
                "ASSISTANT_REQUEST_PARSER_CONFIDENCE_THRESHOLD",
                str(DEFAULT_ASSISTANT_REQUEST_PARSER_CONFIDENCE_THRESHOLD),
            )
        ),
        pending_action_parser_confidence_threshold=float(
            os.getenv(
                "PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD",
                str(DEFAULT_PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD),
            )
        ),
        google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        jade_project_sheet_id=os.getenv("JADE_PROJECT_SHEET_ID", ""),
        project_sheet_range=os.getenv("PROJECT_SHEET_RANGE", "Tasks!A1:Z"),
        project_sheet_cache_ttl_seconds=int(os.getenv("PROJECT_SHEET_CACHE_TTL_SECONDS", "30")),
        slack_thinking_reaction=os.getenv("SLACK_THINKING_REACTION", "eyes"),
        project_lookup_keywords=keywords,
        knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", DEFAULT_KNOWLEDGE_BASE_DIR).strip() or DEFAULT_KNOWLEDGE_BASE_DIR,
        knowledge_file_types=knowledge_file_types,
        context_window_effective_window=parse_optional_positive_int_env("CONTEXT_WINDOW_EFFECTIVE_WINDOW"),
        context_window_warning_threshold=parse_optional_positive_int_env("CONTEXT_WINDOW_WARNING_THRESHOLD"),
        context_window_auto_compact_threshold=parse_optional_positive_int_env("CONTEXT_WINDOW_AUTO_COMPACT_THRESHOLD"),
        context_window_hard_block_threshold=parse_optional_positive_int_env("CONTEXT_WINDOW_HARD_BLOCK_THRESHOLD"),
        context_window_auto_compact_enabled=parse_bool_env(
            "CONTEXT_WINDOW_AUTO_COMPACT_ENABLED",
            DEFAULT_CONTEXT_WINDOW_AUTO_COMPACT_ENABLED,
        ),
        context_window_auto_compact_preserved_tail_count=parse_non_negative_int_env(
            "CONTEXT_WINDOW_AUTO_COMPACT_PRESERVED_TAIL_COUNT",
            DEFAULT_CONTEXT_WINDOW_AUTO_COMPACT_PRESERVED_TAIL_COUNT,
        ),
        context_window_auto_compact_failure_limit=parse_positive_int_env(
            "CONTEXT_WINDOW_AUTO_COMPACT_FAILURE_LIMIT",
            DEFAULT_CONTEXT_WINDOW_AUTO_COMPACT_FAILURE_LIMIT,
        ),
        session_memory_enabled=parse_bool_env(
            "SESSION_MEMORY_ENABLED",
            DEFAULT_SESSION_MEMORY_ENABLED,
        ),
        session_memory_initialize_threshold_tokens=parse_positive_int_env(
            "SESSION_MEMORY_INITIALIZE_THRESHOLD_TOKENS",
            DEFAULT_SESSION_MEMORY_INITIALIZE_THRESHOLD_TOKENS,
        ),
        session_memory_update_growth_threshold_tokens=parse_positive_int_env(
            "SESSION_MEMORY_UPDATE_GROWTH_THRESHOLD_TOKENS",
            DEFAULT_SESSION_MEMORY_UPDATE_GROWTH_THRESHOLD_TOKENS,
        ),
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
        memory_work_dir=(
            os.getenv("MEMORY_WORK_DIR", DEFAULT_MEMORY_WORK_DIR).strip()
            or DEFAULT_MEMORY_WORK_DIR
        ),
        long_term_memory_enabled=parse_bool_env(
            "LONG_TERM_MEMORY_ENABLED",
            DEFAULT_LONG_TERM_MEMORY_ENABLED,
        ),
        memory_retrieval_enabled=parse_bool_env(
            "MEMORY_RETRIEVAL_ENABLED",
            DEFAULT_MEMORY_RETRIEVAL_ENABLED,
        ),
        memory_retrieval_default_limit=parse_positive_int_env(
            "MEMORY_RETRIEVAL_DEFAULT_LIMIT",
            DEFAULT_MEMORY_RETRIEVAL_DEFAULT_LIMIT,
        ),
    )
    return _cached_settings


def validate_bootstrap_settings(settings: Settings) -> None:
    validate_core_settings(settings)
    validate_interface_settings(settings)


def validate_core_settings(settings: Settings) -> None:
    if not is_agent_runtime_enabled(settings):
        return

    missing: list[str] = []
    agent_llm_config = resolve_agent_llm_config(settings)
    validate_llm_provider(agent_llm_config.provider, env_name="LLM_PROVIDER")
    missing.extend(required_llm_credentials(agent_llm_config))

    if settings.routing_llm_overrides_present:
        routing_llm_config = resolve_routing_llm_config(settings)
        validate_llm_provider(routing_llm_config.provider, env_name="ROUTING_LLM_PROVIDER")
        missing.extend(required_llm_credentials(routing_llm_config, routing=True))

    if not settings.google_application_credentials:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if not settings.jade_project_sheet_id:
        missing.append("JADE_PROJECT_SHEET_ID")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")

    routing_temperature_raw = os.getenv("ROUTING_LLM_TEMPERATURE")
    if (
        settings.routing_llm_overrides_present
        and settings.routing_llm_provider == "minimax"
        and routing_temperature_raw is not None
        and routing_temperature_raw.strip()
        and settings.routing_llm_temperature <= 0
    ):
        raise RuntimeError("ROUTING_LLM_TEMPERATURE must be greater than 0 when ROUTING_LLM_PROVIDER=minimax.")


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
    if settings.web_enabled and settings.web_auth_enabled:
        if not settings.web_auth_session_secret:
            raise RuntimeError("Missing required environment variables: WEB_AUTH_SESSION_SECRET")
        if not settings.web_auth_credentials:
            raise RuntimeError(
                "Missing required environment variables: WEB_AUTH_CREDENTIALS or WEB_AUTH_USERNAME/WEB_AUTH_PASSWORD"
            )
        if settings.web_auth_session_max_age_seconds <= 0:
            raise RuntimeError("WEB_AUTH_SESSION_MAX_AGE_SECONDS must be a positive integer.")


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


def resolve_llm_model(provider: str) -> str:
    generic_model = os.getenv("LLM_MODEL", "").strip()
    return generic_model or default_llm_model_for_provider(provider)


def resolve_llm_temperature() -> float:
    generic_temperature = os.getenv("LLM_TEMPERATURE", "").strip()
    return float(generic_temperature or str(DEFAULT_LLM_TEMPERATURE))


def resolve_llm_http_trust_env() -> bool:
    return parse_bool_env("LLM_HTTP_TRUST_ENV", False)


def resolve_routing_llm_overrides_present() -> bool:
    for name in (
        "ROUTING_LLM_PROVIDER",
        "ROUTING_LLM_MODEL",
        "ROUTING_LLM_TEMPERATURE",
        "ROUTING_LLM_HTTP_TRUST_ENV",
        "ROUTING_GOOGLE_API_KEY",
        "ROUTING_MINIMAX_API_KEY",
        "ROUTING_MINIMAX_BASE_URL",
        "ROUTING_OPENAI_API_KEY",
        "ROUTING_OPENAI_BASE_URL",
    ):
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        if name == "ROUTING_LLM_HTTP_TRUST_ENV":
            return True
        if raw_value.strip():
            return True
    return False


def resolve_routing_llm_provider(
    main_provider: str,
    *,
    overrides_present: bool,
) -> str:
    if not overrides_present:
        return main_provider
    raw_value = os.getenv("ROUTING_LLM_PROVIDER", "").strip()
    return normalize_llm_provider(raw_value or main_provider)


def resolve_routing_llm_model(
    routing_provider: str,
    *,
    main_provider: str,
    main_model: str,
    overrides_present: bool,
) -> str:
    raw_value = os.getenv("ROUTING_LLM_MODEL", "").strip()
    if raw_value:
        return raw_value
    if not overrides_present or routing_provider == main_provider:
        return main_model
    return default_llm_model_for_provider(routing_provider)


def resolve_routing_llm_temperature(main_temperature: float) -> float:
    raw_value = os.getenv("ROUTING_LLM_TEMPERATURE")
    if raw_value is None or not raw_value.strip():
        return float(main_temperature)
    return float(raw_value)


def resolve_routing_llm_http_trust_env(main_trust_env: bool) -> bool:
    raw_value = os.getenv("ROUTING_LLM_HTTP_TRUST_ENV")
    if raw_value is None:
        return bool(main_trust_env)
    return parse_bool_env("ROUTING_LLM_HTTP_TRUST_ENV", main_trust_env)


def resolve_routing_env_value(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip()


def resolve_agent_llm_config(settings: Settings) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        http_trust_env=settings.llm_http_trust_env,
        google_api_key=settings.google_api_key,
        minimax_api_key=settings.minimax_api_key,
        minimax_base_url=settings.minimax_base_url,
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
    )


def resolve_routing_llm_config(settings: Settings) -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        provider=settings.routing_llm_provider,
        model=settings.routing_llm_model,
        temperature=settings.routing_llm_temperature,
        http_trust_env=settings.routing_llm_http_trust_env,
        google_api_key=settings.routing_google_api_key,
        minimax_api_key=settings.routing_minimax_api_key,
        minimax_base_url=settings.routing_minimax_base_url,
        openai_api_key=settings.routing_openai_api_key,
        openai_base_url=settings.routing_openai_base_url,
    )


def validate_llm_provider(provider: str, *, env_name: str) -> None:
    if provider in SUPPORTED_LLM_PROVIDERS:
        return
    supported = ", ".join(SUPPORTED_LLM_PROVIDERS)
    raise RuntimeError(f"Invalid {env_name}: {provider}. Supported values: {supported}")


def required_llm_credentials(
    config: ResolvedLLMConfig,
    *,
    routing: bool = False,
) -> list[str]:
    if config.provider == "google" and not config.google_api_key:
        return ["ROUTING_GOOGLE_API_KEY or GOOGLE_API_KEY" if routing else "GOOGLE_API_KEY"]
    if config.provider == "minimax" and not config.minimax_api_key:
        return ["ROUTING_MINIMAX_API_KEY or MINIMAX_API_KEY" if routing else "MINIMAX_API_KEY"]
    if config.provider == "openai" and not config.openai_api_key:
        return ["ROUTING_OPENAI_API_KEY or OPENAI_API_KEY" if routing else "OPENAI_API_KEY"]
    return []


def parse_csv_env(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    return tuple(
        item.strip()
        for item in raw_value.replace("\n", ",").replace(";", ",").split(",")
        if item.strip()
    )


def resolve_web_auth_credentials() -> tuple[WebAuthCredential, ...]:
    credentials: list[WebAuthCredential] = []
    seen_usernames: set[str] = set()

    single_username = os.getenv("WEB_AUTH_USERNAME", "").strip()
    single_password = os.getenv("WEB_AUTH_PASSWORD", "").strip()
    if bool(single_username) != bool(single_password):
        raise RuntimeError("WEB_AUTH_USERNAME and WEB_AUTH_PASSWORD must be set together.")
    if single_username and single_password:
        credentials.append(WebAuthCredential(username=single_username, password=single_password))
        seen_usernames.add(single_username)

    for entry in parse_csv_env("WEB_AUTH_CREDENTIALS"):
        username, separator, password = entry.partition(":")
        normalized_username = username.strip()
        normalized_password = password.strip()
        if not separator or not normalized_username or not normalized_password:
            raise RuntimeError("WEB_AUTH_CREDENTIALS entries must use username:password format.")
        if normalized_username in seen_usernames:
            raise RuntimeError(f"Duplicate web auth username configured: {normalized_username}")
        credentials.append(
            WebAuthCredential(
                username=normalized_username,
                password=normalized_password,
            )
        )
        seen_usernames.add(normalized_username)

    return tuple(credentials)


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


def parse_optional_positive_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    value = int(raw_value)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def parse_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        value = int(default)
    else:
        value = int(raw_value)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def parse_non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        value = int(default)
    else:
        value = int(raw_value)
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer.")
    return value
