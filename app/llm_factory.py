from __future__ import annotations

from typing import Any

from app.config import Settings, normalize_llm_provider

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - dependency availability is tested through constructor patching
    ChatGoogleGenerativeAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - dependency availability is tested through constructor patching
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency availability is tested through constructor patching
    ChatOpenAI = None


def build_runtime_llms(settings: Settings) -> tuple[Any, Any]:
    primary_llm = build_chat_model(
        settings,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    parser_llm = build_chat_model(
        settings,
        model=settings.pending_action_parser_model,
        temperature=settings.pending_action_parser_temperature,
    )
    return primary_llm, parser_llm


def build_chat_model(
    settings: Settings,
    *,
    model: str,
    temperature: float,
):
    provider = normalize_llm_provider(settings.llm_provider)
    if provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=settings.google_api_key,
            client_args={"trust_env": settings.llm_http_trust_env},
        )

    if provider == "minimax":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic is not installed.")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.minimax_api_key,
            base_url=settings.minimax_base_url,
        )

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai is not installed.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    raise RuntimeError(f"Unsupported LLM provider: {settings.llm_provider}")
