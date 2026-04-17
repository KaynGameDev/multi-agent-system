from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from app.config import (
    ResolvedLLMConfig,
    Settings,
    normalize_llm_provider,
    resolve_agent_llm_config,
    resolve_routing_llm_config,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - dependency availability is tested through constructor patching
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency availability is tested through constructor patching
    ChatOpenAI = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeLLMs:
    agent_llm: Any
    routing_llm: Any


class LoggedLLM:
    def __init__(
        self,
        delegate: Any,
        *,
        role: str,
        provider: str,
        model: str,
        operation: str = "chat_model",
    ) -> None:
        self.delegate = delegate
        self.role = role
        self.provider = provider
        self.model = model
        self.operation = operation

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._log_request("invoke")
        return self.delegate.invoke(input, *args, **kwargs)

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._log_request("ainvoke")
        return await self.delegate.ainvoke(input, *args, **kwargs)

    def stream(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._log_request("stream")
        return self.delegate.stream(input, *args, **kwargs)

    async def astream(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._log_request("astream")
        async for chunk in self.delegate.astream(input, *args, **kwargs):
            yield chunk

    def bind_tools(self, tools: Any, *args: Any, **kwargs: Any) -> Any:
        bound = self.delegate.bind_tools(tools, *args, **kwargs)
        return self._wrap_child(
            bound,
            operation=f"{self.operation}.bind_tools",
        )

    def with_structured_output(self, schema: Any, *args: Any, **kwargs: Any) -> Any:
        structured = self.delegate.with_structured_output(schema, *args, **kwargs)
        schema_name = getattr(schema, "__name__", schema.__class__.__name__)
        return self._wrap_child(
            structured,
            operation=f"{self.operation}.with_structured_output[{schema_name}]",
        )

    def _wrap_child(self, child: Any, *, operation: str) -> Any:
        if child is None:
            return None
        if isinstance(child, LoggedLLM):
            return LoggedLLM(
                child.delegate,
                role=self.role,
                provider=self.provider,
                model=self.model,
                operation=operation,
            )
        return LoggedLLM(
            child,
            role=self.role,
            provider=self.provider,
            model=self.model,
            operation=operation,
        )

    def _log_request(self, method: str) -> None:
        logger.info(
            "LLM request role=%s provider=%s model=%s operation=%s method=%s",
            self.role,
            self.provider,
            self.model,
            self.operation,
            method,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def build_runtime_llms(settings: Settings) -> RuntimeLLMs:
    agent_config = resolve_agent_llm_config(settings)
    routing_config = resolve_routing_llm_config(settings)
    agent_llm = instrument_llm(
        build_chat_model(agent_config),
        role="agent_llm",
        config=agent_config,
    )
    routing_llm = (
        agent_llm
        if routing_config == agent_config
        else instrument_llm(
            build_chat_model(routing_config),
            role="routing_llm",
            config=routing_config,
        )
    )
    return RuntimeLLMs(
        agent_llm=agent_llm,
        routing_llm=routing_llm,
    )


def instrument_llm(
    llm: Any,
    *,
    role: str,
    config: ResolvedLLMConfig,
) -> LoggedLLM:
    logger.info(
        "Configured LLM role=%s provider=%s model=%s",
        role,
        config.provider,
        config.model,
    )
    return LoggedLLM(
        llm,
        role=role,
        provider=config.provider,
        model=config.model,
    )


def build_chat_model(config: ResolvedLLMConfig):
    provider = normalize_llm_provider(config.provider)
    if provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai is not installed.")
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            google_api_key=config.google_api_key,
            client_args={"trust_env": config.http_trust_env},
        )

    if provider == "minimax":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai is not installed.")
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.minimax_api_key,
            base_url=config.minimax_base_url,
        )

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai is not installed.")
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

    raise RuntimeError(f"Unsupported LLM provider: {config.provider}")
