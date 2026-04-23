from __future__ import annotations

import os
from typing import TYPE_CHECKING

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest


def _normalize_provider(value: str | None) -> str:
    return (value or "openai").strip().lower()


def _normalize_temperature(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _current_model_name() -> str:
    return (os.getenv("LLM_MODEL") or "gpt-5.4").strip()


def current_runtime_label() -> str:
    provider = _normalize_provider(os.getenv("LLM_PROVIDER"))
    model_name = _current_model_name()
    if provider == "openai" and "/" not in model_name:
        return f"openai/{model_name}"
    return f"{provider}/{model_name}"


class StubLlm(BaseLlm):
    async def generate_content_async(
        self, llm_request: "LlmRequest", stream: bool = False
    ):
        latest_user_text = ""
        for content in reversed(llm_request.contents or []):
            if getattr(content, "role", None) != "user":
                continue
            parts = getattr(content, "parts", None) or []
            fragments = [part.text for part in parts if getattr(part, "text", None)]
            if fragments:
                latest_user_text = "\n".join(fragment for fragment in fragments if fragment).strip()
                break

        reply_text = (
            f"Stub LLM reply from {self.model}: {latest_user_text or 'hello'}"
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=reply_text)],
            ),
            finish_reason=types.FinishReason.STOP,
        )


def build_model():
    provider = _normalize_provider(os.getenv("LLM_PROVIDER"))
    model_name = _current_model_name()

    if provider == "stub":
        return StubLlm(model=model_name or "stub-general-chat")

    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            os.environ.setdefault("OPENAI_API_BASE", base_url)
        try:
            from google.adk.models.lite_llm import LiteLlm
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI-backed ADK routing requires LiteLLM support. "
                "Install `litellm` or `google-adk[extensions]`."
            ) from exc
        normalized_model = model_name if "/" in model_name else f"openai/{model_name}"
        return LiteLlm(model=normalized_model)

    if provider == "gemini":
        from google.adk.models import Gemini

        base_url = os.getenv("GEMINI_BASE_URL")
        kwargs = {"model": model_name}
        if base_url:
            kwargs["base_url"] = base_url
        return Gemini(**kwargs)

    raise RuntimeError(
        "Unsupported LLM_PROVIDER. Expected one of: openai, gemini."
    )


def build_generation_config() -> types.GenerateContentConfig:
    temperature = _normalize_temperature(os.getenv("LLM_TEMPERATURE"))
    if temperature is None:
        return types.GenerateContentConfig()
    return types.GenerateContentConfig(temperature=temperature)
