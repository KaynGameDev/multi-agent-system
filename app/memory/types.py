from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

MemoryScope = Literal["thread", "user", "workspace", "global"]


class MemoryReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    value: str = Field(min_length=1)

    @field_validator("kind", "value")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


class SessionMemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    summary_markdown: str = Field(min_length=1)
    last_message_id: str = ""
    covered_message_count: int = Field(default=0, ge=0)
    covered_tokens: int = Field(default=0, ge=0)

    @field_validator("thread_id", "summary_markdown")
    @classmethod
    def validate_required_text_fields(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("last_message_id")
    @classmethod
    def validate_optional_text_fields(cls, value: str) -> str:
        return str(value or "").strip()


class LongTermMemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    scope: MemoryScope = "global"
    namespace: str = ""
    content_markdown: str = Field(min_length=1)
    references: list[MemoryReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "content_markdown")
    @classmethod
    def validate_required_record_text_fields(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("namespace")
    @classmethod
    def validate_optional_record_text_fields(cls, value: str) -> str:
        return str(value or "").strip()


class MemoryRetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    namespace: str = ""
    scope: MemoryScope = "global"
    thread_id: str = ""
    user_id: str = ""
    top_k: int = Field(default=8, ge=1)

    @field_validator("query")
    @classmethod
    def validate_required_query_text_fields(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("namespace", "thread_id", "user_id")
    @classmethod
    def validate_optional_query_text_fields(cls, value: str) -> str:
        return str(value or "").strip()


class MemoryRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    content_markdown: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    scope: MemoryScope = "global"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "content_markdown")
    @classmethod
    def validate_result_text_fields(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


class ConversationCompactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    trigger: str = "manual"
    preserved_tail_count: int = Field(default=0, ge=0)
    allow_session_memory_reuse: bool = True

    @field_validator("thread_id")
    @classmethod
    def validate_required_compaction_text_fields(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("trigger")
    @classmethod
    def validate_trigger(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        return cleaned or "manual"


class ConversationCompactionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    summary_markdown: str = ""
    compacted_source_count: int = Field(default=0, ge=0)
    preserved_tail_count: int = Field(default=0, ge=0)
    used_session_memory: bool = False

    @field_validator("thread_id")
    @classmethod
    def validate_required_summary_text_fields(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("summary_markdown")
    @classmethod
    def validate_optional_summary_text_fields(cls, value: str) -> str:
        return str(value or "").strip()
