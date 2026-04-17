from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

MemoryScope = Literal["thread", "user", "workspace", "global"]
LongTermMemoryType = Literal["user", "feedback", "project", "reference"]
AgentMemoryScope = Literal["user", "project", "local"]
LongTermMemorySnapshotChoice = Literal["keep", "merge", "replace"]
SessionMemorySectionName = Literal[
    "current_state",
    "task_spec",
    "key_files",
    "workflow",
    "errors_corrections",
    "learnings",
    "worklog",
]


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


class SessionMemoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    source_path: str = ""
    current_state: str = ""
    task_spec: str = ""
    key_files: list[str] = Field(default_factory=list)
    workflow: str = ""
    errors_corrections: str = ""
    learnings: str = ""
    worklog: str = ""

    @field_validator("thread_id")
    @classmethod
    def validate_required_thread_id(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator(
        "source_path",
        "current_state",
        "task_spec",
        "workflow",
        "errors_corrections",
        "learnings",
        "worklog",
    )
    @classmethod
    def validate_optional_session_memory_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("key_files", mode="before")
    @classmethod
    def normalize_key_files(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("key_files must be a list of strings.")
        return [str(item) for item in value]

    @field_validator("key_files")
    @classmethod
    def validate_key_files(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized


class SessionMemoryFileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_state: str | None = None
    task_spec: str | None = None
    key_files: list[str] | None = None
    workflow: str | None = None
    errors_corrections: str | None = None
    learnings: str | None = None
    worklog: str | None = None

    @field_validator(
        "current_state",
        "task_spec",
        "workflow",
        "errors_corrections",
        "learnings",
        "worklog",
    )
    @classmethod
    def validate_optional_session_memory_update_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).strip()

    @field_validator("key_files", mode="before")
    @classmethod
    def normalize_update_key_files(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("key_files must be a list of strings.")
        return [str(item) for item in value]

    @field_validator("key_files")
    @classmethod
    def validate_update_key_files(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized


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


class LongTermMemoryFrontmatter(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    memory_type: LongTermMemoryType = Field(alias="type")

    @field_validator("name", "description")
    @classmethod
    def validate_required_frontmatter_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


class LongTermMemoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    memory_type: LongTermMemoryType
    content_markdown: str = ""

    @field_validator("memory_id", "relative_path", "source_path", "name", "description")
    @classmethod
    def validate_required_file_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("content_markdown")
    @classmethod
    def validate_optional_file_content(cls, value: str) -> str:
        return str(value or "").strip()


class LongTermMemoryIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    memory_type: LongTermMemoryType

    @field_validator("memory_id", "relative_path", "name", "description")
    @classmethod
    def validate_required_index_entry_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


class LongTermMemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    memory_type: LongTermMemoryType
    content_markdown: str = ""

    @field_validator("memory_id", "name", "description")
    @classmethod
    def validate_required_write_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("content_markdown")
    @classmethod
    def validate_optional_write_content(cls, value: str) -> str:
        return str(value or "").strip()


class LongTermMemoryCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: str = Field(min_length=1)
    index_file: LongTermMemoryFile
    index_entries: list[LongTermMemoryIndexEntry] = Field(default_factory=list)
    topic_files: list[LongTermMemoryFile] = Field(default_factory=list)

    @field_validator("root_dir")
    @classmethod
    def validate_required_catalog_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


class AgentMemoryRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    scope: AgentMemoryScope
    scope_key: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    memory_type: LongTermMemoryType
    content_markdown: str = ""
    score: float = Field(ge=0.0)

    @field_validator("memory_id", "scope_key", "relative_path", "source_path", "name", "description")
    @classmethod
    def validate_required_agent_retrieval_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("content_markdown")
    @classmethod
    def validate_optional_agent_retrieval_content(cls, value: str) -> str:
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


class LongTermMemoryConsolidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: str = Field(min_length=1)
    examined_count: int = Field(default=0, ge=0)
    updated_memory_ids: list[str] = Field(default_factory=list)
    deleted_memory_ids: list[str] = Field(default_factory=list)
    noisy_group_count: int = Field(default=0, ge=0)
    duplicate_group_count: int = Field(default=0, ge=0)

    @field_validator("root_dir")
    @classmethod
    def validate_required_root_dir(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("updated_memory_ids", "deleted_memory_ids", mode="before")
    @classmethod
    def normalize_memory_id_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("Value must be a list of strings.")
        return [str(item) for item in value]

    @field_validator("updated_memory_ids", "deleted_memory_ids")
    @classmethod
    def validate_memory_id_lists(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized


class LongTermMemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    root_dir: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    memory_count: int = Field(default=0, ge=0)
    memories: list[LongTermMemoryFile] = Field(default_factory=list)

    @field_validator("snapshot_id", "root_dir", "fingerprint")
    @classmethod
    def validate_required_snapshot_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned


class LongTermMemorySnapshotSyncState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    action: LongTermMemorySnapshotChoice
    updated_at: str = ""

    @field_validator("snapshot_id", "fingerprint")
    @classmethod
    def validate_required_sync_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("updated_at")
    @classmethod
    def validate_optional_sync_text(cls, value: str) -> str:
        return str(value or "").strip()


class LongTermMemorySnapshotApplySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    action: LongTermMemorySnapshotChoice
    user_root_dir: str = Field(min_length=1)
    project_root_dir: str = Field(min_length=1)
    created_memory_ids: list[str] = Field(default_factory=list)
    updated_memory_ids: list[str] = Field(default_factory=list)
    deleted_memory_ids: list[str] = Field(default_factory=list)

    @field_validator("snapshot_id", "fingerprint", "user_root_dir", "project_root_dir")
    @classmethod
    def validate_required_apply_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("created_memory_ids", "updated_memory_ids", "deleted_memory_ids", mode="before")
    @classmethod
    def normalize_snapshot_memory_id_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("Value must be a list of strings.")
        return [str(item) for item in value]

    @field_validator("created_memory_ids", "updated_memory_ids", "deleted_memory_ids")
    @classmethod
    def validate_snapshot_memory_id_lists(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized


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
