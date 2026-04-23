from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


ALLOWED_KINDS = {"agent", "agent_group"}
ALLOWED_RUNTIME = "adk"


class ManifestModel(BaseModel):
    id: str
    version: str
    kind: str
    runtime: str
    entrypoint: str
    owner: str
    description: str
    tags: list[str] = Field(default_factory=list)
    required_secrets: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    test_paths: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.isidentifier():
            raise ValueError("id must be a valid Python identifier.")
        return normalized

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in ALLOWED_KINDS:
            raise ValueError(f"kind must be one of {sorted(ALLOWED_KINDS)}.")
        return normalized

    @field_validator("runtime")
    @classmethod
    def validate_runtime(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != ALLOWED_RUNTIME:
            raise ValueError(f"runtime must be '{ALLOWED_RUNTIME}'.")
        return normalized

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        normalized = value.strip()
        module_name, separator, function_name = normalized.partition(":")
        if not separator or not module_name or not function_name:
            raise ValueError("entrypoint must use module:function syntax.")
        return normalized

    @field_validator("owner", "description", "version")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field cannot be empty.")
        return normalized

    @field_validator("tags", "required_secrets", "capabilities", "test_paths")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = str(value).strip()
            if item:
                normalized.append(item)
        return normalized


@dataclass(slots=True)
class AgentPackage:
    package_root: Path
    manifest_path: Path
    manifest: ManifestModel

    @property
    def id(self) -> str:
        return self.manifest.id

    def effective_test_paths(self) -> list[Path]:
        if self.manifest.test_paths:
            return [self.package_root / relative_path for relative_path in self.manifest.test_paths]
        default_tests_dir = self.package_root / "tests"
        if default_tests_dir.exists():
            return [default_tests_dir]
        return []


@dataclass(slots=True)
class ValidationReport:
    package: AgentPackage
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

