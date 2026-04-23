from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from google.adk.apps import App


MANIFEST_NAME = "manifest.yaml"


@dataclass(slots=True)
class AgentManifest:
    id: str
    version: str
    kind: str
    runtime: str
    entrypoint: str
    owner: str
    description: str
    tags: list[str]
    required_secrets: list[str]
    capabilities: list[str]
    test_paths: list[str]


@dataclass(slots=True)
class AgentPackage:
    package_root: Path
    manifest_path: Path
    manifest: AgentManifest

    @property
    def id(self) -> str:
        return self.manifest.id


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        return Path.cwd().resolve()
    return Path(repo_root).resolve()


def agents_root(repo_root: str | Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / "agents"


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item:
            normalized.append(item)
    return normalized


def _parse_manifest(raw_payload: dict[str, Any], manifest_path: Path) -> AgentManifest:
    required_keys = [
        "id",
        "version",
        "kind",
        "runtime",
        "entrypoint",
        "owner",
        "description",
    ]
    for key in required_keys:
        value = str(raw_payload.get(key, "")).strip()
        if not value:
            raise ValueError(f"Manifest missing required field '{key}': {manifest_path}")

    manifest = AgentManifest(
        id=str(raw_payload["id"]).strip(),
        version=str(raw_payload["version"]).strip(),
        kind=str(raw_payload["kind"]).strip(),
        runtime=str(raw_payload["runtime"]).strip(),
        entrypoint=str(raw_payload["entrypoint"]).strip(),
        owner=str(raw_payload["owner"]).strip(),
        description=str(raw_payload["description"]).strip(),
        tags=_normalize_string_list(raw_payload.get("tags", [])),
        required_secrets=_normalize_string_list(raw_payload.get("required_secrets", [])),
        capabilities=_normalize_string_list(raw_payload.get("capabilities", [])),
        test_paths=_normalize_string_list(raw_payload.get("test_paths", [])),
    )
    if not manifest.id.isidentifier():
        raise ValueError(f"Manifest id must be a valid Python identifier: {manifest_path}")
    if ":" not in manifest.entrypoint:
        raise ValueError(f"Manifest entrypoint must use module:function syntax: {manifest_path}")
    return manifest


def iter_package_dirs(repo_root: str | Path | None = None):
    root = agents_root(repo_root)
    if not root.exists():
        raise FileNotFoundError(f"Agents directory not found: {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        yield child


def load_package(package_root: Path) -> AgentPackage:
    manifest_path = package_root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Package '{package_root.name}' is missing {MANIFEST_NAME}.")
    raw_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(f"Manifest must be a mapping: {manifest_path}")
    return AgentPackage(
        package_root=package_root,
        manifest_path=manifest_path,
        manifest=_parse_manifest(raw_payload, manifest_path),
    )


def discover_packages(
    repo_root: str | Path | None = None,
    *,
    exclude_ids: set[str] | None = None,
) -> dict[str, AgentPackage]:
    excluded = exclude_ids or set()
    registry: dict[str, AgentPackage] = {}
    for package_root in iter_package_dirs(repo_root):
        package = load_package(package_root)
        if package.id in excluded:
            continue
        if package.id in registry:
            raise ValueError(
                f"Duplicate agent id '{package.id}' found in "
                f"{registry[package.id].package_root} and {package.package_root}."
            )
        registry[package.id] = package
    return registry


def ensure_repo_root_on_path(repo_root: str | Path | None = None) -> Path:
    resolved_root = resolve_repo_root(repo_root)
    root_str = str(resolved_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    importlib.invalidate_caches()
    return resolved_root


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"Invalid entrypoint '{entrypoint}'.")
    return module_name, function_name


def load_entrypoint(
    package: AgentPackage,
    repo_root: str | Path | None = None,
) -> Callable[[], App]:
    ensure_repo_root_on_path(repo_root)
    module_name, function_name = split_entrypoint(package.manifest.entrypoint)
    module = importlib.import_module(module_name)
    entrypoint = getattr(module, function_name, None)
    if not callable(entrypoint):
        raise TypeError(f"Entrypoint '{package.manifest.entrypoint}' is not callable.")
    return entrypoint


def load_app(package: AgentPackage, repo_root: str | Path | None = None) -> App:
    entrypoint = load_entrypoint(package, repo_root=repo_root)
    app = entrypoint()
    if not isinstance(app, App):
        raise TypeError(
            f"Entrypoint '{package.manifest.entrypoint}' must return google.adk.apps.App."
        )
    return app
