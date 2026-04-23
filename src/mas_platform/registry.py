from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from mas_platform.errors import RegistryError
from mas_platform.models import AgentPackage, ManifestModel


MANIFEST_NAME = "manifest.yaml"


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        return Path.cwd().resolve()
    return Path(repo_root).resolve()


def agents_root(repo_root: str | Path | None = None) -> Path:
    return resolve_repo_root(repo_root) / "agents"


def iter_package_dirs(repo_root: str | Path | None = None) -> Iterable[Path]:
    root = agents_root(repo_root)
    if not root.exists():
        raise RegistryError(f"Agents directory not found: {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        yield child


def load_package(package_root: Path) -> AgentPackage:
    manifest_path = package_root / MANIFEST_NAME
    if not manifest_path.exists():
        raise RegistryError(f"Package '{package_root.name}' is missing {MANIFEST_NAME}.")

    try:
        raw_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"Failed to parse {manifest_path}: {exc}") from exc

    if not isinstance(raw_payload, dict):
        raise RegistryError(f"Manifest must be a mapping: {manifest_path}")

    try:
        manifest = ManifestModel.model_validate(raw_payload)
    except Exception as exc:
        raise RegistryError(f"Invalid manifest at {manifest_path}: {exc}") from exc

    return AgentPackage(
        package_root=package_root,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_registry(repo_root: str | Path | None = None) -> dict[str, AgentPackage]:
    registry: dict[str, AgentPackage] = {}
    for package_root in iter_package_dirs(repo_root):
        package = load_package(package_root)
        if package.id in registry:
            raise RegistryError(
                f"Duplicate agent id '{package.id}' found in "
                f"{registry[package.id].package_root} and {package.package_root}."
            )
        registry[package.id] = package
    return registry

