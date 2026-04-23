from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from google.adk.apps import App

from mas_platform.errors import LoadError
from mas_platform.models import AgentPackage
from mas_platform.registry import resolve_repo_root


def ensure_repo_root_on_path(repo_root: str | Path | None = None) -> Path:
    resolved_root = resolve_repo_root(repo_root)
    root_str = str(resolved_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    agents_path = resolved_root / "agents"
    agents_module = sys.modules.get("agents")
    if agents_module is not None and hasattr(agents_module, "__path__") and agents_path.exists():
        agents_module.__path__ = [str(agents_path), *list(agents_module.__path__)]
    importlib.invalidate_caches()
    return resolved_root


def split_entrypoint(entrypoint: str) -> tuple[str, str]:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise LoadError(f"Invalid entrypoint '{entrypoint}'.")
    return module_name, function_name


def import_module(module_name: str, repo_root: str | Path | None = None) -> ModuleType:
    ensure_repo_root_on_path(repo_root)
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise LoadError(f"Failed to import module '{module_name}': {exc}") from exc


def load_entrypoint(package: AgentPackage, repo_root: str | Path | None = None) -> Callable[[], App]:
    module_name, function_name = split_entrypoint(package.manifest.entrypoint)
    module = import_module(module_name, repo_root=repo_root)
    try:
        entrypoint = getattr(module, function_name)
    except AttributeError as exc:
        raise LoadError(
            f"Entrypoint function '{function_name}' not found in module '{module_name}'."
        ) from exc
    if not callable(entrypoint):
        raise LoadError(f"Entrypoint '{package.manifest.entrypoint}' is not callable.")
    return entrypoint


def load_app(package: AgentPackage, repo_root: str | Path | None = None) -> App:
    entrypoint = load_entrypoint(package, repo_root=repo_root)
    try:
        app = entrypoint()
    except Exception as exc:
        raise LoadError(
            f"Entrypoint '{package.manifest.entrypoint}' raised while building the ADK app: {exc}"
        ) from exc
    if not isinstance(app, App):
        raise LoadError(
            f"Entrypoint '{package.manifest.entrypoint}' must return google.adk.apps.App."
        )
    return app
