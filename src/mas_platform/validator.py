from __future__ import annotations

import re
from pathlib import Path

from mas_platform.loader import load_app
from mas_platform.models import AgentPackage, ValidationReport
from mas_platform.registry import load_registry


ENV_PATTERNS = (
    re.compile(r"os\.getenv\(\s*['\"]([A-Z0-9_]+)['\"]"),
    re.compile(r"os\.environ\.get\(\s*['\"]([A-Z0-9_]+)['\"]"),
    re.compile(r"os\.environ\[\s*['\"]([A-Z0-9_]+)['\"]\s*\]"),
)


def detect_env_secret_usage(package_root: Path) -> set[str]:
    matches: set[str] = set()
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in ENV_PATTERNS:
            matches.update(pattern.findall(source))
    return matches


def validate_package(package: AgentPackage, *, repo_root: str | Path | None = None) -> ValidationReport:
    report = ValidationReport(package=package)

    if package.package_root.name != package.id:
        report.errors.append(
            f"Folder name '{package.package_root.name}' must match manifest id '{package.id}'."
        )

    test_paths = package.effective_test_paths()
    if not test_paths:
        report.errors.append("Package must define tests via test_paths or a tests/ directory.")
    else:
        for test_path in test_paths:
            if not test_path.exists():
                report.errors.append(f"Declared test path does not exist: {test_path}")

    detected_secrets = detect_env_secret_usage(package.package_root)
    declared_secrets = set(package.manifest.required_secrets)
    undeclared_secrets = sorted(detected_secrets - declared_secrets)
    if undeclared_secrets:
        report.errors.append(
            "Detected env-secret usage not declared in required_secrets: "
            + ", ".join(undeclared_secrets)
        )

    try:
        app = load_app(package, repo_root=repo_root)
    except Exception as exc:
        report.errors.append(str(exc))
    else:
        if app.name != package.id:
            report.warnings.append(
                f"App name '{app.name}' differs from package id '{package.id}'."
            )

    return report


def validate_registry(repo_root: str | Path | None = None) -> list[ValidationReport]:
    registry = load_registry(repo_root)
    return [
        validate_package(package, repo_root=repo_root)
        for package in registry.values()
    ]
