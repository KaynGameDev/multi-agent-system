from pathlib import Path

from mas_platform.registry import load_package, load_registry
from mas_platform.validator import validate_package, validate_registry


def test_validate_registry_passes_for_published_packages(repo_root: Path) -> None:
    reports = validate_registry(repo_root)
    assert reports
    assert all(report.ok for report in reports)


def test_general_chat_agent_exposes_module_level_app(repo_root: Path) -> None:
    registry = load_registry(repo_root)
    report = validate_package(registry["general_chat_agent"], repo_root=repo_root)
    assert report.ok


def test_frontdoor_exposes_module_level_app(repo_root: Path) -> None:
    registry = load_registry(repo_root)
    report = validate_package(registry["frontdoor"], repo_root=repo_root)
    assert report.ok


def test_validate_package_rejects_non_app_entrypoint(tmp_path: Path) -> None:
    package_root = tmp_path / "agents" / "bad_agent"
    package_root.mkdir(parents=True)
    (tmp_path / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "manifest.yaml").write_text(
        "\n".join(
            [
                "id: bad_agent",
                "version: 0.1.0",
                "kind: agent",
                "runtime: adk",
                "entrypoint: agents.bad_agent.agent:build_app",
                "owner: platform-team",
                "description: invalid entrypoint sample",
                "test_paths:",
                "  - tests",
            ]
        ),
        encoding="utf-8",
    )
    (package_root / "tests").mkdir()
    (package_root / "tests" / "test_smoke.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    (package_root / "agent.py").write_text("def build_app():\n    return object()\n", encoding="utf-8")

    report = validate_package(load_package(package_root), repo_root=tmp_path)
    assert not report.ok
    assert any("must return google.adk.apps.App" in error for error in report.errors)


def test_validate_package_detects_undeclared_secrets(tmp_path: Path) -> None:
    package_root = tmp_path / "agents" / "secret_agent"
    package_root.mkdir(parents=True)
    (tmp_path / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "manifest.yaml").write_text(
        "\n".join(
            [
                "id: secret_agent",
                "version: 0.1.0",
                "kind: agent",
                "runtime: adk",
                "entrypoint: agents.secret_agent.agent:build_app",
                "owner: platform-team",
                "description: secret usage sample",
                "test_paths:",
                "  - tests",
            ]
        ),
        encoding="utf-8",
    )
    (package_root / "tests").mkdir()
    (package_root / "tests" / "test_smoke.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    (package_root / "agent.py").write_text(
        "\n".join(
            [
                "import os",
                "from google.adk.apps import App",
                "",
                "def build_app():",
                "    os.getenv('SECRET_TOKEN')",
                "    return App(name='secret_agent', root_agent=None)",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_package(load_package(package_root), repo_root=tmp_path)
    assert not report.ok
    assert any("SECRET_TOKEN" in error for error in report.errors)
