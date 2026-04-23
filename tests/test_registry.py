from pathlib import Path

import pytest

from mas_platform.errors import RegistryError
from mas_platform.registry import load_registry


def test_registry_discovers_published_packages(repo_root: Path) -> None:
    registry = load_registry(repo_root)
    assert set(registry) == {"frontdoor", "general_chat_agent"}


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "alpha").mkdir()
    (tmp_path / "agents" / "beta").mkdir()
    for folder_name in ("alpha", "beta"):
        (tmp_path / "agents" / folder_name / "manifest.yaml").write_text(
            "\n".join(
                [
                    "id: duplicate_agent",
                    "version: 0.1.0",
                    "kind: agent",
                    "runtime: adk",
                    f"entrypoint: agents.{folder_name}.agent:build_app",
                    "owner: platform-team",
                    "description: duplicate",
                ]
            ),
            encoding="utf-8",
        )

    with pytest.raises(RegistryError):
        load_registry(tmp_path)
