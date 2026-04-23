from pathlib import Path

from mas_platform.cli import main


def test_cli_list(repo_root: Path, capsys) -> None:
    exit_code = main(["--repo-root", str(repo_root), "list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "frontdoor" in captured.out
    assert "general_chat_agent" in captured.out


def test_cli_validate(repo_root: Path, capsys) -> None:
    exit_code = main(["--repo-root", str(repo_root), "validate", "frontdoor"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OK\tfrontdoor" in captured.out


def test_cli_scaffold(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "scaffold",
            "agent",
            "roadmap_agent",
            "--owner",
            "platform-team",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "agents" / "roadmap_agent" / "manifest.yaml").exists()
    assert (tmp_path / "agents" / "roadmap_agent" / "prompts.py").exists()
    assert (tmp_path / "agents" / "roadmap_agent" / "tools.py").exists()
    assert (tmp_path / "agents" / "roadmap_agent" / "tests" / "test_smoke.py").exists()


def test_cli_test_runs_package_tests(repo_root: Path) -> None:
    exit_code = main(["--repo-root", str(repo_root), "test", "general_chat_agent"])
    assert exit_code == 0
