from pathlib import Path

from mas_platform.runtime import run_package


def test_run_general_chat_agent(repo_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_MODEL", "stub-general-chat")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    result = run_package(repo_root, agent_id="general_chat_agent", message="ping")
    assert result.app_name == "general_chat_agent"
    assert any("Stub LLM reply" in line for line in result.event_lines)
    assert any("ping" in line for line in result.event_lines)
