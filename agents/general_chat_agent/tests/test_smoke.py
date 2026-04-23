from pathlib import Path

from mas_platform.runtime import run_package


def test_general_chat_agent_smoke(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("LLM_MODEL", "stub-general-chat")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    repo_root = Path(__file__).resolve().parents[3]
    result = run_package(repo_root, agent_id="general_chat_agent", message="hello team")
    assert any("Stub LLM reply" in line for line in result.event_lines)
    assert any("hello team" in line.lower() for line in result.event_lines)
