from pathlib import Path

from mas_platform.loader import load_app
from mas_platform.registry import load_registry


def test_frontdoor_builds_and_discovers_sub_agents() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    registry = load_registry(repo_root)
    app = load_app(registry["frontdoor"], repo_root=repo_root)

    assert app.name == "frontdoor"
    sub_agent_names = [agent.name for agent in app.root_agent.sub_agents]
    assert sub_agent_names == ["general_chat_agent"]
