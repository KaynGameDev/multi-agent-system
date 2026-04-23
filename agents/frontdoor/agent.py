from __future__ import annotations

import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps import App


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.catalog import discover_packages, load_app
from shared.llm import build_generation_config, build_model


FRONTDOOR_ID = "frontdoor"
FALLBACK_AGENT_ID = "general_chat_agent"


def repo_root() -> Path:
    return REPO_ROOT


def discover_routable_apps() -> list[tuple[str, str, object]]:
    packages = discover_packages(repo_root(), exclude_ids={FRONTDOOR_ID})
    ordered_ids = sorted(packages)
    if FALLBACK_AGENT_ID in ordered_ids:
        ordered_ids.remove(FALLBACK_AGENT_ID)
        ordered_ids.insert(0, FALLBACK_AGENT_ID)

    routable: list[tuple[str, str, object]] = []
    for agent_id in ordered_ids:
        package = packages[agent_id]
        app = load_app(package, repo_root=repo_root())
        routable.append((agent_id, package.manifest.description, app.root_agent))
    return routable


def build_instruction(agent_summaries: list[tuple[str, str]]) -> str:
    roster = "\n".join(
        f"- {agent_id}: {description}" for agent_id, description in agent_summaries
    )
    fallback = (
        f"Delegate greetings, broad requests, and unclear requests to `{FALLBACK_AGENT_ID}`."
        if any(agent_id == FALLBACK_AGENT_ID for agent_id, _ in agent_summaries)
        else "If none of the available agents is a strong match, answer briefly and ask a clarifying question."
    )
    return (
        "You are the team's front-door routing agent.\n"
        "Greet the user warmly, then decide whether a specialist sub-agent should take over.\n"
        "Transfer only when another agent is clearly a better fit than you.\n"
        f"{fallback}\n"
        "Do not invent capabilities. Use the agent descriptions below as the source of truth.\n"
        "Available agents:\n"
        f"{roster}"
    )


def build_app() -> App:
    routable_apps = discover_routable_apps()
    sub_agents = [root_agent for _, _, root_agent in routable_apps]
    instruction = build_instruction(
        [(agent_id, description) for agent_id, description, _ in routable_apps]
    )

    return App(
        name=FRONTDOOR_ID,
        root_agent=LlmAgent(
            name=FRONTDOOR_ID,
            description="Greets the user and routes them to the best merged team agent.",
            model=build_model(),
            instruction=instruction,
            generate_content_config=build_generation_config(),
            sub_agents=sub_agents,
        ),
    )


app = build_app()
root_agent = app.root_agent
