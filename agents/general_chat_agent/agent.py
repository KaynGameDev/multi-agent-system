from __future__ import annotations

import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps import App


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.llm import build_generation_config, build_model, current_runtime_label


def build_instruction() -> str:
    return (
        "You are the team's general chat fallback agent.\n"
        "You are the default conversational assistant when no specialist agent is a better fit.\n"
        "Greet users naturally, answer broad questions helpfully, and keep the tone warm and concise.\n"
        "Be honest about the current platform state: right now you are the only specialist available, "
        "so you handle general chat while the team adds more agents.\n"
        f"If asked what model you are using, explain that you are currently configured with `{current_runtime_label()}`.\n"
        "Do not say you are only a greeter. You are allowed to have a normal conversation."
    )


def build_app() -> App:
    return App(
        name="general_chat_agent",
        root_agent=LlmAgent(
            name="general_chat_agent",
            description="Handles greetings and broad general chat when no specialist agent is a better fit.",
            model=build_model(),
            instruction=build_instruction(),
            generate_content_config=build_generation_config(),
        ),
    )


app = build_app()
root_agent = app.root_agent
