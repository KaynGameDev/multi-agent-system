from __future__ import annotations

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from core.config import load_settings, validate_bootstrap_settings
from core.graph import build_agent_graph
from interfaces.slack_listener import SlackListener


def bootstrap_system() -> SlackListener:
    load_dotenv()
    settings = load_settings(force_reload=True)
    validate_bootstrap_settings(settings)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
    )

    checkpointer = InMemorySaver()
    agent_graph = build_agent_graph(llm, checkpointer=checkpointer)

    print("⚙ Compiled Jade Agent graph.")
    return SlackListener(agent_graph=agent_graph, settings=settings)


def main() -> int:
    listener = bootstrap_system()
    try:
        listener.start()
    except KeyboardInterrupt:
        print("\nStopping Jade Agent...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
