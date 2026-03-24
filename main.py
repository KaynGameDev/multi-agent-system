from __future__ import annotations

from collections.abc import Sequence
from threading import Thread

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from core.config import load_settings, validate_bootstrap_settings
from core.graph import build_agent_graph
from interfaces.slack_listener import SlackListener


def bootstrap_system() -> list[object]:
    load_dotenv()
    settings = load_settings(force_reload=True)
    validate_bootstrap_settings(settings)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
    )

    checkpointer = InMemorySaver()
    agent_graph = build_agent_graph(llm, checkpointer=checkpointer, settings=settings)

    listeners: list[object] = []
    if settings.slack_bot_token and settings.slack_app_token:
        listeners.append(SlackListener(agent_graph=agent_graph, settings=settings))
    if not listeners:
        raise RuntimeError("No communication interface is configured.")

    print("⚙ Compiled Jade Agent graph.")
    return listeners


def _start_background_listeners(listeners: Sequence[object]) -> list[Thread]:
    threads: list[Thread] = []
    for listener in listeners:
        thread = Thread(
            target=listener.start,
            name=listener.__class__.__name__,
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    return threads


def _stop_listener(listener: object) -> None:
    stop = getattr(listener, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            pass


def main() -> int:
    listeners = bootstrap_system()
    try:
        if len(listeners) == 1:
            listeners[0].start()
        else:
            _start_background_listeners(listeners[:-1])
            listeners[-1].start()
    except KeyboardInterrupt:
        print("\nStopping Jade Agent...")
    finally:
        for listener in reversed(listeners):
            _stop_listener(listener)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
