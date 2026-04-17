from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from threading import Thread

from dotenv import load_dotenv

from app.checkpoints import build_checkpoint_store
from app.config import is_agent_runtime_enabled, is_slack_enabled, load_settings, validate_bootstrap_settings
from app.graph import build_agent_graph, build_web_agent_registrations
from app.interpretation.intent_parser import IntentParser
from app.interpretation.model_config import IntentParserModelConfig
from app.llm_factory import build_runtime_llms
from app.routing.agent_router import AgentRouter
from app.routing.pending_action_router import PendingActionRouter
from interfaces.slack.listener import SlackListener
from interfaces.web.server import WebServer, format_web_chat_url

logger = logging.getLogger(__name__)


class _BelowErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR


class _RuntimeResourceCloser:
    def __init__(self, *resources: object) -> None:
        self._resources = resources

    def start(self) -> None:
        return None

    def stop(self) -> None:
        for resource in reversed(self._resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(_BelowErrorFilter())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(max(level, logging.ERROR))
    stderr_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


def bootstrap_system() -> list[object]:
    load_dotenv()
    configure_logging()
    settings = load_settings(force_reload=True)
    validate_bootstrap_settings(settings)
    logger.debug(
        "Configuring LLMs agent_provider=%s agent_model=%s agent_temperature=%s "
        "routing_provider=%s routing_model=%s routing_temperature=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.llm_temperature,
        settings.routing_llm_provider,
        settings.routing_llm_model,
        settings.routing_llm_temperature,
    )

    listeners: list[object] = []
    if is_agent_runtime_enabled(settings):
        runtime_llms = build_runtime_llms(settings)
        routing_backup_llm = (
            None
            if runtime_llms.routing_llm is runtime_llms.agent_llm
            else runtime_llms.agent_llm
        )
        assistant_request_parser_config = IntentParserModelConfig(
            confidence_threshold=settings.assistant_request_parser_confidence_threshold,
        )
        pending_action_parser_config = IntentParserModelConfig(
            confidence_threshold=settings.pending_action_parser_confidence_threshold,
        )
        assistant_request_parser = IntentParser(
            runtime_llms.routing_llm,
            backup_llm=routing_backup_llm,
            config=assistant_request_parser_config,
        )
        pending_action_parser = IntentParser(
            runtime_llms.routing_llm,
            backup_llm=routing_backup_llm,
            config=pending_action_parser_config,
        )
        pending_action_router = PendingActionRouter(pending_action_parser)
        agent_router = AgentRouter(assistant_request_parser)
        checkpoint_store = build_checkpoint_store(settings)
        try:
            agent_graph = build_agent_graph(
                runtime_llms.agent_llm,
                checkpointer=checkpoint_store.saver,
                settings=settings,
                pending_action_router=pending_action_router,
                agent_router=agent_router,
            )
            web_graph = build_agent_graph(
                runtime_llms.agent_llm,
                checkpointer=checkpoint_store.saver,
                settings=settings,
                agent_registrations=build_web_agent_registrations(
                    settings=settings,
                    pending_action_router=pending_action_router,
                ),
                pending_action_router=pending_action_router,
                agent_router=agent_router,
            )
        except Exception:
            checkpoint_store.close()
            raise

        listeners.append(_RuntimeResourceCloser(checkpoint_store))

        if is_slack_enabled(settings):
            listeners.append(
                SlackListener(
                    agent_graph=agent_graph,
                    settings=settings,
                )
            )
        elif not settings.slack_enabled and (settings.slack_bot_token or settings.slack_app_token):
            print("💤 Slack listener disabled via SLACK_ENABLED=false")
        if settings.web_enabled:
            if not settings.web_auth_enabled:
                logger.warning(
                    "Web interface is running WITHOUT authentication (WEB_AUTH_ENABLED=false). "
                    "Do not expose this server on a public URL without enabling auth."
                )
            listeners.append(
                WebServer(
                    agent_graph=web_graph,
                    settings=settings,
                    checkpoint_store=checkpoint_store,
                )
            )
            print(f"🌐 Web chat: {format_web_chat_url(settings.web_host, settings.web_port)}")

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
            logger.debug("Error stopping listener %s", listener.__class__.__name__, exc_info=True)


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
