from __future__ import annotations

import re

from langchain_core.messages import HumanMessage
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from core.config import Settings

MENTION_PATTERN = re.compile(r"<@[A-Z0-9]+>")


class SlackListener:
    def __init__(self, agent_graph, settings: Settings) -> None:
        self.agent_graph = agent_graph
        self.settings = settings
        self.app = App(token=settings.slack_bot_token)
        self._register_handlers()

    def start(self) -> None:
        handler = SocketModeHandler(self.app, self.settings.slack_app_token)
        print("⚡ Jade Agent is listening on Slack...")
        handler.start()

    def _register_handlers(self) -> None:
        self.app.event("app_mention")(self.handle_app_mention)
        self.app.event("message")(self.handle_message_event)

    def handle_app_mention(self, event, say) -> None:
        self.process_and_respond(event=event, say=say, is_mention=True)

    def handle_message_event(self, event, say) -> None:
        if event.get("bot_id"):
            return
        if event.get("subtype"):
            return
        if event.get("channel_type") != "im":
            return

        self.process_and_respond(event=event, say=say, is_mention=False)

    def process_and_respond(self, event, say, is_mention: bool) -> None:
        user_id = event.get("user")
        channel_id = event.get("channel")
        message_ts = event.get("ts")

        if not user_id or not channel_id:
            return

        text = self._extract_text(event.get("text", ""), is_mention=is_mention)
        if not text:
            return

        self._add_thinking_reaction(channel_id, message_ts)

        try:
            initial_state = {
                "messages": [HumanMessage(content=text)],
                "user_id": user_id,
                "channel_id": channel_id,
            }
            final_state = self.agent_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": self._build_thread_id(event, user_id)}},
            )
            final_text = self._extract_final_text(final_state)
        except Exception as exc:
            final_text = f"I hit an error while processing that request: {exc}"
        finally:
            self._remove_thinking_reaction(channel_id, message_ts)

        say(text=final_text, channel=channel_id)

    def _extract_text(self, raw_text: str, is_mention: bool) -> str:
        if not raw_text:
            return ""

        cleaned = raw_text.strip()
        if is_mention:
            cleaned = MENTION_PATTERN.sub("", cleaned).strip()
        return cleaned

    def _build_thread_id(self, event, user_id: str) -> str:
        channel_id = event.get("channel", "unknown-channel")
        root_ts = event.get("thread_ts") or event.get("ts") or "root"
        return f"{channel_id}:{root_ts}:{user_id}"

    def _extract_final_text(self, final_state: dict) -> str:
        messages = final_state.get("messages") or []
        if not messages:
            return "I couldn't generate a response."

        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        return self._stringify_content(content) or "I couldn't generate a response."

    def _stringify_content(self, content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()

        return str(content)

    def _add_thinking_reaction(self, channel_id: str, timestamp: str | None) -> None:
        if not timestamp:
            return
        try:
            self.app.client.reactions_add(
                channel=channel_id,
                timestamp=timestamp,
                name=self.settings.slack_thinking_reaction,
            )
        except Exception:
            pass

    def _remove_thinking_reaction(self, channel_id: str, timestamp: str | None) -> None:
        if not timestamp:
            return
        try:
            self.app.client.reactions_remove(
                channel=channel_id,
                timestamp=timestamp,
                name=self.settings.slack_thinking_reaction,
            )
        except Exception:
            pass
