from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from core.config import Settings
from core.identity_map import build_user_identity_context, resolve_identity
from core.slack_formatting import to_slack_mrkdwn
from interfaces.slack_home import build_home_view

MENTION_PATTERN = re.compile(r"<@([A-Z0-9]+)>")
logger = logging.getLogger(__name__)


class SlackListener:
    def __init__(self, agent_graph, settings: Settings) -> None:
        self.agent_graph = agent_graph
        self.settings = settings
        self.app = App(token=settings.slack_bot_token)
        self._handler: SocketModeHandler | None = None
        self._register_handlers()

    def start(self) -> None:
        self._handler = SocketModeHandler(self.app, self.settings.slack_app_token)
        print("⚡ Jade Agent is listening on Slack...")
        self._handler.start()

    def stop(self) -> None:
        if self._handler is None:
            return
        try:
            self._handler.close()
        finally:
            self._handler = None

    def _register_handlers(self) -> None:
        self.app.event("app_home_opened")(self.handle_app_home_opened)
        self.app.event("app_mention")(self.handle_app_mention)
        self.app.event("message")(self.handle_message_event)

    def handle_app_home_opened(self, event) -> None:
        user_id = event.get("user")
        if not user_id:
            return
        self.publish_home_view(user_id)

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
            user_context = self._load_user_context(user_id)
            initial_state = {
                "messages": [HumanMessage(content=text)],
                "interface_name": "slack",
                "user_id": user_id,
                "channel_id": channel_id,
                **user_context,
            }
            final_state = self.agent_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": self._build_thread_id(event)}},
            )
            final_text = to_slack_mrkdwn(self._extract_final_text(final_state))
        except Exception as exc:
            logger.exception("Failed while processing Slack event", extra={"channel_id": channel_id, "user_id": user_id})
            final_text = to_slack_mrkdwn(f"I hit an error while processing that request: {exc}")
        finally:
            self._remove_thinking_reaction(channel_id, message_ts)

        reply_kwargs = {"text": final_text, "channel": channel_id}
        reply_thread_ts = self._build_reply_thread_ts(event)
        if reply_thread_ts:
            reply_kwargs["thread_ts"] = reply_thread_ts
        say(**reply_kwargs)

    def _extract_text(self, raw_text: str, is_mention: bool) -> str:
        if not raw_text:
            return ""

        cleaned = raw_text.strip()
        if is_mention:
            cleaned = MENTION_PATTERN.sub("", cleaned).strip()

        cleaned = self._replace_user_mentions(cleaned)
        return cleaned.strip()

    def _replace_user_mentions(self, text: str) -> str:
        def replace_match(match: re.Match[str]) -> str:
            mentioned_user_id = match.group(1)
            identity = self._load_user_context(mentioned_user_id)
            return (
                identity.get("user_sheet_name")
                or identity.get("user_google_name")
                or identity.get("user_display_name")
                or identity.get("user_real_name")
                or f"user_{mentioned_user_id}"
            )

        return MENTION_PATTERN.sub(replace_match, text)

    def _build_thread_id(self, event) -> str:
        channel_id = event.get("channel", "unknown-channel")
        if event.get("channel_type") == "im":
            return channel_id

        root_ts = event.get("thread_ts") or event.get("ts") or "root"
        return f"{channel_id}:{root_ts}"

    def _build_reply_thread_ts(self, event) -> str | None:
        if event.get("channel_type") == "im":
            return None
        return event.get("thread_ts") or event.get("ts")

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

    def _load_user_context(self, user_id: str) -> dict:
        try:
            response = self.app.client.users_info(user=user_id)
            user = response.get("user", {})
            profile = user.get("profile", {})
            display_name = profile.get("display_name") or user.get("name") or ""
            real_name = profile.get("real_name") or user.get("real_name") or ""
            email = profile.get("email", "")

            identity_context = build_user_identity_context(
                slack_display_name=display_name,
                slack_real_name=real_name,
                email=email,
            )
            if identity_context:
                return identity_context

            fallback_identity = (
                resolve_identity(display_name)
                or resolve_identity(real_name)
                or resolve_identity(email)
            )
            if fallback_identity:
                return {
                    "user_display_name": display_name or real_name,
                    "user_real_name": real_name,
                    "user_email": email,
                    "user_google_name": fallback_identity["google_name"],
                    "user_sheet_name": fallback_identity["sheet_name"],
                    "user_job_title": fallback_identity["job_title"],
                    "user_mapped_slack_name": fallback_identity["slack_name"],
                }

            return {
                "user_display_name": display_name or real_name,
                "user_real_name": real_name,
                "user_email": email,
            }
        except Exception:
            return {}

    def publish_home_view(self, user_id: str) -> None:
        try:
            user_context = self._load_user_context(user_id)
            self.app.client.views_publish(
                user_id=user_id,
                view=build_home_view(user_context),
            )
        except Exception:
            logger.exception("Failed to publish Slack Home view", extra={"user_id": user_id})

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
