from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.config import Settings
from app.identity import build_user_identity_context, resolve_identity
from app.messages import extract_final_text
from app.paths import resolve_project_path
from interfaces.slack.formatting import to_slack_mrkdwn
from interfaces.slack.home import build_home_view
from tax_monitor_tool.verification import TaxMonitorVerificationBroker
from tools.conversion_google_sources import extract_google_document_references
from tools.document_conversion import ConversionSessionStore, UPLOAD_ONLY_FALLBACK_TEXT

MENTION_PATTERN = re.compile(r"<@([A-Z0-9]+)>")
CASUAL_GREETING_NORMALIZED_TEXTS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "早上好",
    "下午好",
    "晚上好",
}
logger = logging.getLogger(__name__)


def truncate_for_log(text: str, limit: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


class SlackListener:
    def __init__(
        self,
        agent_graph,
        settings: Settings,
        *,
        tax_monitor_verification_broker: TaxMonitorVerificationBroker | None = None,
    ) -> None:
        self.agent_graph = agent_graph
        self.settings = settings
        self.tax_monitor_verification_broker = tax_monitor_verification_broker
        self.app = App(token=settings.slack_bot_token)
        self.conversion_store = ConversionSessionStore(self._resolve_path(settings.conversion_work_dir))
        self._user_context_cache: dict[str, dict] = {}
        self._bot_user_id = self._load_bot_user_id()
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
        channel_id = str(event.get("channel", "")).strip()
        user_id = str(event.get("user", "")).strip()
        subtype = str(event.get("subtype", "")).strip()
        pending_request = (
            self.tax_monitor_verification_broker.get_active_request()
            if self.tax_monitor_verification_broker is not None
            else None
        )
        should_trace_verification = pending_request is not None or channel_id == self.settings.tax_monitor_slack_channel
        if should_trace_verification:
            logger.info(
                "Received Slack message event channel=%s user=%s subtype=%s channel_type=%s pending_tax_monitor_verification=%s text=%r",
                channel_id,
                user_id,
                subtype or "",
                str(event.get("channel_type", "")).strip(),
                pending_request is not None,
                truncate_for_log(str(event.get("text", ""))),
            )
        if event.get("bot_id"):
            if should_trace_verification:
                logger.info(
                    "Ignoring Slack message event from a bot during tax monitor verification channel=%s user=%s",
                    channel_id,
                    user_id,
                )
            return
        if subtype and subtype != "file_share":
            if should_trace_verification:
                logger.info(
                    "Ignoring Slack message event with unsupported subtype during tax monitor verification channel=%s user=%s subtype=%s",
                    channel_id,
                    user_id,
                    subtype,
                )
            return
        if self._handle_tax_monitor_verification_event(event=event, say=say):
            return
        if event.get("channel_type") != "im" and not (
            self._has_active_conversion_session(event) or self._contains_bot_mention(event.get("text", ""))
        ):
            if should_trace_verification:
                logger.info(
                    "Slack message event did not route to normal chat handling channel=%s user=%s reason=no_mention_or_active_session",
                    channel_id,
                    user_id,
                )
            return

        self.process_and_respond(
            event=event,
            say=say,
            is_mention=self._contains_bot_mention(event.get("text", "")),
        )

    def _handle_tax_monitor_verification_event(self, *, event, say) -> bool:
        if self.tax_monitor_verification_broker is None:
            return False

        channel_id = str(event.get("channel", "")).strip()
        user_id = str(event.get("user", "")).strip()
        text = str(event.get("text", ""))
        active_request = self.tax_monitor_verification_broker.get_active_request()
        contains_code = bool(re.search(r"\b\d{6}\b", text))
        logger.info(
            "Received Slack message during tax monitor verification channel=%s user=%s has_active_request=%s contains_code=%s",
            channel_id,
            user_id,
            active_request is not None,
            contains_code,
        )
        if not self.tax_monitor_verification_broker.has_pending_request(channel_id=channel_id):
            if active_request is not None:
                logger.warning(
                    "Ignoring Slack message for tax monitor verification because it arrived in a different channel active_channel=%s event_channel=%s user=%s",
                    active_request.channel_id,
                    channel_id,
                    user_id,
                )
            return False

        submission = self.tax_monitor_verification_broker.submit_code_from_text(
            channel_id=channel_id,
            text=text,
            user_id=user_id,
        )
        if submission is None:
            logger.warning(
                "Slack message did not satisfy the pending tax monitor verification request channel=%s user=%s contains_code=%s",
                channel_id,
                user_id,
                contains_code,
            )
            return False

        logger.info(
            "Slack message satisfied the pending tax monitor verification request channel=%s user=%s request_id=%s",
            channel_id,
            user_id,
            submission.request_id,
        )
        active_request = self.tax_monitor_verification_broker.get_active_request()
        reply_kwargs = {
            "channel": channel_id,
            "text": to_slack_mrkdwn("已收到税率监控登录验证码，正在重试登录。"),
        }
        if active_request is not None and active_request.message_ts:
            reply_kwargs["thread_ts"] = active_request.message_ts
        say(**reply_kwargs)
        return True

    def process_and_respond(self, event, say, is_mention: bool) -> None:
        user_id = event.get("user")
        channel_id = event.get("channel")
        message_ts = event.get("ts")
        thread_id = self._build_thread_id(event)
        uploaded_files = self._extract_uploaded_files(event)
        active_session = self.conversion_store.get_active_session_by_thread(thread_id)

        if not user_id or not channel_id:
            return

        text = self._extract_text(event.get("text", ""), is_mention=is_mention)
        if not text and uploaded_files:
            text = UPLOAD_ONLY_FALLBACK_TEXT
        if not text and not uploaded_files:
            return

        logger.debug(
            "Processing Slack event channel=%s user=%s thread=%s mention=%s uploads=%s active_conversion_session=%s text=%r",
            channel_id,
            user_id,
            thread_id,
            is_mention,
            len(uploaded_files),
            active_session.session_id if active_session is not None else "",
            truncate_for_log(text),
        )

        self._add_thinking_reaction(channel_id, message_ts)

        try:
            user_context = self._load_user_context(user_id)
            initial_state = {
                "messages": [HumanMessage(content=text)],
                "interface_name": "slack",
                "thread_id": thread_id,
                "user_id": user_id,
                "channel_id": channel_id,
                "uploaded_files": uploaded_files,
                **user_context,
            }
            if self._should_route_to_conversion(active_session=active_session, text=text, uploaded_files=uploaded_files):
                initial_state["route"] = "document_conversion_agent"
                initial_state["route_reason"] = "Slack document conversion session."
            if active_session is not None:
                initial_state["conversion_session_id"] = active_session.session_id

            logger.debug(
                "Invoking agent graph channel=%s user=%s thread=%s route=%s conversion_session_id=%s",
                channel_id,
                user_id,
                thread_id,
                initial_state.get("route", "gateway"),
                initial_state.get("conversion_session_id", ""),
            )

            final_state = self.agent_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )
            final_text = to_slack_mrkdwn(extract_final_text(final_state))
            logger.debug(
                "Completed Slack event channel=%s user=%s thread=%s response_chars=%s",
                channel_id,
                user_id,
                thread_id,
                len(final_text),
            )
        except Exception as exc:
            logger.exception(
                "Failed while processing Slack event channel=%s user=%s thread=%s active_conversion_session=%s",
                channel_id,
                user_id,
                thread_id,
                active_session.session_id if active_session is not None else "",
            )
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

    def _load_user_context(self, user_id: str) -> dict:
        cached = self._user_context_cache.get(user_id)
        if cached is not None:
            logger.debug("Using cached Slack user context user=%s", user_id)
            return dict(cached)
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
                self._user_context_cache[user_id] = dict(identity_context)
                return dict(identity_context)

            fallback_identity = (
                resolve_identity(display_name)
                or resolve_identity(real_name)
                or resolve_identity(email)
            )
            if fallback_identity:
                resolved_context = {
                    "user_display_name": display_name or real_name,
                    "user_real_name": real_name,
                    "user_email": email,
                    "user_google_name": fallback_identity["google_name"],
                    "user_sheet_name": fallback_identity["sheet_name"],
                    "user_job_title": fallback_identity["job_title"],
                    "user_mapped_slack_name": fallback_identity["slack_name"],
                }
                self._user_context_cache[user_id] = dict(resolved_context)
                return resolved_context

            resolved_context = {
                "user_display_name": display_name or real_name,
                "user_real_name": real_name,
                "user_email": email,
            }
            self._user_context_cache[user_id] = dict(resolved_context)
            return resolved_context
        except Exception:
            logger.debug("Failed to load Slack user context user=%s", user_id, exc_info=True)
            return {}

    def _extract_uploaded_files(self, event) -> list[dict[str, str]]:
        files = event.get("files")
        if not isinstance(files, list):
            return []

        extracted: list[dict[str, str]] = []
        for file_payload in files:
            if not isinstance(file_payload, dict):
                continue
            file_id = str(file_payload.get("id", "")).strip()
            name = str(file_payload.get("name") or file_payload.get("title") or file_id).strip()
            if not file_id or not name:
                continue
            extracted.append(
                {
                    "id": file_id,
                    "name": name,
                    "title": str(file_payload.get("title", "")).strip(),
                    "mimetype": str(file_payload.get("mimetype", "")).strip(),
                    "filetype": str(file_payload.get("filetype", "")).strip(),
                    "url_private": str(file_payload.get("url_private", "")).strip(),
                    "url_private_download": str(file_payload.get("url_private_download", "")).strip(),
                    "user": str(file_payload.get("user") or event.get("user") or "").strip(),
                }
            )
        return extracted

    def _has_active_conversion_session(self, event) -> bool:
        return self.conversion_store.has_active_session(self._build_thread_id(event))

    def _should_route_to_conversion(self, *, active_session, text: str, uploaded_files: list[dict[str, str]]) -> bool:
        if uploaded_files:
            return True
        if extract_google_document_references(text):
            return True
        if active_session is None:
            return False
        return not self._is_casual_greeting(text)

    def _is_casual_greeting(self, text: str) -> bool:
        normalized = re.sub(r"[!?,.，。！？\s]+", " ", text.strip().lower()).strip()
        if not normalized:
            return False
        return normalized in CASUAL_GREETING_NORMALIZED_TEXTS

    def _contains_bot_mention(self, text: str) -> bool:
        if not self._bot_user_id or not text:
            return False
        return f"<@{self._bot_user_id}>" in text

    def _load_bot_user_id(self) -> str:
        try:
            auth_test = getattr(self.app.client, "auth_test", None)
            if callable(auth_test):
                response = auth_test()
                return str(response.get("user_id", "")).strip()
        except Exception:
            return ""
        return ""

    def _resolve_path(self, configured_value: str):
        return resolve_project_path(configured_value, self.settings.conversion_work_dir)

    def publish_home_view(self, user_id: str) -> None:
        try:
            user_context = self._load_user_context(user_id)
            self.app.client.views_publish(
                user_id=user_id,
                view=build_home_view(user_context),
            )
        except Exception:
            logger.debug("Failed to publish Slack Home view user=%s", user_id, exc_info=True)

    def _add_thinking_reaction(self, channel_id: str, timestamp: str | None) -> None:
        if not timestamp or not str(self.settings.slack_thinking_reaction).strip():
            return
        try:
            self.app.client.reactions_add(
                channel=channel_id,
                timestamp=timestamp,
                name=self.settings.slack_thinking_reaction,
            )
        except Exception:
            logger.debug(
                "Failed to add thinking reaction channel=%s ts=%s reaction=%s",
                channel_id,
                timestamp,
                self.settings.slack_thinking_reaction,
                exc_info=True,
            )

    def _remove_thinking_reaction(self, channel_id: str, timestamp: str | None) -> None:
        if not timestamp or not str(self.settings.slack_thinking_reaction).strip():
            return
        try:
            self.app.client.reactions_remove(
                channel=channel_id,
                timestamp=timestamp,
                name=self.settings.slack_thinking_reaction,
            )
        except Exception:
            logger.debug(
                "Failed to remove thinking reaction channel=%s ts=%s reaction=%s",
                channel_id,
                timestamp,
                self.settings.slack_thinking_reaction,
                exc_info=True,
            )
