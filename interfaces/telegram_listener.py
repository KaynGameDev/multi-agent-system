from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from urllib import error, request

from langchain_core.messages import HumanMessage

from core.config import Settings
from core.identity_map import build_user_identity_context
from core.telegram_formatting import to_telegram_html

logger = logging.getLogger(__name__)

TELEGRAM_API_ROOT = "https://api.telegram.org/bot"
MAX_TELEGRAM_MESSAGE_LENGTH = 3500
MAX_TRACKED_REPLY_THREADS = 512
MENTION_TEMPLATE = r"(?i)(?<!\w)@{username}\b"
ASK_COMMAND_PATTERN = re.compile(
    r"^/(?P<command>ask|jade)(?:@(?P<target>[A-Za-z0-9_]+))?(?:\s+(?P<body>.*))?$",
    re.IGNORECASE,
)
HELP_COMMAND_PATTERN = re.compile(
    r"^/(?P<command>start|help)(?:@(?P<target>[A-Za-z0-9_]+))?\s*$",
    re.IGNORECASE,
)


class TelegramListener:
    def __init__(self, agent_graph, settings: Settings) -> None:
        self.agent_graph = agent_graph
        self.settings = settings
        self._stop_requested = False
        self._next_update_offset: int | None = None
        self._bot_user_id: int | None = None
        self._bot_username = ""
        self._thread_ids_by_reply: OrderedDict[tuple[str, int], str] = OrderedDict()

    def start(self) -> None:
        self._stop_requested = False
        self._ensure_bot_identity()
        print("📨 Jade Agent is listening on Telegram...")

        while not self._stop_requested:
            try:
                updates = self._fetch_updates()
            except Exception:
                if self._stop_requested:
                    break
                logger.exception("Failed to poll Telegram updates")
                time.sleep(2)
                continue

            for update in updates:
                try:
                    self.handle_update(update)
                except Exception:
                    logger.exception("Failed while processing Telegram update")

    def stop(self) -> None:
        self._stop_requested = True

    def handle_update(self, update: dict) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self._next_update_offset = update_id + 1

        message = update.get("message")
        if not isinstance(message, dict):
            return
        if not self._should_process_message(message):
            return

        text = (message.get("text") or "").strip()
        if not text:
            return

        if self._is_help_command(text):
            self._send_help_message(message)
            return

        prompt = self._extract_prompt_text(message)
        if not prompt:
            return

        thread_id = self._build_thread_id(message)
        chat = message.get("chat") or {}
        user = message.get("from") or {}
        chat_id = str(chat.get("id", ""))
        user_id = str(user.get("id", ""))

        try:
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "interface_name": "telegram",
                "user_id": user_id,
                "channel_id": chat_id,
                **self._build_user_context(user),
            }
            final_state = self.agent_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )
            final_text = self._extract_final_text(final_state)
        except Exception as exc:
            logger.exception(
                "Failed while processing Telegram message",
                extra={"chat_id": chat_id, "user_id": user_id},
            )
            final_text = f"I hit an error while processing that request: {exc}"

        sent_message_ids = self._send_response(
            chat_id=chat_id,
            text=final_text,
            reply_to_message_id=message.get("message_id"),
            message_thread_id=message.get("message_thread_id"),
        )
        for sent_message_id in sent_message_ids:
            self._remember_reply_thread(chat_id, sent_message_id, thread_id)

    def _ensure_bot_identity(self) -> None:
        if self._bot_user_id and self._bot_username:
            return

        self._api_request("deleteWebhook", {"drop_pending_updates": False})
        bot_info = self._api_request("getMe")
        self._bot_user_id = bot_info.get("id")
        self._bot_username = str(bot_info.get("username", "")).strip()
        if not self._bot_user_id or not self._bot_username:
            raise RuntimeError("Telegram getMe did not return a valid bot identity.")

    def _fetch_updates(self) -> list[dict]:
        payload: dict[str, object] = {
            "timeout": 15,
            "allowed_updates": ["message"],
        }
        if self._next_update_offset is not None:
            payload["offset"] = self._next_update_offset

        result = self._api_request("getUpdates", payload, timeout=25)
        if not isinstance(result, list):
            return []
        return [item for item in result if isinstance(item, dict)]

    def _should_process_message(self, message: dict) -> bool:
        from_user = message.get("from") or {}
        if from_user.get("is_bot"):
            return False

        chat = message.get("chat") or {}
        chat_type = chat.get("type")
        if chat_type not in {"private", "group", "supergroup"}:
            return False
        if not self._is_allowed_chat(chat):
            return False
        if not isinstance(message.get("text"), str):
            return False

        text = (message.get("text") or "").strip()
        if not text:
            return False

        if chat_type == "private":
            return True

        return (
            self._is_addressed_command(text)
            or self._mentions_bot(text)
            or self._is_reply_to_bot(message)
        )

    def _extract_prompt_text(self, message: dict) -> str:
        text = (message.get("text") or "").strip()
        command_match = ASK_COMMAND_PATTERN.match(text)
        if command_match and self._matches_bot_target(command_match.group("target")):
            return (command_match.group("body") or "").strip()

        if self._mentions_bot(text):
            mention_pattern = re.compile(
                MENTION_TEMPLATE.format(username=re.escape(self._bot_username))
            )
            text = mention_pattern.sub("", text).strip()
            text = re.sub(r"^[\s,:-]+", "", text)
            text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    def _build_thread_id(self, message: dict) -> str:
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", "unknown-chat"))
        chat_type = chat.get("type")

        if chat_type == "private":
            return f"telegram:{chat_id}"

        message_thread_id = message.get("message_thread_id")
        if message_thread_id is not None:
            return f"telegram:{chat_id}:topic:{message_thread_id}"

        reply_to_message = message.get("reply_to_message") or {}
        reply_from = reply_to_message.get("from") or {}
        reply_message_id = reply_to_message.get("message_id")
        if (
            isinstance(reply_message_id, int)
            and reply_from.get("id") == self._bot_user_id
        ):
            tracked_thread_id = self._thread_ids_by_reply.get((chat_id, reply_message_id))
            if tracked_thread_id:
                return tracked_thread_id

        return f"telegram:{chat_id}:message:{message.get('message_id', 'root')}"

    def _build_user_context(self, user: dict) -> dict:
        username = str(user.get("username") or "").strip()
        real_name = self._compose_full_name(user)
        return build_user_identity_context(
            slack_display_name=username or real_name,
            slack_real_name=real_name,
            email="",
        )

    def _send_help_message(self, message: dict) -> None:
        help_text = (
            "Ask me in a private chat, mention me in a group, reply to one of my answers, "
            "or use /jade <question>."
        )
        self._send_response(
            chat_id=str((message.get("chat") or {}).get("id", "")),
            text=help_text,
            reply_to_message_id=message.get("message_id"),
            message_thread_id=message.get("message_thread_id"),
        )

    def _send_response(
        self,
        *,
        chat_id: str,
        text: str,
        reply_to_message_id: int | None,
        message_thread_id: int | None,
    ) -> list[int]:
        sent_message_ids: list[int] = []
        for chunk in self._split_text(text):
            formatted_chunk = to_telegram_html(chunk)
            payload: dict[str, object] = {
                "chat_id": chat_id,
                "text": formatted_chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            if reply_to_message_id is not None:
                payload["reply_parameters"] = {
                    "message_id": reply_to_message_id,
                    "allow_sending_without_reply": True,
                }
            if message_thread_id is not None:
                payload["message_thread_id"] = message_thread_id

            result = self._api_request("sendMessage", payload)
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                sent_message_ids.append(message_id)

        return sent_message_ids

    def _remember_reply_thread(self, chat_id: str, reply_message_id: int, thread_id: str) -> None:
        key = (chat_id, reply_message_id)
        if key in self._thread_ids_by_reply:
            self._thread_ids_by_reply.move_to_end(key)
        self._thread_ids_by_reply[key] = thread_id
        while len(self._thread_ids_by_reply) > MAX_TRACKED_REPLY_THREADS:
            self._thread_ids_by_reply.popitem(last=False)

    def _is_allowed_chat(self, chat: dict) -> bool:
        if not self.settings.telegram_allowed_chat_ids:
            return True
        return str(chat.get("id", "")) in self.settings.telegram_allowed_chat_ids

    def _is_reply_to_bot(self, message: dict) -> bool:
        reply_to_message = message.get("reply_to_message") or {}
        reply_from = reply_to_message.get("from") or {}
        return reply_from.get("id") == self._bot_user_id

    def _mentions_bot(self, text: str) -> bool:
        if not self._bot_username:
            return False
        mention_pattern = re.compile(
            MENTION_TEMPLATE.format(username=re.escape(self._bot_username))
        )
        return bool(mention_pattern.search(text))

    def _is_help_command(self, text: str) -> bool:
        match = HELP_COMMAND_PATTERN.match(text)
        return bool(match and self._matches_bot_target(match.group("target")))

    def _is_addressed_command(self, text: str) -> bool:
        match = ASK_COMMAND_PATTERN.match(text)
        return bool(match and self._matches_bot_target(match.group("target")))

    def _matches_bot_target(self, target: str | None) -> bool:
        if not target:
            return True
        return target.casefold() == self._bot_username.casefold()

    def _compose_full_name(self, user: dict) -> str:
        first_name = str(user.get("first_name") or "").strip()
        last_name = str(user.get("last_name") or "").strip()
        return " ".join(part for part in (first_name, last_name) if part).strip()

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

    def _split_text(self, text: str) -> list[str]:
        cleaned = text.strip() or "I couldn't generate a response."
        if len(cleaned) <= MAX_TELEGRAM_MESSAGE_LENGTH:
            return [cleaned]

        chunks: list[str] = []
        current_lines: list[str] = []
        current_length = 0

        for line in cleaned.splitlines(keepends=True):
            line_length = len(line)
            if current_lines and current_length + line_length > MAX_TELEGRAM_MESSAGE_LENGTH:
                chunks.append("".join(current_lines).rstrip())
                current_lines = []
                current_length = 0

            if line_length > MAX_TELEGRAM_MESSAGE_LENGTH:
                if current_lines:
                    chunks.append("".join(current_lines).rstrip())
                    current_lines = []
                    current_length = 0
                for start in range(0, line_length, MAX_TELEGRAM_MESSAGE_LENGTH):
                    chunks.append(line[start : start + MAX_TELEGRAM_MESSAGE_LENGTH].rstrip())
                continue

            current_lines.append(line)
            current_length += line_length

        if current_lines:
            chunks.append("".join(current_lines).rstrip())

        return [chunk for chunk in chunks if chunk]

    def _api_request(
        self,
        method: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: float = 20,
    ):
        url = f"{TELEGRAM_API_ROOT}{self.settings.telegram_bot_token}/{method}"
        body = json.dumps(payload or {}).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Telegram API {method} failed with HTTP {exc.code}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Telegram API {method} request failed: {exc}") from exc

        parsed = json.loads(raw_response)
        if not parsed.get("ok"):
            raise RuntimeError(
                f"Telegram API {method} failed: {parsed.get('description', 'unknown error')}"
            )
        return parsed.get("result")
