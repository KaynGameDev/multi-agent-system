from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Condition, Event

VERIFICATION_CODE_PATTERN = re.compile(r"\b(?P<code>\d{6})\b")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaxMonitorVerificationRequest:
    request_id: str
    channel_id: str
    prompt: str
    created_at: str
    message_ts: str = ""


@dataclass(frozen=True)
class TaxMonitorVerificationSubmission:
    request_id: str
    channel_id: str
    code: str
    submitted_at: str
    user_id: str = ""


class TaxMonitorVerificationBroker:
    def __init__(self, *, channel_id: str) -> None:
        self.channel_id = channel_id.strip()
        self._condition = Condition()
        self._active_request: TaxMonitorVerificationRequest | None = None
        self._submission: TaxMonitorVerificationSubmission | None = None

    def open_request(self, *, prompt: str) -> TaxMonitorVerificationRequest:
        request = TaxMonitorVerificationRequest(
            request_id=secrets.token_hex(8),
            channel_id=self.channel_id,
            prompt=prompt.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._condition:
            self._active_request = request
            self._submission = None
            self._condition.notify_all()
        logger.info(
            "Opened tax monitor verification request request_id=%s channel=%s",
            request.request_id,
            request.channel_id,
        )
        return request

    def record_request_message(self, *, request_id: str, message_ts: str) -> None:
        with self._condition:
            if self._active_request is None or self._active_request.request_id != request_id:
                return
            self._active_request = TaxMonitorVerificationRequest(
                request_id=self._active_request.request_id,
                channel_id=self._active_request.channel_id,
                prompt=self._active_request.prompt,
                created_at=self._active_request.created_at,
                message_ts=str(message_ts or "").strip(),
            )
            self._condition.notify_all()
        logger.debug(
            "Recorded tax monitor verification request message request_id=%s channel=%s message_ts=%s",
            request_id,
            self.channel_id,
            str(message_ts or "").strip(),
        )

    def get_active_request(self) -> TaxMonitorVerificationRequest | None:
        with self._condition:
            return self._active_request

    def has_pending_request(self, *, channel_id: str) -> bool:
        with self._condition:
            return bool(
                self._active_request is not None
                and self._active_request.channel_id == str(channel_id or "").strip()
            )

    def submit_code_from_text(
        self,
        *,
        channel_id: str,
        text: str,
        user_id: str = "",
    ) -> TaxMonitorVerificationSubmission | None:
        normalized_channel_id = str(channel_id or "").strip()
        if not normalized_channel_id:
            return None

        match = VERIFICATION_CODE_PATTERN.search(str(text or ""))
        if match is None:
            logger.debug(
                "Ignoring tax monitor verification submission without a 6-digit code channel=%s user=%s",
                normalized_channel_id,
                str(user_id or "").strip(),
            )
            return None

        with self._condition:
            if self._active_request is None or self._active_request.channel_id != normalized_channel_id:
                logger.debug(
                    "Ignoring tax monitor verification submission with no matching active request channel=%s user=%s",
                    normalized_channel_id,
                    str(user_id or "").strip(),
                )
                return None

            submission = TaxMonitorVerificationSubmission(
                request_id=self._active_request.request_id,
                channel_id=normalized_channel_id,
                code=match.group("code"),
                submitted_at=datetime.now(timezone.utc).isoformat(),
                user_id=str(user_id or "").strip(),
            )
            self._submission = submission
            self._condition.notify_all()
            logger.info(
                "Accepted tax monitor verification submission request_id=%s channel=%s user=%s code=%s",
                submission.request_id,
                submission.channel_id,
                submission.user_id,
                _mask_code(submission.code),
            )
            return submission

    def wait_for_code(
        self,
        *,
        request_id: str,
        timeout_seconds: int,
        stop_event: Event | None = None,
    ) -> TaxMonitorVerificationSubmission | None:
        deadline = datetime.now(timezone.utc).timestamp() + max(int(timeout_seconds), 1)
        with self._condition:
            while True:
                if self._active_request is None or self._active_request.request_id != request_id:
                    logger.debug(
                        "Stopped waiting for tax monitor verification code because the request changed request_id=%s",
                        request_id,
                    )
                    return None
                if self._submission is not None and self._submission.request_id == request_id:
                    logger.info(
                        "Tax monitor verification wait completed request_id=%s channel=%s",
                        request_id,
                        self._submission.channel_id,
                    )
                    return self._submission
                if stop_event is not None and stop_event.is_set():
                    logger.warning(
                        "Stopped waiting for tax monitor verification code because the service is stopping request_id=%s",
                        request_id,
                    )
                    return None
                remaining = deadline - datetime.now(timezone.utc).timestamp()
                if remaining <= 0:
                    logger.warning(
                        "Timed out waiting for tax monitor verification code request_id=%s channel=%s",
                        request_id,
                        self.channel_id,
                    )
                    return None
                self._condition.wait(timeout=min(remaining, 1.0))

    def clear_request(self, *, request_id: str) -> None:
        with self._condition:
            if self._active_request is None or self._active_request.request_id != request_id:
                return
            self._active_request = None
            self._submission = None
            self._condition.notify_all()
        logger.debug("Cleared tax monitor verification request request_id=%s channel=%s", request_id, self.channel_id)


def _mask_code(code: str) -> str:
    normalized = str(code or "").strip()
    if len(normalized) != 6:
        return "***"
    return f"{normalized[:2]}****"
