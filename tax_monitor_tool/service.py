from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

from slack_sdk import WebClient

from app.config import Settings
from interfaces.slack.formatting import to_slack_mrkdwn
from tax_monitor_tool.monitoring import (
    PlaywrightTaxPageClient,
    TaxMonitorAlert,
    TaxMonitorStateStore,
    build_monitor_error_alert,
    evaluate_tax_snapshot,
)
from tax_monitor_tool.verification import TaxMonitorVerificationBroker

logger = logging.getLogger(__name__)
DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 180


class SlackAlertPublisher:
    def __init__(self, *, slack_bot_token: str, channel: str) -> None:
        self.channel = channel.strip()
        self.client = WebClient(token=slack_bot_token)

    def publish(self, alert: TaxMonitorAlert) -> None:
        self.client.chat_postMessage(
            channel=self.channel,
            text=to_slack_mrkdwn(alert.message),
        )


class SlackVerificationCodeProvider:
    def __init__(
        self,
        *,
        slack_bot_token: str,
        channel: str,
        broker: TaxMonitorVerificationBroker,
        stop_event: Event,
        timeout_seconds: int = DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
    ) -> None:
        self.channel = channel.strip()
        self.broker = broker
        self.stop_event = stop_event
        self.timeout_seconds = max(int(timeout_seconds), 1)
        self.client = WebClient(token=slack_bot_token)

    def request_code(self, *, prompt: str) -> str:
        request = self.broker.open_request(prompt=prompt)
        logger.info(
            "Posting tax monitor verification prompt to Slack request_id=%s channel=%s timeout_seconds=%s",
            request.request_id,
            self.channel,
            self.timeout_seconds,
        )
        response = self.client.chat_postMessage(
            channel=self.channel,
            text=to_slack_mrkdwn(self._format_prompt(prompt)),
        )
        logger.debug(
            "Posted tax monitor verification prompt to Slack request_id=%s channel=%s message_ts=%s",
            request.request_id,
            self.channel,
            str(response.get("ts", "")).strip(),
        )
        self.broker.record_request_message(
            request_id=request.request_id,
            message_ts=str(response.get("ts", "")).strip(),
        )
        logger.info(
            "Waiting for tax monitor verification code from Slack request_id=%s channel=%s",
            request.request_id,
            self.channel,
        )
        submission = self.broker.wait_for_code(
            request_id=request.request_id,
            timeout_seconds=self.timeout_seconds,
            stop_event=self.stop_event,
        )
        self.broker.clear_request(request_id=request.request_id)
        if submission is None:
            logger.warning(
                "Did not receive a tax monitor verification code from Slack request_id=%s channel=%s",
                request.request_id,
                self.channel,
            )
            raise RuntimeError("Timed out waiting for the tax monitor verification code from Slack.")
        logger.info(
            "Received tax monitor verification code from Slack request_id=%s channel=%s user=%s",
            request.request_id,
            self.channel,
            submission.user_id,
        )
        return submission.code

    def _format_prompt(self, prompt: str) -> str:
        cleaned_prompt = str(prompt or "").strip()
        if cleaned_prompt.startswith("<!channel>"):
            return cleaned_prompt
        return f"<!channel> {cleaned_prompt}"


class TaxMonitorService:
    def __init__(
        self,
        settings: Settings,
        *,
        page_client=None,
        publisher: SlackAlertPublisher | None = None,
        state_store: TaxMonitorStateStore | None = None,
        verification_broker: TaxMonitorVerificationBroker | None = None,
        verification_code_provider: SlackVerificationCodeProvider | None = None,
        now_fn=None,
    ) -> None:
        self.settings = settings
        self._stop_event = Event()
        self.verification_broker = verification_broker
        self.verification_code_provider = verification_code_provider
        self.publisher = publisher or SlackAlertPublisher(
            slack_bot_token=settings.slack_bot_token,
            channel=settings.tax_monitor_slack_channel,
        )
        self.state_store = state_store or TaxMonitorStateStore(
            self._resolve_path(settings.tax_monitor_state_path)
        )
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        if self.verification_code_provider is None and self.verification_broker is not None:
            self.verification_code_provider = SlackVerificationCodeProvider(
                slack_bot_token=settings.slack_bot_token,
                channel=settings.tax_monitor_slack_channel,
                broker=self.verification_broker,
                stop_event=self._stop_event,
            )
        self.page_client = page_client or PlaywrightTaxPageClient(
            settings,
            verification_code_provider=self.verification_code_provider,
        )

    def start(self) -> None:
        logger.info(
            "Starting tax monitor service url=%s channel=%s interval_seconds=%s",
            self.settings.tax_monitor_url,
            self.settings.tax_monitor_slack_channel,
            self.settings.tax_monitor_poll_interval_seconds,
        )
        while not self._stop_event.is_set():
            self.run_cycle()
            self._stop_event.wait(max(int(self.settings.tax_monitor_poll_interval_seconds), 1))

    def stop(self) -> None:
        self._stop_event.set()

    def run_cycle(self) -> None:
        state = self.state_store.load()
        now = self._now_fn()
        try:
            snapshot = self.page_client.fetch_snapshot()
            alerts = evaluate_tax_snapshot(
                snapshot,
                state=state,
                now=now,
                alert_cooldown_seconds=max(int(self.settings.tax_monitor_alert_cooldown_seconds), 1),
            )
        except Exception as exc:
            logger.exception("Tax monitor cycle failed url=%s", self.settings.tax_monitor_url)
            alert = build_monitor_error_alert(
                state=state,
                now=now,
                error_message=str(exc),
                cooldown_seconds=max(int(self.settings.tax_monitor_error_cooldown_seconds), 1),
            )
            if alert is None:
                return
            self.publisher.publish(alert)
            self.state_store.save(state)
            return

        for project_keyword, error_message in snapshot.project_errors.items():
            alert = build_monitor_error_alert(
                state=state,
                now=now,
                error_message=f"项目 {project_keyword} 抓取失败：{error_message}",
                category=f"project:{project_keyword}",
                cooldown_seconds=max(int(self.settings.tax_monitor_error_cooldown_seconds), 1),
            )
            if alert is not None:
                alerts.append(alert)

        if not alerts:
            logger.debug("Tax monitor cycle completed without alerts url=%s", self.settings.tax_monitor_url)
            return

        for alert in alerts:
            self.publisher.publish(alert)
        self.state_store.save(state)
        logger.info("Published tax monitor alerts count=%s", len(alerts))

    def _resolve_path(self, configured_value: str) -> Path:
        configured_path = Path(configured_value or self.settings.tax_monitor_state_path).expanduser()
        if configured_path.is_absolute():
            return configured_path.resolve()
        return (Path(__file__).resolve().parent.parent / configured_path).resolve()
