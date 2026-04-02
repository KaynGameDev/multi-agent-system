from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from core.config import Settings, load_settings, validate_tax_monitor_settings
from tax_monitor_tool.service import TaxMonitorService
from tax_monitor_tool.verification import TaxMonitorVerificationBroker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaxMonitorRuntime:
    settings: Settings
    service: TaxMonitorService
    verification_broker: TaxMonitorVerificationBroker | None


def load_tool_settings(*, force_reload: bool = False, use_dotenv: bool = True) -> Settings:
    if use_dotenv:
        load_dotenv()
    settings = load_settings(force_reload=force_reload)
    _ensure_tax_monitor_enabled(settings)
    validate_tax_monitor_settings(settings)
    return settings


def build_runtime(
    settings: Settings | None = None,
    *,
    force_reload: bool = False,
    use_dotenv: bool = True,
) -> TaxMonitorRuntime:
    resolved_settings = settings or load_tool_settings(force_reload=force_reload, use_dotenv=use_dotenv)
    _ensure_tax_monitor_enabled(resolved_settings)
    validate_tax_monitor_settings(resolved_settings)
    verification_broker = TaxMonitorVerificationBroker(
        channel_id=resolved_settings.tax_monitor_slack_channel,
    )
    service = TaxMonitorService(
        settings=resolved_settings,
        verification_broker=verification_broker,
    )
    return TaxMonitorRuntime(
        settings=resolved_settings,
        service=service,
        verification_broker=verification_broker,
    )


def run_once(
    settings: Settings | None = None,
    *,
    force_reload: bool = False,
    use_dotenv: bool = True,
) -> TaxMonitorRuntime:
    runtime = build_runtime(settings=settings, force_reload=force_reload, use_dotenv=use_dotenv)
    logger.info("Running tax monitor tool once.")
    runtime.service.run_cycle()
    return runtime


def run_forever(
    settings: Settings | None = None,
    *,
    force_reload: bool = False,
    use_dotenv: bool = True,
) -> TaxMonitorRuntime:
    runtime = build_runtime(settings=settings, force_reload=force_reload, use_dotenv=use_dotenv)
    logger.info(
        "Starting tax monitor tool loop interval_seconds=%s",
        runtime.settings.tax_monitor_poll_interval_seconds,
    )
    runtime.service.start()
    return runtime


def configure_tool_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)


def _ensure_tax_monitor_enabled(settings: Settings) -> None:
    if settings.tax_monitor_enabled:
        return
    raise RuntimeError(
        "TAX_MONITOR_ENABLED must be true when running the standalone tax monitor tool."
    )
