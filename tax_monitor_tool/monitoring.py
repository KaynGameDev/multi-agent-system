from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Protocol

from core.config import Settings

logger = logging.getLogger(__name__)

_PERCENT_PATTERN = re.compile(r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*%")
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"[ \t\u00a0]+")
_LINEBREAK_PATTERN = re.compile(r"\r\n?|\n")

DEFAULT_NAVIGATION_PATH = ("税收调控管理", "税收详情（新）")
INITIAL_LOGIN_PAGE_SETTLE_MS = 5000
FOLDED_NAVIGATION_EXPAND_WAIT_MS = 2000
TAX_DETAIL_FRAME_URL_KEYWORDS = ("TaxDetailsNew.html", "/Page/TaxRegulation/TaxDetailsNew")
LOGIN_USERNAME_SELECTORS = (
    "input[name='account']",
    "input[name='username']",
    "input[name*='user' i]",
    "input[id='account']",
    "input[id='username']",
    "input[placeholder*='账号']",
    "input[placeholder*='用户名']",
    "input[placeholder*='account' i]",
    "input[placeholder*='user' i]",
    "input[type='text']",
)
LOGIN_PASSWORD_SELECTORS = (
    "input[type='password']",
    "input[name='password']",
    "input[name*='pass' i]",
    "input[id='password']",
    "input[placeholder*='密码']",
    "input[placeholder*='password' i]",
)
DEFAULT_TARGET_PROJECT_NAME_MAP: tuple[tuple[str, str], ...] = (
    ("4 Player Fishing", "四人捕鱼"),
    ("West Journey Fishing", "西游捕鱼"),
    ("2 Player Fishing", "二人捕鱼"),
    ("SkyFire Fishing", "飞机捕鱼"),
)
DEFAULT_TARGET_PROJECT_GAME_MAP = dict(DEFAULT_TARGET_PROJECT_NAME_MAP)

PROJECT_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "4 Player Fishing": ("4 Player Fishing", "四人捕鱼", "四人捕鱼税率"),
    "West Journey Fishing": ("West Journey Fishing", "西游捕鱼", "西游捕鱼税率"),
    "2 Player Fishing": ("2 Player Fishing", "二人捕鱼", "二人捕鱼税率"),
    "SkyFire Fishing": ("SkyFire Fishing", "飞机捕鱼", "飞机捕鱼税率"),
}
NAVIGATION_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "税收调控管理": ("税收调控管理",),
    "税收详情(新)": ("税收详情(新)", "税收详情（新）", "税收详情"),
    "税收详情（新）": ("税收详情（新）", "税收详情(新)", "税收详情"),
}


@dataclass(frozen=True)
class TaxGameReading:
    game_name: str
    current_rate_percent: float
    matched_text: str


@dataclass(frozen=True)
class TaxProjectSnapshot:
    project_keyword: str
    game_readings: dict[str, TaxGameReading]
    raw_excerpt: str = ""


@dataclass(frozen=True)
class TaxMonitorSnapshot:
    observed_at: str
    source_url: str
    project_snapshots: dict[str, TaxProjectSnapshot]
    project_errors: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaxMonitorAlert:
    category: str
    dedupe_key: str
    message: str
    project_keyword: str = ""
    game_name: str = ""
    rate_percent: float | None = None


@dataclass
class TaxMonitorState:
    negative_alert_days: dict[str, str] = field(default_factory=dict)
    threshold_alert_timestamps: dict[str, str] = field(default_factory=dict)
    error_alert_timestamps: dict[str, str] = field(default_factory=dict)


class TaxPageClient(Protocol):
    def fetch_snapshot(self) -> TaxMonitorSnapshot:
        ...


class VerificationCodeProvider(Protocol):
    def request_code(self, *, prompt: str) -> str:
        ...


class TaxMonitorStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> TaxMonitorState:
        if not self.path.exists():
            return TaxMonitorState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load tax monitor state path=%s", self.path, exc_info=True)
            return TaxMonitorState()
        if not isinstance(payload, dict):
            return TaxMonitorState()
        return TaxMonitorState(
            negative_alert_days=_clean_string_map(payload.get("negative_alert_days")),
            threshold_alert_timestamps=_clean_string_map(payload.get("threshold_alert_timestamps")),
            error_alert_timestamps=_clean_string_map(payload.get("error_alert_timestamps")),
        )

    def save(self, state: TaxMonitorState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


@dataclass(frozen=True)
class ThresholdRule:
    game_name: str
    low_percent: float
    high_percent: float

    def is_abnormal(self, current_rate_percent: float) -> bool:
        return current_rate_percent < self.low_percent or current_rate_percent > self.high_percent


DEFAULT_THRESHOLD_RULES = (
    ThresholdRule(game_name="二人捕鱼", low_percent=0.5, high_percent=0.7),
    ThresholdRule(game_name="四人捕鱼", low_percent=0.5, high_percent=0.7),
    ThresholdRule(game_name="飞机捕鱼", low_percent=0.5, high_percent=0.7),
    ThresholdRule(game_name="西游捕鱼", low_percent=0.5, high_percent=0.7),
)
DEFAULT_THRESHOLD_RULES_BY_GAME = {rule.game_name: rule for rule in DEFAULT_THRESHOLD_RULES}


class PlaywrightTaxPageClient:
    def __init__(
        self,
        settings: Settings,
        *,
        verification_code_provider: VerificationCodeProvider | None = None,
    ) -> None:
        self.settings = settings
        self.url = settings.tax_monitor_url.strip()
        self.username = settings.tax_monitor_username.strip()
        self.password = settings.tax_monitor_password
        self.token = settings.tax_monitor_token
        self.capture_group = settings.tax_monitor_capture_group.strip()
        self.verification_code_provider = verification_code_provider
        self.timeout_ms = max(int(settings.tax_monitor_browser_timeout_seconds), 1) * 1000
        self.headless = settings.tax_monitor_headless
        self.navigation_path = tuple(
            item.strip()
            for item in settings.tax_monitor_navigation_path
            if str(item).strip()
        ) or DEFAULT_NAVIGATION_PATH

    def fetch_snapshot(self) -> TaxMonitorSnapshot:
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for TAX_MONITOR_ENABLED. Install dependencies and run "
                "`playwright install chromium`."
            ) from exc

        snapshot_time = datetime.now(timezone.utc)

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=self.headless)
                context = browser.new_context(ignore_https_errors=True)
                page = context.new_page()
                try:
                    page.goto(self.url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    self._login(page)
                    self._navigate_to_tax_detail(page)
                    content_root = self._resolve_tax_detail_context(page)
                    self._select_capture_group(content_root)
                    self._refresh_results(content_root)
                    body_text = content_root.locator("body").inner_text(timeout=self.timeout_ms)
                finally:
                    context.close()
                    browser.close()
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Timed out while loading the tax monitor page: {exc}") from exc

        return parse_tax_snapshot_from_text(
            body_text,
            source_url=self.url,
            observed_at=snapshot_time,
        )

    def _login(self, page) -> None:
        page.wait_for_timeout(INITIAL_LOGIN_PAGE_SETTLE_MS)
        self._wait_for_login_form(page)
        for attempt in range(3):
            if attempt > 0 and not self._has_login_fields(page):
                logger.info(
                    "Tax monitor login form is no longer visible before retry attempt=%s; checking authenticated shell instead",
                    attempt + 1,
                )
                if self._wait_for_authenticated_shell(page):
                    logger.info(
                        "Tax monitor login succeeded while waiting for the authenticated shell attempt=%s",
                        attempt + 1,
                    )
                    return
                raise RuntimeError(
                    "Tax monitor login form disappeared, but the authenticated page shell did not become ready."
                )
            logger.info("Tax monitor login attempt started attempt=%s", attempt + 1)
            self._fill_value(
                page,
                value=self.username,
                explicit_selector=self.settings.tax_monitor_username_selector,
                candidate_selectors=LOGIN_USERNAME_SELECTORS,
                field_name="username",
            )
            self._fill_value(
                page,
                value=self.password,
                explicit_selector=self.settings.tax_monitor_password_selector,
                candidate_selectors=LOGIN_PASSWORD_SELECTORS,
                field_name="password",
            )
            self._fill_verification_code(page, attempt=attempt)
            self._click(
                page,
                explicit_selector=self.settings.tax_monitor_login_button_selector,
                candidate_button_texts=("登录", "登 录", "登入", "Login", "Sign in"),
            )
            logger.info("Tax monitor submitted login form attempt=%s", attempt + 1)
            if self._wait_for_authenticated_shell(page):
                logger.info("Tax monitor login succeeded attempt=%s", attempt + 1)
                return
            logger.warning("Tax monitor login did not reach the authenticated shell attempt=%s", attempt + 1)
            page.wait_for_timeout(600)

        raise RuntimeError("Tax monitor login did not complete. The verification code may be invalid or expired.")

    def _wait_for_login_form(self, page) -> None:
        max_attempts = max(self.timeout_ms // 500, 1)
        for _ in range(max_attempts):
            if self._has_login_fields(page):
                return
            page.wait_for_timeout(500)
        raise RuntimeError(
            "Tax monitor login page did not load the username/password fields. "
            "Provide explicit login selectors if the page uses custom controls."
        )

    def _wait_for_authenticated_shell(self, page) -> bool:
        try:
            page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 5000))
        except Exception:
            pass

        max_attempts = max(self.timeout_ms // 500, 1)
        for _ in range(max_attempts):
            if self._is_authenticated_shell(page):
                return True
            if self._has_login_fields(page):
                return False
            page.wait_for_timeout(500)
        return self._is_authenticated_shell(page)

    def _has_login_fields(self, page) -> bool:
        return bool(self._visible_locators(page, LOGIN_PASSWORD_SELECTORS))

    def _is_authenticated_shell(self, page) -> bool:
        checks = (
            page.get_by_text("税收调控管理", exact=False),
            page.get_by_text("实时数据", exact=False),
            page.get_by_text("概况数据", exact=False),
            page.get_by_text("税收详情", exact=False),
            page.get_by_text("欢迎", exact=False),
            page.locator("input[placeholder*='搜索系统功能']"),
        )
        for locator in checks:
            try:
                if locator.count() and locator.first.is_visible():
                    return True
            except Exception:
                continue
        return False

    def _fill_verification_code(self, page, *, attempt: int, override_code: str = "") -> bool:
        verification_locator = self._resolve_optional_locator(
            page,
            explicit_selector=self.settings.tax_monitor_token_selector,
            candidate_selectors=(
                "input[name='token']",
                "input[name*='token' i]",
                "input[id='token']",
                "input[placeholder*='令牌']",
                "input[placeholder*='动态口令']",
                "input[placeholder*='验证码']",
                "input[placeholder*='6位']",
                "input[placeholder*='token' i]",
            ),
        )
        if verification_locator is None:
            logger.debug("Tax monitor login did not detect a verification-code input attempt=%s", attempt + 1)
            return False

        logger.info("Tax monitor login detected a verification-code input attempt=%s", attempt + 1)
        code = override_code.strip() or self.token.strip()
        if not code:
            if self.verification_code_provider is None:
                raise RuntimeError(
                    "Tax monitor login requires a verification code, but no Slack verification flow is available."
                )
            prompt = (
                "税率监控登录需要 6 位动态口令。"
                "请直接在当前 Slack 频道回复 6 位验证码。"
            )
            if attempt > 0:
                prompt = (
                    "上一条税率监控验证码无效或已过期。"
                    "请在当前 Slack 频道重新回复最新的 6 位验证码。"
                )
            logger.info("Tax monitor login is requesting a verification code from Slack attempt=%s", attempt + 1)
            code = self.verification_code_provider.request_code(prompt=prompt).strip()
            logger.info("Tax monitor login received a verification code from Slack attempt=%s", attempt + 1)

        verification_locator.fill(code, timeout=self.timeout_ms)
        logger.debug("Tax monitor login filled the verification-code input attempt=%s", attempt + 1)
        return True

    def _navigate_to_tax_detail(self, page) -> None:
        if not self._wait_for_authenticated_shell(page):
            raise RuntimeError("Tax monitor did not reach the authenticated page shell after login.")
        for index, item in enumerate(self.navigation_path):
            self._click_navigation_item(page, item)
            if index < len(self.navigation_path) - 1:
                page.wait_for_timeout(FOLDED_NAVIGATION_EXPAND_WAIT_MS)
            else:
                page.wait_for_timeout(400)

    def _resolve_tax_detail_context(self, page):
        frames = list(getattr(page, "frames", []) or [])
        for frame in frames:
            frame_url = str(getattr(frame, "url", "") or "")
            if any(keyword in frame_url for keyword in TAX_DETAIL_FRAME_URL_KEYWORDS):
                logger.info("Using tax detail iframe url=%s", frame_url)
                return frame
        logger.info("Tax detail iframe not found; using the main page context instead.")
        return page

    def _click_navigation_item(self, page, item: str) -> None:
        aliases = NAVIGATION_NAME_ALIASES.get(item, (item,))
        for label in aliases:
            if self._try_click_visible_text(page, label):
                return
        raise RuntimeError(
            "Unable to navigate to the tax detail page. Missing menu item: " + ", ".join(aliases)
        )

    def _select_capture_group(self, page) -> None:
        if not self.capture_group:
            return

        strategies = (
            ("explicit selector", lambda: self._try_select_capture_group_with_explicit_selector(page)),
            (
                "query type select",
                lambda: self._try_set_visible_control(
                    page,
                    ("select#QueryType", "select[name='QueryType']", "#QueryType"),
                    0,
                    self.capture_group,
                ),
            ),
            ("labeled control", lambda: self._try_select_capture_group_near_labels(page, ("捕获组", "捕获组名", "组"))),
            (
                "second visible select",
                lambda: self._try_set_visible_control(page, ("select", "[role='combobox']"), 1, self.capture_group),
            ),
            (
                "first visible select",
                lambda: self._try_set_visible_control(page, ("select", "[role='combobox']"), 0, self.capture_group),
            ),
            (
                "group-like input",
                lambda: self._try_set_visible_control(
                    page,
                    (
                        "input[name*='group' i]",
                        "input[id*='group' i]",
                        "input[placeholder*='组']",
                        "input[placeholder*='group' i]",
                    ),
                    0,
                    self.capture_group,
                ),
            ),
            ("capture group text", lambda: self._try_click_value_text(page, self.capture_group)),
        )

        for description, strategy in strategies:
            try:
                if strategy():
                    logger.info(
                        "Selected tax monitor capture group strategy=%s value=%s",
                        description,
                        self.capture_group,
                    )
                    return
            except Exception:
                logger.debug(
                    "Tax monitor capture group strategy failed strategy=%s value=%s",
                    description,
                    self.capture_group,
                    exc_info=True,
                )

        logger.warning(
            "Unable to locate a capture group control automatically value=%s visible_controls=%s",
            self.capture_group,
            self._summarize_visible_controls(page),
        )
        raise RuntimeError(
            "Unable to locate a capture group control on the tax detail page. "
            "Provide TAX_MONITOR_CAPTURE_GROUP_SELECTOR if the page uses a custom widget."
        )

    def _refresh_results(self, page) -> None:
        clicked = self._try_click(
            page,
            explicit_selector=self.settings.tax_monitor_query_button_selector,
            candidate_button_texts=("查询",),
        )
        if clicked:
            page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        page.wait_for_timeout(1200)

    def _fill_value(
        self,
        page,
        *,
        value: str,
        explicit_selector: str,
        candidate_selectors: tuple[str, ...],
        required: bool = True,
        field_name: str = "input",
    ) -> None:
        if not value:
            if required:
                raise RuntimeError(f"Missing a required monitor {field_name} value.")
            return

        if explicit_selector:
            locator = page.locator(explicit_selector)
            if locator.count() == 0:
                raise RuntimeError(f"Tax monitor selector did not match any elements: {explicit_selector}")
            locator.first.fill(value, timeout=self.timeout_ms)
            return

        visible_locators = self._visible_locators(page, candidate_selectors)
        if visible_locators:
            visible_locators[0].fill(value, timeout=self.timeout_ms)
            return

        if required:
            logger.warning(
                "Unable to locate tax monitor field field=%s selectors=%s visible_controls=%s",
                field_name,
                ", ".join(candidate_selectors),
                self._summarize_visible_controls(page),
            )
            raise RuntimeError(
                f"Unable to find the tax monitor {field_name} input on the current page. "
                "Provide an explicit selector in environment configuration."
            )

    def _resolve_optional_locator(
        self,
        page,
        *,
        explicit_selector: str,
        candidate_selectors: tuple[str, ...],
    ):
        if explicit_selector:
            locator = page.locator(explicit_selector)
            if locator.count() == 0:
                return None
            return locator.first

        visible_locators = self._visible_locators(page, candidate_selectors)
        if visible_locators:
            return visible_locators[0]
        return None

    def _click(
        self,
        page,
        *,
        explicit_selector: str = "",
        candidate_button_texts: tuple[str, ...],
    ) -> None:
        if not self._try_click(
            page,
            explicit_selector=explicit_selector,
            candidate_button_texts=candidate_button_texts,
        ):
            raise RuntimeError(
                "Unable to find the expected control on the tax monitor page: "
                + ", ".join(candidate_button_texts)
            )

    def _try_click(
        self,
        page,
        *,
        explicit_selector: str = "",
        candidate_button_texts: tuple[str, ...],
    ) -> bool:
        if explicit_selector:
            locator = page.locator(explicit_selector)
            if locator.count() == 0:
                return False
            locator.first.click(timeout=self.timeout_ms)
            return True

        for button_text in candidate_button_texts:
            button = page.get_by_role("button", name=button_text)
            if button.count():
                button.first.click(timeout=self.timeout_ms)
                return True
            link = page.get_by_role("link", name=button_text)
            if link.count():
                link.first.click(timeout=self.timeout_ms)
                return True
            text = page.get_by_text(button_text, exact=True)
            if text.count():
                text.first.click(timeout=self.timeout_ms)
                return True
            fuzzy_text = page.get_by_text(button_text, exact=False)
            if fuzzy_text.count():
                fuzzy_text.first.click(timeout=self.timeout_ms)
                return True
        return False

    def _try_click_visible_text(self, page, label: str) -> bool:
        for selector in ("[class*='sidebar']", "[class*='menu']", ".left"):
            area = page.locator(selector)
            if area.count() == 0:
                continue
            text = area.first.get_by_text(label, exact=False)
            if text.count() == 0:
                continue
            try:
                text.first.click(timeout=self.timeout_ms)
                return True
            except Exception:
                continue

        text = page.get_by_text(label, exact=False)
        if text.count() == 0:
            return False
        try:
            text.first.click(timeout=self.timeout_ms)
            return True
        except Exception:
            return False

    def _try_select_capture_group_with_explicit_selector(self, page) -> bool:
        selector = self.settings.tax_monitor_capture_group_selector
        if not selector:
            return False
        locator = page.locator(selector)
        if locator.count() == 0:
            logger.warning("Tax monitor capture group selector matched no elements selector=%s", selector)
            return False
        return self._try_set_locator_value(locator.first, self.capture_group)

    def _try_select_capture_group_near_labels(self, page, labels: tuple[str, ...]) -> bool:
        descendant_selectors = ("select", "[role='combobox']", "input", "textarea")
        for label in labels:
            label_locator = page.get_by_text(label, exact=False)
            count = min(label_locator.count(), 5)
            for index in range(count):
                candidate = label_locator.nth(index)
                try:
                    if not candidate.is_visible():
                        continue
                except Exception:
                    continue

                if self._try_set_locator_value_from_relative(
                    candidate,
                    "xpath=following::*[self::select or self::input or self::textarea or @role='combobox'][1]",
                    self.capture_group,
                ):
                    return True

                for depth in range(1, 5):
                    container = candidate.locator(f"xpath=ancestor::*[{depth}]")
                    if self._try_set_first_descendant_value(container, descendant_selectors, self.capture_group):
                        return True
        return False

    def _try_set_visible_control(
        self,
        page,
        selectors: tuple[str, ...],
        preferred_index: int,
        value: str,
    ) -> bool:
        visible_locators = self._visible_locators(page, selectors)
        if preferred_index >= len(visible_locators):
            return False
        return self._try_set_locator_value(visible_locators[preferred_index], value)

    def _try_click_value_text(self, page, value: str) -> bool:
        locator = page.get_by_text(value, exact=False)
        count = min(locator.count(), 6)
        for index in range(count):
            candidate = locator.nth(index)
            try:
                if not candidate.is_visible():
                    continue
                candidate.click(timeout=self.timeout_ms)
                return True
            except Exception:
                continue
        return False

    def _try_set_locator_value_from_relative(self, locator, relative_selector: str, value: str) -> bool:
        try:
            target = locator.locator(relative_selector)
        except Exception:
            return False
        count = min(target.count(), 3)
        for index in range(count):
            if self._try_set_locator_value(target.nth(index), value):
                return True
        return False

    def _try_set_first_descendant_value(self, locator, selectors: tuple[str, ...], value: str) -> bool:
        for selector in selectors:
            try:
                target = locator.locator(selector)
            except Exception:
                continue
            count = min(target.count(), 4)
            for index in range(count):
                if self._try_set_locator_value(target.nth(index), value):
                    return True
        return False

    def _try_set_locator_value(self, locator, value: str) -> bool:
        try:
            self._set_locator_value(locator, value)
            return True
        except Exception:
            return False

    def _set_locator_value(self, locator, value: str) -> None:
        tag_name = str(locator.evaluate("element => element.tagName")).lower()
        if tag_name == "select":
            try:
                locator.select_option(label=value, timeout=self.timeout_ms)
                return
            except Exception:
                locator.select_option(value=value, timeout=self.timeout_ms)
                return
        locator.click(timeout=self.timeout_ms)
        locator.fill(value, timeout=self.timeout_ms)
        locator.press("Enter", timeout=self.timeout_ms)

    def _visible_locators(self, page, selectors: tuple[str, ...]) -> list:
        visible = []
        for selector in selectors:
            locator = page.locator(selector)
            count = min(locator.count(), 8)
            for index in range(count):
                candidate = locator.nth(index)
                try:
                    if candidate.is_visible():
                        visible.append(candidate)
                except Exception:
                    continue
        return visible

    def _summarize_visible_controls(self, page) -> str:
        summaries: list[str] = []
        selectors = ("select", "[role='combobox']", "input", "button", "a")
        for selector in selectors:
            locator = page.locator(selector)
            count = min(locator.count(), 5)
            for index in range(count):
                candidate = locator.nth(index)
                try:
                    if not candidate.is_visible():
                        continue
                    tag_name = str(candidate.evaluate("element => element.tagName")).lower()
                    name = str(candidate.get_attribute("name") or "").strip()
                    placeholder = str(candidate.get_attribute("placeholder") or "").strip()
                    text_lines = normalize_page_text(str(candidate.inner_text(timeout=500) or ""))
                    text_preview = text_lines[0] if text_lines else ""
                    summaries.append(
                        f"{tag_name}[name={name or '-'}, placeholder={placeholder or '-'}, text={text_preview or '-'}]"
                    )
                except Exception:
                    continue
        return "; ".join(summaries[:12]) or "none"


def build_tax_monitor_snapshot(
    *,
    source_url: str,
    project_snapshots: dict[str, TaxProjectSnapshot],
    project_errors: dict[str, str] | None = None,
    observed_at: datetime | None = None,
) -> TaxMonitorSnapshot:
    snapshot_time = observed_at or datetime.now(timezone.utc)
    return TaxMonitorSnapshot(
        observed_at=snapshot_time.astimezone(timezone.utc).isoformat(),
        source_url=source_url,
        project_snapshots=dict(project_snapshots),
        project_errors=dict(project_errors or {}),
    )


def parse_tax_project_snapshot_from_text(
    raw_text: str,
    *,
    source_url: str,
    project_keyword: str,
    game_name: str = "",
    observed_at: datetime | None = None,
) -> TaxProjectSnapshot:
    del source_url
    del observed_at

    canonical_game_name = game_name or DEFAULT_TARGET_PROJECT_GAME_MAP.get(project_keyword, "")
    if not canonical_game_name:
        raise ValueError(f"Unknown tax monitor project: {project_keyword}")

    normalized_lines = normalize_page_text(raw_text)
    joined_text = "\n".join(normalized_lines)
    matched_text, rate_percent = _extract_project_rate(
        normalized_lines,
        joined_text,
        project_keyword=project_keyword,
        game_name=canonical_game_name,
    )
    if matched_text is None or rate_percent is None:
        raise ValueError(f"Missing tax rate value for project {project_keyword}")

    return TaxProjectSnapshot(
        project_keyword=project_keyword,
        game_readings={
            canonical_game_name: TaxGameReading(
                game_name=canonical_game_name,
                current_rate_percent=rate_percent,
                matched_text=matched_text,
            )
        },
        raw_excerpt=matched_text[:1000],
    )


def parse_tax_snapshot_from_text(
    raw_text: str,
    *,
    source_url: str,
    observed_at: datetime | None = None,
) -> TaxMonitorSnapshot:
    project_snapshots: dict[str, TaxProjectSnapshot] = {}
    project_errors: dict[str, str] = {}

    for project_keyword, canonical_game_name in DEFAULT_TARGET_PROJECT_NAME_MAP:
        try:
            project_snapshots[project_keyword] = parse_tax_project_snapshot_from_text(
                raw_text,
                source_url=source_url,
                project_keyword=project_keyword,
                game_name=canonical_game_name,
                observed_at=observed_at,
            )
        except ValueError as exc:
            project_errors[project_keyword] = str(exc)

    return build_tax_monitor_snapshot(
        source_url=source_url,
        project_snapshots=project_snapshots,
        project_errors=project_errors,
        observed_at=observed_at,
    )


def normalize_page_text(raw_text: str) -> list[str]:
    text = unescape(raw_text or "")
    text = _TAG_PATTERN.sub(" ", text)
    text = _LINEBREAK_PATTERN.sub("\n", text)
    normalized_lines: list[str] = []
    for raw_line in text.split("\n"):
        cleaned = _WHITESPACE_PATTERN.sub(" ", raw_line).strip()
        if cleaned:
            normalized_lines.append(cleaned)
    return normalized_lines


def evaluate_tax_snapshot(
    snapshot: TaxMonitorSnapshot,
    *,
    state: TaxMonitorState,
    now: datetime,
    alert_cooldown_seconds: int,
) -> list[TaxMonitorAlert]:
    normalized_now = _normalize_datetime(now)
    today_key = normalized_now.date().isoformat()
    alerts: list[TaxMonitorAlert] = []

    for project_keyword, project_snapshot in snapshot.project_snapshots.items():
        for game_name, reading in project_snapshot.game_readings.items():
            scope_key = _build_project_game_key(project_keyword, game_name)
            if reading.current_rate_percent < 0 and state.negative_alert_days.get(scope_key) != today_key:
                state.negative_alert_days[scope_key] = today_key
                alerts.append(
                    TaxMonitorAlert(
                        category="negative-rate",
                        dedupe_key=f"negative-rate:{scope_key}:{today_key}",
                        message=(
                            f"税率预警：项目 {project_keyword} 的 {game_name} "
                            f"当日首次出现负税率，当前税率 {format_percent(reading.current_rate_percent)}。"
                        ),
                        project_keyword=project_keyword,
                        game_name=game_name,
                        rate_percent=reading.current_rate_percent,
                    )
                )

            rule = DEFAULT_THRESHOLD_RULES_BY_GAME.get(game_name)
            if rule is None or not rule.is_abnormal(reading.current_rate_percent):
                continue

            if not _is_alert_due(
                state.threshold_alert_timestamps.get(scope_key),
                now=normalized_now,
                cooldown_seconds=alert_cooldown_seconds,
            ):
                continue
            state.threshold_alert_timestamps[scope_key] = normalized_now.isoformat()
            alerts.append(
                TaxMonitorAlert(
                    category="threshold",
                    dedupe_key=f"threshold:{scope_key}",
                    message=(
                        f"税率预警：项目 {project_keyword} 的 {game_name} 当前税率 "
                        f"{format_percent(reading.current_rate_percent)}，超出阈值范围 "
                        f"{format_percent(rule.low_percent)} - {format_percent(rule.high_percent)}。"
                    ),
                    project_keyword=project_keyword,
                    game_name=game_name,
                    rate_percent=reading.current_rate_percent,
                )
            )

    return alerts


def build_monitor_error_alert(
    *,
    state: TaxMonitorState,
    now: datetime,
    error_message: str,
    category: str = "fetch",
    cooldown_seconds: int,
) -> TaxMonitorAlert | None:
    normalized_now = _normalize_datetime(now)
    if not _is_alert_due(
        state.error_alert_timestamps.get(category),
        now=normalized_now,
        cooldown_seconds=cooldown_seconds,
    ):
        return None
    state.error_alert_timestamps[category] = normalized_now.isoformat()
    return TaxMonitorAlert(
        category="monitor-error",
        dedupe_key=f"monitor-error:{category}",
        message=f"税率监控异常：{error_message}",
    )


def format_percent(value: float) -> str:
    return f"{value:.3f}%"


def _extract_project_rate(
    normalized_lines: list[str],
    joined_text: str,
    *,
    project_keyword: str,
    game_name: str,
) -> tuple[str | None, float | None]:
    alias_candidates = PROJECT_NAME_ALIASES.get(
        project_keyword,
        (project_keyword, game_name, f"{game_name}税率"),
    )

    for line in normalized_lines:
        if any(_line_contains_alias(line, alias) for alias in alias_candidates):
            percent_match = _PERCENT_PATTERN.search(line)
            if percent_match:
                return line, float(percent_match.group("value"))

    for alias in alias_candidates:
        pattern = re.compile(
            rf"({re.escape(alias)}[\s\S]{{0,200}}?(?P<value>[+-]?\d+(?:\.\d+)?)\s*%)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(joined_text)
        if match:
            return match.group(1), float(match.group("value"))

    return None, None


def _line_contains_alias(line: str, alias: str) -> bool:
    if any("A" <= char <= "z" for char in alias):
        return alias.lower() in line.lower()
    return alias in line


def _clean_string_map(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, raw in value.items():
        normalized_key = str(key).strip()
        normalized_value = str(raw).strip()
        if normalized_key and normalized_value:
            cleaned[normalized_key] = normalized_value
    return cleaned


def _build_project_game_key(project_keyword: str, game_name: str) -> str:
    return f"{project_keyword}|||{game_name}"


def _is_alert_due(last_sent_at: str | None, *, now: datetime, cooldown_seconds: int) -> bool:
    if not last_sent_at:
        return True
    try:
        last_sent = _normalize_datetime(datetime.fromisoformat(last_sent_at))
    except ValueError:
        return True
    return (now - last_sent).total_seconds() >= cooldown_seconds


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
