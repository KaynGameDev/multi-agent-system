from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_core.tools import tool

from core.config import load_settings
from core.identity_map import normalize_sheet_identity

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

ITERATION_ALIASES = ("迭代", "iteration")
PERSON_ALIASES = ("人员", "assignee", "owner", "developer", "pic", "person_in_charge")
CONTENT_ALIASES = ("内容", "title", "feature", "name", "summary", "description")
PLATFORM_ALIASES = ("平台", "platform")
PROJECT_ALIASES = ("项目", "project", "initiative", "epic")
START_DATE_ALIASES = ("start", "开始", "开始时间")
END_DATE_ALIASES = ("end", "结束", "结束时间")
SUBMIT_TEST_DATE_ALIASES = ("提测日期", "test_date", "qa_date")
UPDATE_DATE_ALIASES = ("更新日期", "update_date", "updated_at")
COLOR_ALIASES = ("color", "颜色")
DEV_DAYS_ALIASES = ("开发天数", "dev_days")
TEST_DAYS_ALIASES = ("测试天数", "test_days", "qa_days")
CLIENT_ALIASES = ("客户端", "client")
SERVER_ALIASES = ("服务器", "server")
TEST_OWNER_ALIASES = ("测试", "qa", "tester")
PRODUCT_OWNER_ALIASES = ("产品", "product", "pm")
PRIORITY_ALIASES = ("优先级", "priority")
DATE_FIELD_NAMES = ("start_date", "end_date", "submit_test_date", "updated_at")
DUE_SCOPE_ALIASES = {
    "": "",
    "today": "today",
    "overdue": "overdue",
    "late": "overdue",
    "this_week": "this_week",
    "thisweek": "this_week",
    "week": "this_week",
    "current_week": "this_week",
    "next_7_days": "next_7_days",
    "next7days": "next_7_days",
    "upcoming": "next_7_days",
}
TASK_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "content": CONTENT_ALIASES,
    "project": PROJECT_ALIASES,
    "iteration": ITERATION_ALIASES,
    "assignee": PERSON_ALIASES,
    "platform": PLATFORM_ALIASES,
    "start_date": START_DATE_ALIASES,
    "end_date": END_DATE_ALIASES,
    "submit_test_date": SUBMIT_TEST_DATE_ALIASES,
    "updated_at": UPDATE_DATE_ALIASES,
    "priority": PRIORITY_ALIASES,
    "color": COLOR_ALIASES,
    "dev_days": DEV_DAYS_ALIASES,
    "test_days": TEST_DAYS_ALIASES,
    "client_owner": CLIENT_ALIASES,
    "server_owner": SERVER_ALIASES,
    "test_owner": TEST_OWNER_ALIASES,
    "product_owner": PRODUCT_OWNER_ALIASES,
}


@dataclass
class CacheEntry:
    created_at: float
    records: list[dict[str, str]]


@dataclass(frozen=True)
class DateFilterOptions:
    due_scope: str
    as_of_date: date
    end_date_from: date | None
    end_date_to: date | None


@dataclass(frozen=True)
class TaskSearchResult:
    records: list[dict[str, str]]
    total_count: int


class GoogleSheetsClient:
    def __init__(self) -> None:
        self._service = None
        self._cache: CacheEntry | None = None

    def get_records(self, force_refresh: bool = False) -> list[dict[str, str]]:
        settings = load_settings()
        now = time.time()

        if (
            not force_refresh
            and self._cache is not None
            and now - self._cache.created_at < settings.project_sheet_cache_ttl_seconds
        ):
            return self._cache.records

        service = self._get_service()
        result = (
            service.spreadsheets()
            .values()
            .get(
                spreadsheetId=settings.jade_project_sheet_id,
                range=settings.project_sheet_range,
            )
            .execute()
        )

        values = result.get("values", [])
        if not values:
            self._cache = CacheEntry(created_at=now, records=[])
            return []

        headers = [normalize_header(value, index) for index, value in enumerate(values[0])]
        records: list[dict[str, str]] = []

        for row in values[1:]:
            record: dict[str, str] = {}
            for index, header in enumerate(headers):
                record[header] = row[index].strip() if index < len(row) else ""
            if any(value for value in record.values()):
                records.append(record)

        self._cache = CacheEntry(created_at=now, records=records)
        return records

    def search_tasks(
        self,
        query: str = "",
        assignee: str = "",
        project: str = "",
        platform: str = "",
        priority: str = "",
        iteration: str = "",
        date_filters: DateFilterOptions | None = None,
        limit: int = 10,
    ) -> TaskSearchResult:
        query = query.strip().lower()
        assignee = normalize_sheet_identity(assignee).strip().lower()
        project = project.strip().lower()
        platform = platform.strip().lower()
        priority = priority.strip().lower()
        iteration = iteration.strip().lower()

        normalized_limit = max(limit, 1)
        matches: list[dict[str, str]] = []
        dated_matches: list[tuple[date, dict[str, str]]] = []
        total_count = 0
        has_date_filters = date_filters is not None and (
            bool(date_filters.due_scope) or date_filters.end_date_from is not None or date_filters.end_date_to is not None
        )

        for record in self.get_records():
            if assignee:
                person_value = get_first_value(record, PERSON_ALIASES).lower()
                client_value = get_first_value(record, CLIENT_ALIASES).lower()
                server_value = get_first_value(record, SERVER_ALIASES).lower()
                test_value = get_first_value(record, TEST_OWNER_ALIASES).lower()
                product_value = get_first_value(record, PRODUCT_OWNER_ALIASES).lower()
                people_haystack = " ".join(
                    value for value in [person_value, client_value, server_value, test_value, product_value] if value
                )
                if assignee not in people_haystack:
                    continue

            if project:
                project_value = get_first_value(record, PROJECT_ALIASES).lower()
                if project not in project_value:
                    continue

            if platform:
                platform_value = get_first_value(record, PLATFORM_ALIASES).lower()
                if platform not in platform_value:
                    continue

            if priority:
                priority_value = get_first_value(record, PRIORITY_ALIASES).lower()
                if priority not in priority_value:
                    continue

            if iteration:
                iteration_value = get_first_value(record, ITERATION_ALIASES).lower()
                if iteration not in iteration_value:
                    continue

            if query:
                haystack = " ".join(record.values()).lower()
                if query not in haystack:
                    continue

            if has_date_filters:
                end_date = parse_sheet_date(
                    get_first_value(record, END_DATE_ALIASES),
                    reference_date=date_filters.as_of_date,
                )
                if end_date is None:
                    continue
                if not task_matches_due_scope(end_date, date_filters):
                    continue
                if date_filters.end_date_from is not None and end_date < date_filters.end_date_from:
                    continue
                if date_filters.end_date_to is not None and end_date > date_filters.end_date_to:
                    continue

                total_count += 1
                dated_matches.append((end_date, record))
                continue

            total_count += 1
            if len(matches) < normalized_limit:
                matches.append(record)

        if has_date_filters:
            dated_matches.sort(key=lambda item: item[0])
            return TaskSearchResult(
                records=[record for _, record in dated_matches[:normalized_limit]],
                total_count=total_count,
            )

        return TaskSearchResult(records=matches, total_count=total_count)

    def _get_service(self):
        if self._service is not None:
            return self._service

        settings = load_settings()

        if not settings.google_application_credentials:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")
        if not settings.jade_project_sheet_id:
            raise RuntimeError("JADE_PROJECT_SHEET_ID is not set.")

        credentials_path = Path(settings.google_application_credentials)
        if not credentials_path.exists():
            raise RuntimeError(f"Google credentials file not found: {credentials_path}")

        credentials = service_account.Credentials.from_service_account_file(
            os.fspath(credentials_path),
            scopes=SCOPES,
        )
        self._service = build("sheets", "v4", credentials=credentials)
        return self._service


client = GoogleSheetsClient()


@tool
def read_project_tasks(
    query: str = "",
    assignee: str = "",
    project: str = "",
    platform: str = "",
    priority: str = "",
    iteration: str = "",
    due_scope: str = "",
    end_date_from: str = "",
    end_date_to: str = "",
    as_of_date: str = "",
    limit: int = 10,
) -> dict[str, object]:
    """Search the Jade Games project tracker and return structured task data.

    Supported time filters:
    - due_scope: overdue, today, this_week, next_7_days
    - end_date_from / end_date_to: inclusive end-date range in YYYY-MM-DD or similar sheet-style formats
    - as_of_date: reference date for relative due_scope calculations
    """
    normalized_limit = max(limit, 1)
    try:
        date_filters = build_date_filter_options(
            due_scope=due_scope,
            as_of_date=as_of_date,
            end_date_from=end_date_from,
            end_date_to=end_date_to,
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "filters": build_search_filters(
                query=query,
                assignee=assignee,
                project=project,
                platform=platform,
                priority=priority,
                iteration=iteration,
                limit=normalized_limit,
                due_scope=due_scope,
                end_date_from=end_date_from,
                end_date_to=end_date_to,
                as_of_date=as_of_date,
            ),
            "match_count": 0,
            "tasks": [],
        }

    filters = build_search_filters(
        query=query,
        assignee=assignee,
        project=project,
        platform=platform,
        priority=priority,
        iteration=iteration,
        limit=normalized_limit,
        due_scope=date_filters.due_scope,
        end_date_from=date_filters.end_date_from.isoformat() if date_filters.end_date_from else "",
        end_date_to=date_filters.end_date_to.isoformat() if date_filters.end_date_to else "",
        as_of_date=date_filters.as_of_date.isoformat(),
    )

    try:
        search_result = client.search_tasks(
            query=query,
            assignee=assignee,
            project=project,
            platform=platform,
            priority=priority,
            iteration=iteration,
            date_filters=date_filters,
            limit=normalized_limit,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "filters": filters,
            "match_count": 0,
            "tasks": [],
        }

    if isinstance(search_result, TaskSearchResult):
        matches = search_result.records
        match_count = search_result.total_count
    else:
        matches = search_result
        match_count = len(matches)

    result: dict[str, object] = {
        "ok": True,
        "filters": filters,
        "match_count": match_count,
        "tasks": [project_record_to_task(record, reference_date=date_filters.as_of_date) for record in matches],
    }
    date_context = build_date_context(date_filters)
    if date_context:
        result["date_context"] = date_context
    return result


@tool
def get_project_sheet_overview(limit: int = 10) -> dict[str, object]:
    """Return structured metadata and a preview of the project tracker."""
    normalized_limit = max(limit, 1)
    today = date.today()
    try:
        records = client.get_records()
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "total_rows": 0,
            "preview_count": 0,
            "tasks": [],
        }

    preview = records[:normalized_limit]
    return {
        "ok": True,
        "total_rows": len(records),
        "preview_count": len(preview),
        "tasks": [project_record_to_task(record, reference_date=today) for record in preview],
    }


def normalize_header(value: str, index: int) -> str:
    cleaned = value.strip().replace(" ", "_").replace("/", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.lower() if cleaned else f"column_{index + 1}"


def get_first_value(record: dict[str, str], aliases: Iterable[str]) -> str:
    for alias in aliases:
        normalized_alias = normalize_header(alias, 0)
        if normalized_alias in record and record[normalized_alias]:
            return record[normalized_alias]
    return ""


def build_search_filters(
    *,
    query: str,
    assignee: str,
    project: str,
    platform: str,
    priority: str,
    iteration: str,
    limit: int,
    due_scope: str = "",
    end_date_from: str = "",
    end_date_to: str = "",
    as_of_date: str = "",
) -> dict[str, object]:
    requested_assignee = assignee.strip()
    normalized_assignee = normalize_sheet_identity(requested_assignee).strip()

    filters: dict[str, object] = {
        "query": query.strip(),
        "assignee": normalized_assignee or requested_assignee,
        "project": project.strip(),
        "platform": platform.strip(),
        "priority": priority.strip(),
        "iteration": iteration.strip(),
        "due_scope": due_scope.strip(),
        "end_date_from": end_date_from.strip(),
        "end_date_to": end_date_to.strip(),
        "as_of_date": as_of_date.strip(),
        "limit": max(limit, 1),
    }
    if requested_assignee and normalized_assignee and normalized_assignee != requested_assignee:
        filters["assignee_requested"] = requested_assignee
    return filters


def build_date_filter_options(
    *,
    due_scope: str,
    as_of_date: str,
    end_date_from: str,
    end_date_to: str,
) -> DateFilterOptions:
    normalized_due_scope = normalize_due_scope(due_scope)
    if normalized_due_scope not in DUE_SCOPE_ALIASES.values():
        raise ValueError(
            "Invalid due_scope. Expected one of: overdue, today, this_week, next_7_days."
        )

    reference_date = parse_filter_date(as_of_date, label="as_of_date") or date.today()
    from_date = parse_filter_date(end_date_from, label="end_date_from")
    to_date = parse_filter_date(end_date_to, label="end_date_to")
    if from_date is not None and to_date is not None and from_date > to_date:
        raise ValueError("end_date_from cannot be later than end_date_to.")

    return DateFilterOptions(
        due_scope=normalized_due_scope,
        as_of_date=reference_date,
        end_date_from=from_date,
        end_date_to=to_date,
    )


def normalize_due_scope(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
    return DUE_SCOPE_ALIASES.get(normalized, normalized)


def parse_filter_date(value: str, *, label: str) -> date | None:
    if not value.strip():
        return None

    parsed = parse_sheet_date(value, reference_date=date.today())
    if parsed is None:
        raise ValueError(f"Invalid {label}: {value}. Use a date like 2026-03-13.")
    return parsed


def parse_sheet_date(value: str, reference_date: date | None = None) -> date | None:
    cleaned = value.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"未填写", "待定", "-", "--", "n/a", "na", "none"}:
        return None

    numeric = re.fullmatch(r"\d+(?:\.\d+)?", cleaned)
    if numeric:
        return date(1899, 12, 30) + timedelta(days=int(float(cleaned)))

    date_formats_with_year = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%m.%d.%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    )
    for fmt in date_formats_with_year:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    ref = reference_date or date.today()
    date_formats_without_year = (
        "%m-%d",
        "%m/%d",
        "%m.%d",
        "%m月%d日",
    )
    for fmt in date_formats_without_year:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
            return parsed.replace(year=ref.year)
        except ValueError:
            continue

    return None


def task_matches_due_scope(end_date: date, date_filters: DateFilterOptions) -> bool:
    if not date_filters.due_scope:
        return True

    if date_filters.due_scope == "overdue":
        return end_date < date_filters.as_of_date
    if date_filters.due_scope == "today":
        return end_date == date_filters.as_of_date
    if date_filters.due_scope == "this_week":
        week_start = date_filters.as_of_date - timedelta(days=date_filters.as_of_date.weekday())
        week_end = week_start + timedelta(days=6)
        return week_start <= end_date <= week_end
    if date_filters.due_scope == "next_7_days":
        window_end = date_filters.as_of_date + timedelta(days=7)
        return date_filters.as_of_date <= end_date <= window_end
    return True


def build_date_context(date_filters: DateFilterOptions) -> dict[str, str] | None:
    if not date_filters.due_scope and date_filters.end_date_from is None and date_filters.end_date_to is None:
        return None

    context: dict[str, str] = {
        "as_of_date": date_filters.as_of_date.isoformat(),
    }
    if date_filters.due_scope:
        context["due_scope"] = date_filters.due_scope
        if date_filters.due_scope == "this_week":
            week_start = date_filters.as_of_date - timedelta(days=date_filters.as_of_date.weekday())
            week_end = week_start + timedelta(days=6)
            context["window_start"] = week_start.isoformat()
            context["window_end"] = week_end.isoformat()
        elif date_filters.due_scope == "next_7_days":
            context["window_start"] = date_filters.as_of_date.isoformat()
            context["window_end"] = (date_filters.as_of_date + timedelta(days=7)).isoformat()
    if date_filters.end_date_from is not None:
        context["end_date_from"] = date_filters.end_date_from.isoformat()
    if date_filters.end_date_to is not None:
        context["end_date_to"] = date_filters.end_date_to.isoformat()
    return context


def project_record_to_task(record: dict[str, str], reference_date: date | None = None) -> dict[str, object]:
    task: dict[str, object] = {}
    for field_name, aliases in TASK_FIELD_ALIASES.items():
        task[field_name] = get_first_value(record, aliases)
    add_normalized_task_dates(task, reference_date=reference_date or date.today())
    return task


def add_normalized_task_dates(task: dict[str, object], *, reference_date: date) -> None:
    for field_name in DATE_FIELD_NAMES:
        raw_value = task.get(field_name)
        if not isinstance(raw_value, str):
            continue

        parsed = parse_sheet_date(raw_value, reference_date=reference_date)
        if parsed is None:
            continue

        task[f"{field_name}_iso"] = parsed.isoformat()
        if field_name == "end_date":
            task["days_until_end"] = (parsed - reference_date).days
            task["due_status"] = classify_due_status(parsed, reference_date)


def classify_due_status(end_date: date, reference_date: date) -> str:
    if end_date < reference_date:
        return "overdue"
    if end_date == reference_date:
        return "today"
    return "upcoming"
