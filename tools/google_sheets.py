from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_core.tools import tool

from core.config import load_settings

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

TASK_ID_ALIASES = ("task_id", "id", "task", "ticket", "item")
TITLE_ALIASES = ("title", "feature", "name", "summary", "description")
STATUS_ALIASES = ("status", "state")
ASSIGNEE_ALIASES = ("assignee", "owner", "developer", "pic", "person_in_charge")
DUE_DATE_ALIASES = ("due", "due_date", "deadline", "eta")
PROJECT_ALIASES = ("project", "initiative", "epic")


@dataclass
class CacheEntry:
    created_at: float
    records: list[dict[str, str]]


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
        task_id: str = "",
        assignee: str = "",
        status: str = "",
        limit: int = 10,
    ) -> list[dict[str, str]]:
        query = query.strip().lower()
        task_id = task_id.strip().lower()
        assignee = assignee.strip().lower()
        status = status.strip().lower()

        matches: list[dict[str, str]] = []
        for record in self.get_records():
            if task_id:
                task_id_value = get_first_value(record, TASK_ID_ALIASES).lower()
                if task_id not in task_id_value:
                    continue

            if assignee:
                assignee_value = get_first_value(record, ASSIGNEE_ALIASES).lower()
                if assignee not in assignee_value:
                    continue

            if status:
                status_value = get_first_value(record, STATUS_ALIASES).lower()
                if status not in status_value:
                    continue

            if query:
                haystack = " ".join(record.values()).lower()
                if query not in haystack:
                    continue

            matches.append(record)
            if len(matches) >= max(limit, 1):
                break

        return matches

    def _get_service(self):
        if self._service is not None:
            return self._service

        settings = load_settings()

        if not settings.google_application_credentials:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS is not set."
            )
        if not settings.jade_project_sheet_id:
            raise RuntimeError(
                "JADE_PROJECT_SHEET_ID is not set."
            )

        credentials_path = Path(settings.google_application_credentials)
        if not credentials_path.exists():
            raise RuntimeError(
                f"Google credentials file not found: {credentials_path}"
            )

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
    task_id: str = "",
    assignee: str = "",
    status: str = "",
    limit: int = 10,
) -> str:
    """Search the Jade Games project tracker for tasks, assignees, statuses, owners, or deadlines."""
    try:
        matches = client.search_tasks(
            query=query,
            task_id=task_id,
            assignee=assignee,
            status=status,
            limit=limit,
        )
    except Exception as exc:
        return f"Could not read the Google Sheet: {exc}"

    if not matches:
        return "No matching tasks were found in the project tracker."

    lines = [format_task(record) for record in matches]
    return "\n\n".join(lines)


@tool
def get_project_sheet_overview(limit: int = 10) -> str:
    """Return a quick overview of the project tracker, including total tasks and a few sample rows."""
    try:
        records = client.get_records()
    except Exception as exc:
        return f"Could not read the Google Sheet: {exc}"

    if not records:
        return "The project tracker is empty."

    preview = records[: max(limit, 1)]
    lines = [f"Total tasks visible in sheet: {len(records)}"]
    lines.extend(format_task(record) for record in preview)
    return "\n\n".join(lines)


def normalize_header(value: str, index: int) -> str:
    cleaned = value.strip().lower().replace(" ", "_").replace("/", "_")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned or f"column_{index + 1}"


def get_first_value(record: dict[str, str], aliases: Iterable[str]) -> str:
    for alias in aliases:
        if alias in record and record[alias]:
            return record[alias]
    return ""


def format_task(record: dict[str, str]) -> str:
    task_id = get_first_value(record, TASK_ID_ALIASES) or "Unknown task id"
    title = get_first_value(record, TITLE_ALIASES) or "Untitled task"
    status = get_first_value(record, STATUS_ALIASES) or "Unknown"
    assignee = get_first_value(record, ASSIGNEE_ALIASES) or "Unassigned"
    due_date = get_first_value(record, DUE_DATE_ALIASES) or "Not set"
    project = get_first_value(record, PROJECT_ALIASES)

    lines = [
        f"- Task: {task_id}",
        f"  Title: {title}",
        f"  Status: {status}",
        f"  Assignee: {assignee}",
        f"  Due: {due_date}",
    ]
    if project:
        lines.append(f"  Project: {project}")
    return "\n".join(lines)
