from __future__ import annotations

import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config import load_settings

GOOGLE_SHEETS_SCOPES = ("https://www.googleapis.com/auth/spreadsheets.readonly",)
GOOGLE_DOCS_SCOPES = ("https://www.googleapis.com/auth/documents.readonly",)

_shared_services: dict[tuple[str, tuple[str, ...]], object] = {}


def get_google_sheets_service():
    return _get_google_workspace_service("sheets", "v4", GOOGLE_SHEETS_SCOPES)


def get_google_docs_service():
    return _get_google_workspace_service("docs", "v1", GOOGLE_DOCS_SCOPES)


def _get_google_workspace_service(api_name: str, api_version: str, scopes: tuple[str, ...]):
    cache_key = (api_name, scopes)
    shared_service = _shared_services.get(cache_key)
    if shared_service is not None:
        return shared_service

    credentials = _load_service_account_credentials(scopes)
    service = build(api_name, api_version, credentials=credentials, cache_discovery=False)
    _shared_services[cache_key] = service
    return service


def _load_service_account_credentials(scopes: tuple[str, ...]):
    settings = load_settings()
    if not settings.google_application_credentials:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set.")

    credentials_path = Path(settings.google_application_credentials)
    if not credentials_path.exists():
        raise RuntimeError(f"Google credentials file not found: {credentials_path}")

    return service_account.Credentials.from_service_account_file(
        os.fspath(credentials_path),
        scopes=list(scopes),
    )
