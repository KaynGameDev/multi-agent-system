from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

from core.config import load_settings
from tools.google_sheets import client, normalize_header

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def main() -> None:
    load_dotenv()
    settings = load_settings(force_reload=True)

    print("=== SETTINGS CHECK ===")
    print(f"GOOGLE_APPLICATION_CREDENTIALS={settings.google_application_credentials}")
    print(f"JADE_PROJECT_SHEET_ID={settings.jade_project_sheet_id}")
    print(f"PROJECT_SHEET_RANGE={settings.project_sheet_range}")
    print()

    credentials_path = Path(settings.google_application_credentials)
    print("=== FILE CHECK ===")
    print(f"Credentials file exists: {credentials_path.exists()}")
    if not credentials_path.exists():
        raise SystemExit("Credentials file is missing.")

    raw_credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    service_account_email = raw_credentials.get("client_email", "")
    project_id = raw_credentials.get("project_id", "")
    print(f"Service account email: {service_account_email}")
    print(f"Google project id: {project_id}")
    print()

    credentials = service_account.Credentials.from_service_account_file(
        os.fspath(credentials_path),
        scopes=SCOPES,
    )
    service = build("sheets", "v4", credentials=credentials)

    print("=== SPREADSHEET METADATA ===")
    spreadsheet = service.spreadsheets().get(spreadsheetId=settings.jade_project_sheet_id).execute()
    print(f"Spreadsheet title: {spreadsheet['properties']['title']}")
    print("Sheets:")
    for sheet in spreadsheet.get("sheets", []):
        print(f"- {sheet['properties']['title']}")
    print()

    print("=== RANGE READ TEST ===")
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
    print(f"Rows returned: {len(values)}")
    if not values:
        raise SystemExit("Range returned no rows. Check tab name and range.")

    headers = values[0]
    normalized_headers = [normalize_header(value, index) for index, value in enumerate(headers)]
    print("Header row:")
    print(headers)
    print("Normalized headers:")
    print(normalized_headers)
    print()

    print("=== TOOL RECORD TEST ===")
    records = client.get_records(force_refresh=True)
    print(f"Parsed records: {len(records)}")
    if records:
        print("First parsed record:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
    print()

    print("=== PERSON LOOKUP TEST ===")
    sample_names = ["刘煜", "@K - Liu Yu", "kayn@songkegame.com"]
    for name in sample_names:
        matches = client.search_tasks(assignee=name, limit=3)
        print(f"Lookup: {name} -> {len(matches)} match(es)")
        for match in matches[:3]:
            print(json.dumps(match, ensure_ascii=False, indent=2))
        print()

    print("Done.")


if __name__ == "__main__":
    main()