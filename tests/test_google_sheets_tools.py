from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.google_sheets import TaskSearchResult, get_project_sheet_overview, parse_sheet_date, read_project_tasks


SAMPLE_RECORD = {
    "内容": "大厅活动优化",
    "项目": "Jade Poker",
    "迭代": "S24",
    "人员": "刘煜",
    "平台": "iOS",
    "start": "2026-03-10",
    "end": "2026-03-14",
    "提测日期": "2026-03-13",
    "更新日期": "2026-03-12",
    "优先级": "P1",
    "color": "green",
    "开发天数": "3",
    "测试天数": "1",
    "客户端": "刘煜",
    "服务器": "郑煜钊",
    "测试": "刘静芳",
    "产品": "叶俊杰",
}


class GoogleSheetsToolTests(unittest.TestCase):
    @patch("tools.google_sheets.client.search_tasks")
    def test_read_project_tasks_returns_structured_payload(self, mock_search_tasks) -> None:
        mock_search_tasks.return_value = TaskSearchResult(records=[SAMPLE_RECORD], total_count=1)

        result = read_project_tasks.invoke({"assignee": "@K - Liu Yu", "limit": 1})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["filters"]["assignee"], "刘煜")
        self.assertEqual(result["filters"]["assignee_requested"], "@K - Liu Yu")
        self.assertEqual(result["tasks"][0]["content"], "大厅活动优化")
        self.assertEqual(result["tasks"][0]["project"], "Jade Poker")
        self.assertEqual(result["tasks"][0]["client_owner"], "刘煜")

    @patch("tools.google_sheets.client.search_tasks")
    def test_read_project_tasks_returns_structured_error_payload(self, mock_search_tasks) -> None:
        mock_search_tasks.side_effect = RuntimeError("sheet unavailable")

        result = read_project_tasks.invoke({"assignee": "刘煜"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["match_count"], 0)
        self.assertEqual(result["tasks"], [])
        self.assertIn("sheet unavailable", result["error"])

    @patch("tools.google_sheets.client.get_records")
    def test_get_project_sheet_overview_returns_preview_records(self, mock_get_records) -> None:
        mock_get_records.return_value = [SAMPLE_RECORD, {**SAMPLE_RECORD, "内容": "支付重构"}]

        result = get_project_sheet_overview.invoke({"limit": 1})

        self.assertTrue(result["ok"])
        self.assertEqual(result["total_rows"], 2)
        self.assertEqual(result["preview_count"], 1)
        self.assertEqual(len(result["tasks"]), 1)
        self.assertEqual(result["tasks"][0]["content"], "大厅活动优化")

    @patch("tools.google_sheets.client.get_records")
    def test_read_project_tasks_filters_overdue_items(self, mock_get_records) -> None:
        mock_get_records.return_value = [
            {**SAMPLE_RECORD, "内容": "过期任务", "end": "2026-03-12"},
            {**SAMPLE_RECORD, "内容": "今天任务", "end": "2026-03-13"},
            {**SAMPLE_RECORD, "内容": "未来任务", "end": "2026-03-15"},
        ]

        result = read_project_tasks.invoke({"due_scope": "overdue", "as_of_date": "2026-03-13", "limit": 10})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 1)
        self.assertEqual(result["date_context"]["as_of_date"], "2026-03-13")
        self.assertEqual(result["tasks"][0]["content"], "过期任务")
        self.assertEqual(result["tasks"][0]["due_status"], "overdue")

    @patch("tools.google_sheets.client.get_records")
    def test_read_project_tasks_filters_this_week_and_sorts_by_end_date(self, mock_get_records) -> None:
        mock_get_records.return_value = [
            {**SAMPLE_RECORD, "内容": "下周任务", "end": "2026-03-16"},
            {**SAMPLE_RECORD, "内容": "周日任务", "end": "2026-03-15"},
            {**SAMPLE_RECORD, "内容": "周六任务", "end": "2026-03-14"},
            {**SAMPLE_RECORD, "内容": "周一任务", "end": "2026-03-09"},
        ]

        result = read_project_tasks.invoke({"due_scope": "this_week", "as_of_date": "2026-03-13", "limit": 10})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 3)
        self.assertEqual(result["date_context"]["window_start"], "2026-03-09")
        self.assertEqual(result["date_context"]["window_end"], "2026-03-15")
        self.assertEqual(
            [task["content"] for task in result["tasks"]],
            ["周一任务", "周六任务", "周日任务"],
        )

    @patch("tools.google_sheets.client.get_records")
    def test_read_project_tasks_reports_total_match_count_beyond_limit(self, mock_get_records) -> None:
        mock_get_records.return_value = [
            {**SAMPLE_RECORD, "内容": "周一任务", "end": "2026-03-09"},
            {**SAMPLE_RECORD, "内容": "周六任务", "end": "2026-03-14"},
            {**SAMPLE_RECORD, "内容": "周日任务", "end": "2026-03-15"},
        ]

        result = read_project_tasks.invoke({"due_scope": "this_week", "as_of_date": "2026-03-13", "limit": 1})

        self.assertTrue(result["ok"])
        self.assertEqual(result["match_count"], 3)
        self.assertEqual(len(result["tasks"]), 1)
        self.assertEqual(result["tasks"][0]["content"], "周一任务")

    def test_parse_sheet_date_supports_dates_without_year(self) -> None:
        parsed = parse_sheet_date("3/14", reference_date=parse_sheet_date("2026-03-13"))

        self.assertEqual(parsed.isoformat(), "2026-03-14")


if __name__ == "__main__":
    unittest.main()
