from __future__ import annotations

import unittest

from core.language import detect_response_language


class LanguageTests(unittest.TestCase):
    def test_detect_response_language_prefers_chinese_for_chinese_dominant_text(self) -> None:
        self.assertEqual(detect_response_language("请 summarize 一下这个文档。"), "zh")

    def test_detect_response_language_prefers_english_when_only_name_is_chinese(self) -> None:
        self.assertEqual(detect_response_language("What is 刘煜 working on this week?"), "en")


if __name__ == "__main__":
    unittest.main()
