from __future__ import annotations

import unittest

from interfaces.slack_home import build_home_view


class SlackHomeTests(unittest.TestCase):
    def test_build_home_view_includes_header_and_capabilities(self) -> None:
        view = build_home_view(
            {
                "user_display_name": "Kayn",
                "user_sheet_name": "刘煜",
                "user_job_title": "客户端",
            }
        )

        self.assertEqual(view["type"], "home")
        self.assertEqual(view["blocks"][0]["text"]["text"], "Jade Agent")
        section_text = view["blocks"][1]["text"]["text"]
        self.assertIn("Hi Kayn", section_text)
        self.assertIn("刘煜", section_text)
        self.assertIn("客户端", section_text)
        self.assertIn("project tracker questions", section_text)


if __name__ == "__main__":
    unittest.main()
