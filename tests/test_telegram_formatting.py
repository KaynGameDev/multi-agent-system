from __future__ import annotations

import unittest

from core.telegram_formatting import to_telegram_html


class TelegramFormattingTests(unittest.TestCase):
    def test_converts_headings_bold_links_and_code(self) -> None:
        text = (
            "# Title\n"
            "Use **bold** text and [OpenAI](https://openai.com).\n"
            "Inline `code` works.\n"
            "```text\n"
            "row_1,value_1\n"
            "```"
        )

        formatted = to_telegram_html(text)

        self.assertIn("<b>Title</b>", formatted)
        self.assertIn("<b>bold</b>", formatted)
        self.assertIn('<a href="https://openai.com">OpenAI</a>', formatted)
        self.assertIn("<code>code</code>", formatted)
        self.assertIn("<pre><code>row_1,value_1</code></pre>", formatted)

    def test_escapes_plain_html_characters(self) -> None:
        formatted = to_telegram_html("5 < 7 & 8 > 3")

        self.assertEqual(formatted, "5 &lt; 7 &amp; 8 &gt; 3")


if __name__ == "__main__":
    unittest.main()
