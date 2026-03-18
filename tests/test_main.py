from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main


class DummyListener:
    def start(self) -> None:
        raise KeyboardInterrupt


class MainTests(unittest.TestCase):
    def test_main_handles_keyboard_interrupt_cleanly(self) -> None:
        stdout = io.StringIO()
        with patch("main.bootstrap_system", return_value=DummyListener()):
            with redirect_stdout(stdout):
                exit_code = main.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Stopping Jade Agent...", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
