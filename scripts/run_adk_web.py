from __future__ import annotations

import sys

from google.adk.cli import main


def run() -> int:
    args = ["adk", "web", "--port", "8001", "agents", *sys.argv[1:]]
    sys.argv = args
    return int(main() or 0)


if __name__ == "__main__":
    raise SystemExit(run())
