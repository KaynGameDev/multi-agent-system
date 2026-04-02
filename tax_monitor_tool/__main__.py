from __future__ import annotations

import argparse

from .runtime import configure_tool_logging, run_forever, run_once


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone tax monitor tool")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously using TAX_MONITOR_POLL_INTERVAL_SECONDS.",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Skip loading environment variables from .env.",
    )
    args = parser.parse_args()

    configure_tool_logging()
    if args.daemon:
        run_forever(use_dotenv=not args.no_dotenv)
    else:
        run_once(use_dotenv=not args.no_dotenv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
