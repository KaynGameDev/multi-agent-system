from __future__ import annotations


def build_home_view(user_context: dict | None = None) -> dict:
    context = user_context or {}
    display_name = (
        context.get("user_display_name")
        or context.get("user_real_name")
        or context.get("user_sheet_name")
        or "there"
    )
    display_name = escape_slack_text(display_name)

    identity_line = build_identity_line(context)

    return {
        "type": "home",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Jade Agent",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Hi {display_name}.* I'm your internal Slack assistant.\n"
                        "I can help with project tracker questions, internal documentation, and general chat."
                        f"{identity_line}"
                    ),
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*What I can help with*\n"
                        "• Project tracker: task owners, deadlines, overdue items, and weekly workload\n"
                        "• Documentation: summarize design docs, setup docs, and internal workflow notes\n"
                        "• General help: quick questions and everyday chat"
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*Try asking*\n"
                        "• What are my tasks due this week?\n"
                        "• Summarize the latest design document in knowledge base\n"
                        "• What does our architecture say about tool formatting?"
                    ),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Open a DM with me or mention me in a channel. Channel replies stay threaded.",
                    }
                ],
            },
        ],
    }


def build_identity_line(user_context: dict) -> str:
    sheet_name = escape_slack_text(user_context.get("user_sheet_name", ""))
    job_title = escape_slack_text(user_context.get("user_job_title", ""))
    if not sheet_name:
        return ""
    if job_title:
        return f"\nYour current identity mapping is *{sheet_name}* ({job_title})."
    return f"\nYour current identity mapping is *{sheet_name}*."


def escape_slack_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
