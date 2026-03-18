# Jade Agent Multi-Agent System

A lean LangGraph + Slack + Gemini starter for a company-facing multi-agent assistant.

## Current architecture

- `gateway` decides which specialist should handle the request
- `project_task_agent` answers questions that depend on the Google Sheets project tracker
- `knowledge_agent` answers questions about internal docs, architecture, and setup guidance
- `general_chat_agent` handles greetings, general chat, and everything outside the project tracker
- `project_tools` executes Google Sheets tools when the project agent decides they are needed
- `knowledge_tools` search and read repository documentation when the knowledge agent needs evidence

## Why this shape

This project uses a **supervisor + specialist agents** pattern instead of a fully autonomous swarm.
That keeps routing explicit, tool execution bounded, and the runtime easier to debug.

## Project layout

```text
core/
  config.py
  gateway.py
  graph.py
  state.py
agents/workers/
  general_chat_agent.py
  project_task_agent.py
interfaces/
  slack_listener.py
tools/
  google_sheets.py
main.py
```

## Environment variables

Copy `.env.example` to `.env` and fill in the values.

Required:
- `SLACK_BOT_TOKEN`
- `SLACK_APP_TOKEN`
- `GOOGLE_API_KEY`
- `JADE_PROJECT_SHEET_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`

Optional:
- `GEMINI_MODEL` (default: `gemini-3-flash-preview`)
- `GEMINI_TEMPERATURE` (default: `0.2`)
- `PROJECT_SHEET_RANGE` (default: `Tasks!A1:Z`)
- `PROJECT_SHEET_CACHE_TTL_SECONDS` (default: `30`)
- `SLACK_THINKING_REACTION` (default: `eyes`)
- `KNOWLEDGE_BASE_DIR` (default: `data/knowledge`)
- `KNOWLEDGE_FILE_TYPES` (default: `.md,.txt,.rst,.csv,.tsv,.xlsx,.xlsm`)

## Slack setup

The bot expects Socket Mode.

Recommended scopes/events:
- Bot token scopes:
  - `app_mentions:read`
  - `channels:history`
  - `chat:write`
  - `im:history`
  - `reactions:write` (optional, for the “thinking” reaction)
- Event subscriptions:
  - `app_home_opened`
  - `app_mention`
  - `message.im`

## Run

```bash
python main.py
```

## Notes

- Conversation memory is handled with a LangGraph checkpointer keyed by Slack thread/channel.
- The Google Sheets tool is cached briefly to avoid reading the whole sheet on every request.
- The knowledge agent reads local files from `KNOWLEDGE_BASE_DIR`; the default local folder is [`data/knowledge/`](/Users/kayngame/jade_ai_core/data/knowledge/).
- Excel exports from Google Sheets should be placed in `KNOWLEDGE_BASE_DIR` as `.xlsx` or `.xlsm` files.
- Do **not** commit `.env`, `credentials.json`, `.venv`, or IDE folders.
