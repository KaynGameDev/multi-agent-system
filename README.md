# Jade Agent Multi-Agent System

A lean LangGraph + Slack + Gemini starter for a company-facing multi-agent assistant.

## Current architecture

- `gateway` decides which specialist should handle the request
- `project_task_agent` answers questions that depend on the Google Sheets project tracker
- `knowledge_agent` answers questions about internal docs, architecture, and setup guidance
- `document_conversion_agent` handles Slack-driven design document conversion into canonical AI-friendly knowledge packages
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
- `GOOGLE_API_KEY`
- `JADE_PROJECT_SHEET_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`

Interface configuration:
- Slack: `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`

Optional:
- `GEMINI_MODEL` (default: `gemini-3-flash-preview`)
- `GEMINI_TEMPERATURE` (default: `0.2`)
- `PROJECT_SHEET_RANGE` (default: `Tasks!A1:Z`)
- `PROJECT_SHEET_CACHE_TTL_SECONDS` (default: `30`)
- `SLACK_THINKING_REACTION` (default: `eyes`)
- `KNOWLEDGE_BASE_DIR` (default: `data/knowledge`)
- `KNOWLEDGE_FILE_TYPES` (default: `.md,.txt,.rst,.csv,.tsv,.xlsx,.xlsm`)
- `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH` (default: `data/knowledge/google_sheets_catalog.json`)
- `KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS` (default: `120`)
- `CONVERSION_WORK_DIR` (default: `data/conversion`)

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

- Conversation memory is handled with a LangGraph checkpointer keyed by interface thread/channel.
- The Google Sheets tool is cached briefly to avoid reading the whole sheet on every request.
- The knowledge agent reads local files from `KNOWLEDGE_BASE_DIR`; the default local folder is [`data/knowledge/`](/Users/kayngame/jade_ai_core/data/knowledge/).
- The knowledge agent can also read curated online Google Sheets listed in `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH`.
- The document conversion flow stages Slack-uploaded source files and session state under `CONVERSION_WORK_DIR`.
- Approved canonical packages are written into the knowledge base under `data/knowledge/games/<game_slug>/<market_slug>/<feature_slug>/`.
- Excel exports from Google Sheets should be placed in `KNOWLEDGE_BASE_DIR` as `.xlsx` or `.xlsm` files.
- The online-sheet catalog is a JSON file with one document per spreadsheet. Use [`data/knowledge/google_sheets_catalog.example.json`](/Users/kayngame/jade_ai_core/data/knowledge/google_sheets_catalog.example.json) as the template.
- Each catalog entry supports:
  - `spreadsheet_id`
  - `title`
  - `aliases`
  - `tabs` to allow specific tabs only
  - `ranges` to restrict tab reads to a specific A1 range
- Do **not** commit `.env`, `credentials.json`, `.venv`, or IDE folders.
