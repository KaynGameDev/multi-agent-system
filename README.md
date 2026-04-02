# Jade Agent Multi-Agent System

A lean LangGraph + Slack + Gemini starter for a company-facing multi-agent assistant.

## Current architecture

- `gateway` decides which specialist should handle the request
- `project_task_agent` answers questions that depend on the Google Sheets project tracker
- `knowledge_agent` answers questions about internal docs, architecture, and setup guidance
- `document_conversion_agent` handles Slack-driven design document conversion into canonical AI-friendly knowledge packages
- `general_chat_agent` handles greetings, general chat, and everything outside the project tracker
- `tax_monitor_tool` is the standalone tax-monitor package and CLI entrypoint for polling the tax-control webpage without the agent graph
- `tax_monitor_service` wires the tax monitor into the main app runtime and pushes Slack alerts with cooldown protection
- `project_tools` executes Google Sheets tools when the project agent decides they are needed
- `knowledge_tools` search and read repository documentation when the knowledge agent needs evidence

## Why this shape

This project uses a **supervisor + specialist agents** pattern instead of a fully autonomous swarm.
That keeps routing explicit, tool execution bounded, and the runtime easier to debug.

## Project layout

```text
app/
  config.py
  graph.py
  paths.py
gateway/
  agent.py
agents/
  general_chat/
    agent.py
  knowledge/
    agent.py
  project_task/
    agent.py
  document_conversion/
    agent.py
interfaces/
  slack/
    listener.py
    formatting.py
  web/
    server.py
tools/
  project_tracker_google_sheets.py
  conversion_google_sources.py
  google_workspace_services.py
knowledge/
  ... curated source content
runtime/
  conversion/
  monitoring/
main.py
```

- `app/` holds shared runtime code and bootstrap helpers.
- `gateway/` holds the entrance routing agent.
- `agents/` holds specialist agents and their agent-specific helpers.
- `interfaces/` holds delivery-channel code such as Slack and web.
- `knowledge/` holds curated knowledge content and can later point at an external vault.
- `runtime/` holds generated state such as uploads, sqlite data, and monitor state.

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
- `GEMINI_HTTP_TRUST_ENV` (default: `false`; set to `true` only if Gemini traffic must use `HTTPS_PROXY` / `ALL_PROXY`)
- `PROJECT_SHEET_RANGE` (default: `Tasks!A1:Z`)
- `PROJECT_SHEET_CACHE_TTL_SECONDS` (default: `30`)
- `SLACK_THINKING_REACTION` (default: `eyes`)
- `KNOWLEDGE_BASE_DIR` (default: `knowledge`)
- `KNOWLEDGE_FILE_TYPES` (default: `.md,.txt,.rst,.csv,.tsv,.xlsx,.xlsm`)
- `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH` (default: `knowledge/google_sheets_catalog.json`)
- `KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS` (default: `120`)
- `CONVERSION_WORK_DIR` (default: `runtime/conversion`)
- `TAX_MONITOR_ENABLED` (default: `false`)
- `TAX_MONITOR_URL` (default: `https://ghv2.rydgames.com:62933/Page/index.html`)
- `TAX_MONITOR_USERNAME`
- `TAX_MONITOR_PASSWORD`
- `TAX_MONITOR_TOKEN` (optional if the page does not require it)
- `TAX_MONITOR_CAPTURE_GROUP`
- `TAX_MONITOR_SLACK_CHANNEL`
- `TAX_MONITOR_POLL_INTERVAL_SECONDS` (default: `300`)
- `TAX_MONITOR_ALERT_COOLDOWN_SECONDS` (default: `7200`)
- `TAX_MONITOR_ERROR_COOLDOWN_SECONDS` (default: `1800`)
- `TAX_MONITOR_STATE_PATH` (default: `runtime/monitoring/tax_monitor_state.json`)
- `TAX_MONITOR_BROWSER_TIMEOUT_SECONDS` (default: `45`)
- `TAX_MONITOR_HEADLESS` (default: `true`)
- `TAX_MONITOR_NAVIGATION_PATH` (default: `税收调控管理,税收详情（新）`)
- Optional selector overrides:
  - `TAX_MONITOR_USERNAME_SELECTOR`
  - `TAX_MONITOR_PASSWORD_SELECTOR`
  - `TAX_MONITOR_TOKEN_SELECTOR`
  - `TAX_MONITOR_LOGIN_BUTTON_SELECTOR`
  - `TAX_MONITOR_CAPTURE_GROUP_SELECTOR`
  - `TAX_MONITOR_QUERY_BUTTON_SELECTOR`

## Slack setup

The bot expects Socket Mode.

Recommended scopes/events:
- Bot token scopes:
  - `app_mentions:read`
  - `channels:history`
  - `groups:history` (if the verification channel is private)
  - `chat:write`
  - `im:history`
  - `reactions:write` (optional, for the “thinking” reaction)
- Event subscriptions:
  - `app_home_opened`
  - `app_mention`
  - `message.channels` (required if tax-monitor OTP codes are posted in a public channel)
  - `message.groups` (required if tax-monitor OTP codes are posted in a private channel)
  - `message.im`

## Run

```bash
python main.py
```

## Standalone tax monitor tool

Run one cycle:

```bash
python3 -m tax_monitor_tool
```

Run continuously:

```bash
python3 -m tax_monitor_tool --daemon
```

If you enable the webpage monitor, install the browser runtime once:

```bash
playwright install chromium
```

OTP smoke test (no live website required):

```bash
python3 -m unittest tests.test_tax_monitor_otp_smoke
```

## Notes

- Conversation memory is handled with a LangGraph checkpointer keyed by interface thread/channel.
- The Google Sheets tool is cached briefly to avoid reading the whole sheet on every request.
- The knowledge agent reads local files from `KNOWLEDGE_BASE_DIR`; the default local folder is [`knowledge/`](/Users/kayngame/jade_ai_core/knowledge/).
- The knowledge agent can also read curated online Google Sheets listed in `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH`.
- The document conversion flow stages Slack-uploaded source files and session state under `CONVERSION_WORK_DIR`.
- The tax monitor persists alert cooldown state under [`runtime/monitoring/`](/Users/kayngame/jade_ai_core/runtime/monitoring/) by default.
- Approved canonical packages are written into the knowledge base under `knowledge/games/<game_slug>/<market_slug>/<feature_slug>/`.
- Excel exports from Google Sheets should be placed in `KNOWLEDGE_BASE_DIR` as `.xlsx` or `.xlsm` files.
- The online-sheet catalog is a JSON file with one document per spreadsheet. Use [`knowledge/google_sheets_catalog.example.json`](/Users/kayngame/jade_ai_core/knowledge/google_sheets_catalog.example.json) as the template.
- Each catalog entry supports:
  - `spreadsheet_id`
  - `title`
  - `aliases`
  - `tabs` to allow specific tabs only
  - `ranges` to restrict tab reads to a specific A1 range
- The tax monitor currently checks four visible projects on the `税收详情(新)` page after selecting the `捕获组` dropdown: `4 Player Fishing`, `West Journey Fishing`, `2 Player Fishing`, and `SkyFire Fishing`.
- The monitor maps those project names to the canonical game names `四人捕鱼`, `西游捕鱼`, `二人捕鱼`, and `飞机捕鱼` when evaluating alerts.
- If the login page requires a 6-digit dynamic verification code, the monitor posts a Slack prompt in `TAX_MONITOR_SLACK_CHANNEL`, waits for a user to send the code in that channel, and retries the login automatically.
- For the OTP flow to work, keep the Slack listener enabled so channel messages can be received.
- It sends one alert per project/game per day for a first negative rate, and rate-range alerts no more than once every two hours per project/game.
- Do **not** commit `.env`, `credentials.json`, `.venv`, or IDE folders.
