# Jade Agent Multi-Agent System

A lean LangGraph + Slack + Gemini starter for a company-facing multi-agent assistant.

## Current architecture

- `gateway` is the deterministic policy layer for routing, skill precedence, fallback, and delegation
- `app/contracts.py` and `app/state.py` define the shared runtime contract model
- `app/pending_actions.py` is the only supported confirmation and waiting model
- `app/tool_runtime.py` standardizes tool invocation/results for both ToolNode calls and internal workflows
- `app/skill_runtime.py` attaches runtime-selected skill instructions to prompts
- `app/skills.py` holds the shared skill registry and normalized skill resolution
- `project_task_agent` answers questions that depend on the Google Sheets project tracker
- `knowledge_agent` answers questions about internal docs, architecture, and setup guidance
- `knowledge_base_builder_agent` handles knowledge elicitation, KB document review, layer placement, and KB V1 tracking
- `document_conversion_agent` handles Slack-driven design document conversion into canonical AI-friendly knowledge packages
- `general_chat_agent` handles greetings, general chat, and everything outside the project tracker
- `project_tools` executes Google Sheets tools when the project agent decides they are needed
- `knowledge_tools` search and read repository documentation when the knowledge agent needs evidence

## Why this shape

This project uses a **supervisor + specialist agents** pattern instead of a fully autonomous swarm.
That keeps routing explicit, tool execution bounded, and the runtime easier to debug.

## Runtime model

Each turn follows one supported runtime flow:

1. Slack or web appends the user message and metadata to graph state.
2. The gateway resolves the route deterministically in this order:
   - requested agent
   - forked-skill delegate
   - forked-skill fallback
   - inline-skill-compatible agent
   - pending-action owner
   - tool-intent route
   - deterministic matcher
   - general fallback
3. The selected agent either:
   - resolves a shared `pending_action`
   - renders a standardized tool result deterministically
   - or invokes the model
4. Tool execution flows through one shared envelope model.
5. The final answer is emitted through `assistant_response`.

See [ARCHITECTURE.md](/Users/kayngame/jade_ai_core/ARCHITECTURE.md) for the full runtime description.

## Project layout

```text
app/
  config.py
  graph.py
  skills.py
  paths.py
gateway/
  agent.py
agents/
  general_chat/
    agent.py
  knowledge/
    agent.py
  knowledge_base_builder/
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
  AI/
    Rules/
    Prompts/
  Docs/
    00_Shared/
    10_GameLines/
    20_Deployments/
    30_Review/
    40_Legacy/
    50_Templates/
runtime/
  conversion/
main.py
```

- `app/` holds shared runtime code and bootstrap helpers.
- `gateway/` holds the deterministic gateway policy and entrypoint.
- `agents/` holds specialist agents and their agent-specific helpers.
- `interfaces/` holds delivery-channel code such as Slack and web.
- `knowledge/` holds curated knowledge content, AI workspace rules/prompts, and can later point at an external vault.
- `runtime/` holds generated state such as uploads and sqlite data.

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
- `PENDING_ACTION_PARSER_MODEL` (default: `GEMINI_MODEL`)
- `PENDING_ACTION_PARSER_TEMPERATURE` (default: `0.0`)
- `PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD` (default: `0.75`)
- `GEMINI_HTTP_TRUST_ENV` (default: `false`; set to `true` only if Gemini traffic must use `HTTPS_PROXY` / `ALL_PROXY`)
- `PROJECT_SHEET_RANGE` (default: `Tasks!A1:Z`)
- `PROJECT_SHEET_CACHE_TTL_SECONDS` (default: `30`)
- `SLACK_THINKING_REACTION` (default: `eyes`)
- `KNOWLEDGE_BASE_DIR` (default: `knowledge`)
- `KNOWLEDGE_FILE_TYPES` (default: `.md,.txt,.rst,.csv,.tsv,.xlsx,.xlsm`)
- `JADE_PROJECT_SKILLS_DIR` (default: `.jade/skills`)
- `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH` (default: `knowledge/AI/Rules/google_sheets_catalog.json`)
- `KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS` (default: `120`)
- `CONVERSION_WORK_DIR` (default: `runtime/conversion`)

## Slack setup

The bot expects Socket Mode.

Recommended scopes/events:
- Bot token scopes:
  - `app_mentions:read`
  - `channels:history`
  - `groups:history` (if the bot should continue active conversion threads in private channels)
  - `chat:write`
  - `im:history`
  - `reactions:write` (optional, for the “thinking” reaction)
- Event subscriptions:
  - `app_home_opened`
  - `app_mention`
  - `message.channels` (recommended for channel thread follow-ups and uploads)
  - `message.groups` (recommended for private-channel thread follow-ups and uploads)
  - `message.im`

## Run

```bash
python main.py
```

## Notes

- Conversation memory is handled with a LangGraph checkpointer keyed by interface thread/channel.
- Skill discovery is centralized in the shared registry. Agent-local skills live under `agents/<agent>/Skills/*/SKILL.md`, and project-shared skills live under `JADE_PROJECT_SKILLS_DIR`.
- Skill precedence is deterministic: `path-scoped > agent-local > project-shared`.
- One `skill_id` resolves to one effective definition per request. Same-scope duplicates are surfaced as configuration conflicts in diagnostics.
- Routing is deterministic and centralized in the gateway. Slack and web no longer force worker routes directly.
- `general_chat_agent` is treated as the `GeneralAssistant` fallback in gateway policy.
- The Google Sheets tool is cached briefly to avoid reading the whole sheet on every request.
- The knowledge agent reads local files from `KNOWLEDGE_BASE_DIR`; the default local folder is [`knowledge/`](/Users/kayngame/jade_ai_core/knowledge/).
- The knowledge-base builder agent uses the read-only knowledge tools for evidence gathering and also has builder-only tools to resolve canonical KB paths and write Markdown drafts under `knowledge/Docs/`, but any KB file mutation is approval-gated and requires an explicit follow-up confirmation before execution.
- The knowledge agent can also read curated online Google Sheets listed in `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH`.
- The document conversion flow stages Slack-uploaded source files and session state under `CONVERSION_WORK_DIR`.
- The knowledge base is organized under `knowledge/AI/` and `knowledge/Docs/`.
- Approved canonical packages are written into the knowledge base under `knowledge/Docs/20_Deployments/<deployment>/<game_line>/Features/<feature_slug>/`.
- Shared cross-game knowledge belongs under `knowledge/Docs/00_Shared/`, and game-line shared context belongs under `knowledge/Docs/10_GameLines/<game_line>/`.
- Excel exports from Google Sheets can be placed anywhere under `KNOWLEDGE_BASE_DIR` that fits the new hierarchy, such as `Docs/00_Shared/` or a deployment/game folder.
- The online-sheet catalog is a JSON file with one document per spreadsheet. Use [`knowledge/AI/Rules/google_sheets_catalog.example.json`](/Users/kayngame/jade_ai_core/knowledge/AI/Rules/google_sheets_catalog.example.json) as the template.
- Each catalog entry supports:
  - `spreadsheet_id`
  - `title`
  - `aliases`
  - `tabs` to allow specific tabs only
  - `ranges` to restrict tab reads to a specific A1 range
- The web API now returns structured `skill_resolution_diagnostics`, `agent_selection_diagnostics`, and `selection_warnings` alongside legacy `route` metadata.
- Routing is owned by runtime policy code, not by prompt instructions.
- `pending_interaction`, regex-only approval flows, and prompt-only skill routing are retired live runtime systems.
- Do **not** commit `.env`, `credentials.json`, `.venv`, or IDE folders.
