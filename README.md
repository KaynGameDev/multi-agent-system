# Jade Knowledge Base Builder

A web-first knowledge-base building workspace built on Jade's LangGraph runtime. It helps teams scan KB gaps, review document structure, convert source docs and Sheets into canonical packages, and approval-gate writes into the repository knowledge base.

## Product focus

This repo is the standalone home for the Jade knowledge-base builder product.

- The web interface is the primary surface for day-to-day KB building
- `knowledge_base_builder_agent` drives gap discovery, elicitation, structure review, and draft planning
- `document_conversion_agent` converts design docs and Sheets into approval-gated canonical knowledge packages
- Slack support remains available as an optional interface, but it is no longer the product center of gravity

## Current architecture

- `gateway` is the deterministic policy layer for routing, skill precedence, fallback, and delegation
- `app/contracts/` and `app/state.py` define the shared runtime contract model
- `app/pending_actions.py` is the only supported confirmation and waiting model
- `app/tool_runtime.py` standardizes tool invocation/results for both ToolNode calls and internal workflows
- `app/skill_runtime.py` attaches runtime-selected skill instructions to prompts
- `app/skills.py` holds the shared skill registry and normalized skill resolution
- `project_task_agent` answers questions that depend on the Google Sheets project tracker
- `knowledge_agent` answers questions about internal docs, architecture, and setup guidance
- `knowledge_base_builder_agent` handles knowledge elicitation, KB document review, layer placement, and KB V1 tracking
- `document_conversion_agent` handles document conversion into canonical AI-friendly knowledge packages
- `general_chat_agent` handles greetings, general chat, and everything outside the project tracker
- `project_tools` executes Google Sheets tools when the project agent decides they are needed
- `knowledge_tools` search and read repository documentation when the knowledge agent needs evidence

## Why this shape

This project uses a **supervisor + specialist agents** pattern instead of a fully autonomous swarm.
That keeps routing explicit, tool execution bounded, and the runtime easier to debug.

## Runtime model

Each turn follows one supported runtime flow:

1. Slack or web appends the user message and metadata to graph state.
2. The gateway resolves the route deterministically:
   - active `pending_action` first
   - explicit internal overrides such as requested agent or explicit skill delegation next
   - otherwise parser-based `AssistantRequest` routing
   - otherwise deterministic fallback to `general_chat_agent`
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
- `GOOGLE_APPLICATION_CREDENTIALS`
- `JADE_PROJECT_SHEET_ID`

Interface configuration:
- Slack: `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`
- Web auth and host guard: `WEB_AUTH_ENABLED`, `WEB_AUTH_SESSION_SECRET`, and either `WEB_AUTH_CREDENTIALS` or `WEB_AUTH_USERNAME` + `WEB_AUTH_PASSWORD`

Optional:
- `LLM_PROVIDER` (default: `google`; supported: `google`, `minimax`, `openai`)
- `LLM_MODEL` (default depends on provider: `gemini-3-flash-preview`, `MiniMax-M2.7-highspeed`, `gpt-5-mini`)
- `LLM_TEMPERATURE` (default: `0.2`)
- `GOOGLE_API_KEY` (required only when `LLM_PROVIDER=google`)
- `MINIMAX_API_KEY` (required only when `LLM_PROVIDER=minimax`)
- `MINIMAX_BASE_URL` (default: `https://api.minimaxi.com/v1`; use `https://api.minimax.io/v1` outside China if needed)
- `OPENAI_API_KEY` (required only when `LLM_PROVIDER=openai`)
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `ROUTING_LLM_PROVIDER` (optional; defaults to `LLM_PROVIDER` when omitted)
- `ROUTING_LLM_MODEL` (optional; defaults to `LLM_MODEL` when omitted, or the routing provider default when switching providers)
- `ROUTING_LLM_TEMPERATURE` (optional; defaults to `LLM_TEMPERATURE`)
- `ROUTING_LLM_HTTP_TRUST_ENV` (optional; defaults to `LLM_HTTP_TRUST_ENV`)
- `ROUTING_GOOGLE_API_KEY` (optional; falls back to `GOOGLE_API_KEY`)
- `ROUTING_MINIMAX_API_KEY` (optional; falls back to `MINIMAX_API_KEY`)
- `ROUTING_MINIMAX_BASE_URL` (optional; falls back to `MINIMAX_BASE_URL`)
- `ROUTING_OPENAI_API_KEY` (optional; falls back to `OPENAI_API_KEY`)
- `ROUTING_OPENAI_BASE_URL` (optional; falls back to `OPENAI_BASE_URL`)
- `ASSISTANT_REQUEST_PARSER_CONFIDENCE_THRESHOLD` (default: `0.60`)
- `PENDING_ACTION_PARSER_CONFIDENCE_THRESHOLD` (default: `0.75`)
- `LLM_HTTP_TRUST_ENV` (default: `false`; currently applied to the Google provider path)
- `PROJECT_SHEET_RANGE` (default: `Tasks!A1:Z`)
- `PROJECT_SHEET_CACHE_TTL_SECONDS` (default: `30`)
- `SLACK_THINKING_REACTION` (default: `eyes`)
- `KNOWLEDGE_BASE_DIR` (default: `knowledge`)
- `KNOWLEDGE_FILE_TYPES` (default: `.md,.txt,.rst,.csv,.tsv,.xlsx,.xlsm`)
- `JADE_PROJECT_SKILLS_DIR` (default: `.jade/skills`)
- `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH` (default: `knowledge/AI/Rules/google_sheets_catalog.json`)
- `KNOWLEDGE_GOOGLE_SHEETS_CACHE_TTL_SECONDS` (default: `120`)
- `CONVERSION_WORK_DIR` (default: `runtime/conversion`)
- `WEB_ALLOWED_HOSTS` (comma-separated allowed `Host` headers such as `jade.example.com`; recommended when publishing the web app)
- `WEB_AUTH_ENABLED` (default: `false`)
- `WEB_AUTH_USERNAME` / `WEB_AUTH_PASSWORD` (simple single-user web login)
- `WEB_AUTH_CREDENTIALS` (comma-separated `username:password` pairs for multi-user web login)
- `WEB_AUTH_SESSION_SECRET` (required when `WEB_AUTH_ENABLED=true`)
- `WEB_AUTH_COOKIE_SECURE` (default: `true`; set `false` only for local non-HTTPS testing)
- `WEB_AUTH_SESSION_MAX_AGE_SECONDS` (default: `43200`, which is 12 hours)

Provider examples:

```bash
# Google
LLM_PROVIDER=google
LLM_MODEL=gemini-3-flash-preview
GOOGLE_API_KEY=...

# MiniMax
LLM_PROVIDER=minimax
LLM_MODEL=MiniMax-M2.7-highspeed
MINIMAX_API_KEY=...
MINIMAX_BASE_URL=https://api.minimaxi.com/v1

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-mini
OPENAI_API_KEY=...
```

Split routing example:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4
OPENAI_API_KEY=...

ROUTING_LLM_PROVIDER=minimax
ROUTING_LLM_MODEL=MiniMax-M2.7-highspeed
ROUTING_MINIMAX_API_KEY=...
ROUTING_MINIMAX_BASE_URL=https://api.minimaxi.com/v1
```

`GOOGLE_APPLICATION_CREDENTIALS` and `JADE_PROJECT_SHEET_ID` remain required for the Google Sheets/Docs integrations regardless of which LLM provider is selected.

## Optional Slack setup

Slack remains available for teams that still want a chatops surface or upload-driven flows such as document conversion follow-ups. The bot expects Socket Mode.

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

For the standalone KB builder experience, the web app is the main entrypoint:

```bash
WEB_ENABLED=true
SLACK_ENABLED=false
python main.py
```

Then open `http://127.0.0.1:8000`.

If you also want Slack enabled, keep the Slack tokens configured and set `SLACK_ENABLED=true`.

## Hosting the web interface

If you plan to publish the web UI on a company HTTPS URL, turn on the built-in login and host guard:

```bash
WEB_ENABLED=true
WEB_HOST=127.0.0.1
WEB_PORT=8000
WEB_ALLOWED_HOSTS=jade.example.com
WEB_AUTH_ENABLED=true
WEB_AUTH_SESSION_SECRET=replace-with-a-long-random-secret
WEB_AUTH_CREDENTIALS=alice:replace-me,bob:replace-me-too
WEB_AUTH_COOKIE_SECURE=true
```

Notes:
- Keep `WEB_AUTH_SESSION_SECRET` and the web credentials in your runtime environment or secret manager, not in the repo.
- `WEB_ALLOWED_HOSTS` is the hostname allowlist for the served app and helps protect against bad `Host` headers.
- The built-in login is intended for internal hosting behind HTTPS. For a larger rollout, putting this behind your company SSO or reverse-proxy auth is still the stronger long-term option.
- If you are publishing through Cloudflare Tunnel, keep Jade on `127.0.0.1` and let `cloudflared` expose the local origin.
- A ready-to-fill deployment bundle for `chat.jade-games.com` is available in [`deploy/README.md`](/Users/kayngame/jade_ai_core/deploy/README.md).

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
