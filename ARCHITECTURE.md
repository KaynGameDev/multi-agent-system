Project Name: Jade Games Multi-Agent System (MAS)

Overview
We are building a company-level AI Multi-Agent System in Python to power an internal chat assistant. The system is designed using LangGraph and LangChain and integrates with company tools such as Google Sheets to answer operational questions like project tasks, ownership, schedules, and status.

The assistant currently runs as:
• a Slack bot using Slack Bolt Socket Mode

Primary goals:
• Provide a scalable multi-agent architecture
• Integrate company operational data sources (Google Sheets first)
• Allow the Slack bot to answer questions like:
  - "What are my tasks?"
  - "What is Liu Yu working on?"
  - "Which tasks are due this week?"
• Maintain a clean separation between data tools, agents, and communication layers

Architecture Philosophy
The architecture intentionally separates responsibilities into layers:

1. Tools Layer
Tools access external systems such as Google Sheets.
Tools should remain:
• clean
• reusable
• presentation-agnostic
They must not contain Slack formatting or UI logic.

2. Agent Layer
Agents reason about tasks and decide when to call tools.
Agents include:
• project_task_agent
• general_chat_agent
• knowledge_agent
• knowledge_base_builder_agent
• document_conversion_agent

Agents should be responsible for reasoning but not platform formatting.

3. Gateway Layer
A deterministic gateway decides which worker agent should handle the user request.

Routing currently supports:
• project_task_agent
• general_chat_agent
• knowledge_agent
• knowledge_base_builder_agent
• document_conversion_agent

The gateway uses policy code instead of LLM classification. It centralizes:
• deterministic agent selection
• shared skill precedence
• fallback routing
• forked-skill delegation
• selection diagnostics

Defined in:
gateway/agent.py

4. Graph Layer
LangGraph orchestrates the workflow.

Current flow:

START
 → gateway
 → route to worker agent

General chat route:
gateway → general_chat_agent → END

Knowledge route:
gateway → knowledge_agent → tools → knowledge_agent → END

KB builder route:
gateway → knowledge_base_builder_agent → tools → knowledge_base_builder_agent → END

Project query route:
gateway → project_task_agent → tools → project_task_agent → END

This allows tool-calling loops.

Resolved skills stay attached to the turn state, and tool callbacks do not rerun gateway selection.

5. Skill Registry Layer
Skills resolve through one shared registry.

Current scopes:
• agent-local skills under `agents/<agent>/Skills/*/SKILL.md`
• project-shared skills under `.jade/skills/*/SKILL.md` or `JADE_PROJECT_SKILLS_DIR`

Resolution policy:
• every skill normalizes into one `SkillDefinition`
• one `skill_id` maps to one effective definition per request
• same-scope duplicates are configuration conflicts
• precedence is `path-scoped > agent-local > project-shared`

Execution policy:
• `inline` skills stay in the current agent context
• `forked` skills with `delegate_agent` route to that agent
• `forked` skills without `delegate_agent` route to `GeneralAssistant`
• if `GeneralAssistant` is missing, the first active agent is used and a warning is recorded

6. Interface Layer
Interface listeners are responsible for:
• receiving platform events
• resolving user identity where possible
• invoking the graph
• formatting output for the target platform

Background services can also live at this layer when they own delivery concerns such as outbound Slack alerts.

Slack formatting is applied only at the Slack boundary.
Transport layers no longer inject route overrides. They pass raw request context to the gateway.

Formatting Strategy
We use a 3-layer formatting approach:

Layer 1: Tools
Tools output plain structured data.

Layer 2: Agent Prompts
Agents are instructed to produce Slack-friendly responses.

Layer 3: Slack Boundary
A Slack formatter converts Markdown to Slack mrkdwn.

Example conversions:
**bold** → *bold*
# Heading → *Heading*
[text](url) → <url|text>
- bullet → • bullet

This formatter lives in:
interfaces/slack/formatting.py

Identity Resolution
The system maps Slack users to internal employee identities using an IDENTITY_MAP.

Users may reference people via:
• Slack display name
• email
• Chinese name
• English name
• Slack mention

The system resolves these to the canonical Google Sheets name.

File:
app/identity.py

Google Sheets Integration
The project task data source is a Google Sheet.

Columns:
A-Q

Headers include:
迭代
人员
内容
平台
项目
start
end
提测日期
更新日期
Color
开发天数
测试天数
客户端
服务器
测试
产品
优先级

The tool supports searching by:
• assignee
• project
• platform
• priority
• iteration
• free text query

Files:
tools/project_tracker_google_sheets.py
tools/conversion_google_sources.py
tools/google_workspace_services.py

Important rule:
Tools must NOT contain Slack formatting logic.

State Model
The LangGraph state contains:

messages
route
route_reason
requested_agent
requested_skill_ids
resolved_skill_ids
context_paths
skill_resolution_diagnostics
agent_selection_diagnostics
selection_warnings
user_id
channel_id
user_display_name
user_real_name
user_email
user_google_name
user_sheet_name
user_job_title

Defined in:
app/state.py

Graph Definition
The orchestration graph is defined in:
app/graph.py

It wires:
gateway
general_chat_agent
project_task_agent
knowledge_agent
knowledge_base_builder_agent
document_conversion_agent
tool execution

Interface Entry Points
Slack integration is implemented in:
interfaces/slack/listener.py

Web chat integration is implemented in:
interfaces/web/server.py

Responsibilities:
• handle incoming platform events
• resolve user identity when available
• send messages to the graph
• apply interface-specific formatting before sending responses

Current Agents
GeneralChatAgent
Handles greetings and general conversation.

KnowledgeAgent
Handles internal documentation, architecture, setup, and repository guidance questions.

KnowledgeBaseBuilderAgent
Handles knowledge elicitation, KB document review, layer placement decisions, Feature Spec skeleton building, and KB V1 execution tracking. It uses read-only knowledge tools for evidence gathering, can resolve canonical KB draft paths, and can write Markdown drafts into the knowledge base when explicitly asked, but it does not auto-publish or auto-promote legacy materials.

Current v1 knowledge source:
• local files under `KNOWLEDGE_BASE_DIR`
• recommended default folder: `knowledge/`
• default layout groups AI workspace files under `knowledge/AI/` and curated documentation under `knowledge/Docs/`
• supported formats include Markdown/text and Excel exports (`.xlsx`, `.xlsm`)
• curated online Google Sheets listed in `KNOWLEDGE_GOOGLE_SHEETS_CATALOG_PATH`

Online knowledge-sheet catalog:
• one JSON entry per spreadsheet
• each entry defines `spreadsheet_id`
• optional `title` and `aliases` improve document resolution
• optional `tabs` restricts which tabs are searchable
• optional `ranges` restricts tab reads to approved A1 ranges
• online sheets are rendered into the same semantic block pipeline used for local CSV/XLSX docs

ProjectTaskAgent
Handles queries that require project data and can call tools.

DocumentConversionAgent
Handles Slack-uploaded design docs, asks follow-up questions when required conversion fields are missing, stages canonical knowledge packages, and publishes only after approval.

Future agents that may be added:
• jira_agent
• document_agent
• calendar_agent

Current Objective
The system currently works but is still early-stage.

Next improvements should focus on:

1. improving intent routing accuracy
2. improving project task query reasoning
3. supporting richer Google Sheets queries
4. improving Slack UX
5. adding memory and conversation context
6. preparing the system for additional agents

Development Constraints
• Python 3.12
• LangGraph
• LangChain
• Slack Bolt Socket Mode
• Google Sheets API (service account)

Key Design Rule
Tools should stay platform-agnostic.
Formatting should only happen at the Slack boundary.

Your job is to help extend and maintain this architecture while preserving these design principles.
