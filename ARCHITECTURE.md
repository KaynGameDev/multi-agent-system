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
• document_conversion_agent

Agents should be responsible for reasoning but not platform formatting.

3. Gateway Layer
A gateway agent decides which worker agent should handle the user request.

Routing currently supports:
• project_task_agent
• general_chat_agent
• knowledge_agent
• document_conversion_agent

The gateway uses an LLM classification step to determine intent.

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

Project query route:
gateway → project_task_agent → tools → project_task_agent → END

This allows tool-calling loops.

5. Interface Layer
Interface listeners are responsible for:
• receiving platform events
• resolving user identity where possible
• invoking the graph
• formatting output for the target platform

Slack formatting is applied only at the Slack boundary.

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
core/slack_formatting.py

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
core/identity_map.py

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
tools/google_sheets.py

Important rule:
Tools must NOT contain Slack formatting logic.

State Model
The LangGraph state contains:

messages
route
route_reason
user_id
channel_id
user_display_name
user_real_name
user_email
user_google_name
user_sheet_name
user_job_title

Defined in:
core/state.py

Graph Definition
The orchestration graph is defined in:
core/graph.py

It wires:
gateway
general_chat_agent
project_task_agent
tool execution

Interface Entry Points
Slack integration is implemented in:
interfaces/slack_listener.py

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

Current v1 knowledge source:
• local files under `KNOWLEDGE_BASE_DIR`
• recommended default folder: `data/knowledge/`
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
