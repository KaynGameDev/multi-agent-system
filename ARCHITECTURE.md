# Jade MAS Architecture

## Overview

Jade is a deterministic multi-agent assistant built on LangGraph. It serves internal company users through Slack and web chat, routes each turn through an explicit gateway policy, and standardizes confirmations, skill execution, tool execution, and final responses through shared runtime contracts.

The supported runtime model is:

1. Interfaces collect request context and append messages.
2. The gateway chooses the worker agent deterministically.
3. The worker agent either handles a pending action, renders a standardized tool result, or invokes the model.
4. Tool execution flows through one shared envelope model.
5. The final user-visible answer is emitted through `assistant_response`.

The runtime no longer relies on legacy waiting models or prompt-only routing semantics.

## Core Layers

### Interfaces

Interfaces are delivery boundaries only.

- Slack: `interfaces/slack/listener.py`
- Web: `interfaces/web/server.py`

Interface responsibilities:

- receive inbound events
- resolve user and thread metadata
- append user messages to graph state
- invoke the graph
- format the final response for the target channel

Interfaces do not force worker routes. They pass request context to the gateway and return gateway diagnostics to the caller.

### Graph

The orchestration graph is defined in `app/graph.py`.

Current flow:

```text
START
  -> gateway
  -> selected worker agent
  -> optional ToolNode loop
  -> END
```

Agents with LangGraph tools loop through their `ToolNode` and then return to the same agent. The gateway is not rerun inside the same tool loop.

### Gateway

The gateway is a deterministic policy layer implemented across:

- `gateway/agent.py`
- `gateway/routing_policy.py`
- `gateway/skill_policy.py`
- `gateway/tool_intent.py`
- `gateway/matchers.py`
- `gateway/text_utils.py`

The gateway owns:

- agent routing order
- requested-agent handling
- skill-based delegation
- pending-action owner short-circuits
- tool-intent routing
- deterministic matcher routing
- fallback policy
- routing diagnostics

### Shared Runtime

Shared runtime modules define the supported control model:

- `app/contracts.py`
- `app/state.py`
- `app/messages.py`
- `app/pending_actions.py`
- `app/skill_runtime.py`
- `app/tool_runtime.py`
- `app/skills.py`
- `app/tool_registry.py`

Agents are expected to differ by prompt, tools, and business purpose, not by private confirmation or tool-result semantics.

## Turn Flow

### 1. Request enters state

Interfaces populate graph state with:

- `messages`
- thread and user metadata
- uploaded file metadata when present
- optional `requested_agent`
- optional `requested_skill_ids`
- optional `context_paths`

### 2. Gateway resolves the route

The gateway preserves this routing order:

1. explicit requested agent
2. explicit forked-skill delegates
3. forked-skill fallback to `GeneralAssistant`
4. explicit inline-skill-compatible agents
5. active pending-action owner
6. tool-intent metadata match
7. deterministic matcher scores
8. general-assistant fallback

The gateway records:

- `route`
- `route_reason`
- `route_policy_step`
- `routing_decision`
- `skill_resolution_diagnostics`
- `agent_selection_diagnostics`
- `selection_warnings`

### 3. Gateway emits active skill contracts

The gateway resolves explicit and automatic skill matches into `SkillInvocationContract` entries and stores:

- `skill_invocation_contracts`
- `active_skill_invocation_contracts`
- `skill_execution_diagnostics`
- `resolved_skill_ids`

The contract is the source of truth for what skill ran, why it ran, and which agent executes it.

### 4. Worker agent handles the turn

Each worker agent follows the same runtime shape:

1. if this agent owns an active `pending_action`, resolve the user's reply through the shared pending-action interpreter
2. else, if the latest message is a standardized tool result, render it deterministically when appropriate
3. else, build the prompt and invoke the model

### 5. Tool execution uses one envelope model

Tool calls are normalized into `ToolInvocationEnvelope`.

Tool results are normalized into `ToolResultEnvelope`.

Both LangGraph `ToolNode` results and internal document-conversion workflow steps use the same envelope fields, including:

- tool name
- tool id
- display name
- tool family
- execution backend
- arguments
- status
- payload
- diagnostics
- source
- reason
- tool call id

The runtime also records `tool_execution_trace` so the final state can explain which tool-like operations ran during the turn.

### 6. Final answer is emitted through `assistant_response`

The final state may still contain raw messages, but `assistant_response` is the canonical structured output. `app/messages.py` prefers `assistant_response` content over the last raw message when extracting the final answer.

## Shared Runtime Contracts

The contract vocabulary lives in `app/contracts.py`.

### Assistant response

`AssistantResponse` is the final structured output container. It may carry:

- final text content
- active `pending_action`
- `execution_contract`
- selected `skill_invocation`
- `tool_invocation`
- `tool_result`
- `routing_decision`

### Pending action

`PendingAction` is the only supported waiting/confirmation model.

It describes:

- what the system wants to do
- which agent requested it
- the scoped target
- the risk level
- any selection options or follow-up metadata

### Execution contract

When the user replies to a pending action, the runtime interprets that reply into an `ExecutionContract` and validates it into one of these runtime actions:

- `execute`
- `cancel`
- `request_revision`
- `ask_clarification`
- `select`

This lets all agents share the same deterministic confirmation model.

### Skill invocation contract

`SkillInvocationContract` captures:

- skill id and name
- description
- source path
- mode (`inline` or `fork`)
- target agent
- invocation source
- invocation reason
- context paths

The prompt consumes runtime-selected skill instructions. The prompt does not decide routing or delegation.

### Tool invocation and result envelopes

`ToolInvocationEnvelope` and `ToolResultEnvelope` standardize both normal tool-node work and internal workflow operations.

## Pending Actions and Confirmation

Pending-action logic lives in `app/pending_actions.py`.

Supported flows:

- explicit approval
- rejection
- narrowing or modification requests
- request-for-preview or diff before execution
- deterministic option selection
- ambiguity detection

Current agent usage:

- `knowledge_agent`: document selection and document follow-up
- `project_task_agent`: task selection and task-detail follow-up
- `knowledge_base_builder_agent`: confirmation-gated KB writes
- `document_conversion_agent`: approval-gated package publishing

The gateway routes active pending actions back to the owning agent before normal matcher routing runs.

## Skills

Skill discovery and normalization are implemented in `app/skills.py`.

Supported skill scopes:

- path-scoped project skills
- agent-local skills
- project-shared skills

Precedence:

`path-scoped > agent-local > project-shared`

Execution modes:

- `inline`: stays in the current agent
- `fork`: routes to the delegate agent

If a forked skill has no active delegate agent, the gateway falls back to `GeneralAssistant`.

The skill runtime is implemented in `app/skill_runtime.py`. It filters active contracts for the executing agent and attaches the selected skill's instruction body to the prompt as runtime-selected context.

## Tools

Tool metadata and routing hints live in `app/tool_registry.py`.

Runtime normalization lives in `app/tool_runtime.py`.

Standard tool families currently include:

- knowledge read
- knowledge write
- project tracker
- document conversion internal operations

Tool metadata powers:

- gateway tool-intent routing
- normalized tool ids
- display labels
- runtime tracing

## Agents

### `general_chat_agent`

Purpose:

- greetings
- general chat
- fallback conversational help

It has no business-specific tools and acts as the `GeneralAssistant` fallback in gateway policy.

### `knowledge_agent`

Purpose:

- repository guidance
- setup and architecture questions
- internal document lookup and reading

It uses knowledge read tools and shared pending-action selection for follow-up document selection.

### `knowledge_base_builder_agent`

Purpose:

- knowledge elicitation
- KB review
- layer-placement guidance
- feature-spec skeleton support
- KB file drafting

It uses read tools plus builder-only path resolution and KB write tools. KB file mutation is always confirmation-gated.

### `project_task_agent`

Purpose:

- task lookup
- assignee and deadline questions
- tracker overviews

It uses project tracker tools, renders task results deterministically, and uses shared pending-action selection for task-detail follow-ups.

### `document_conversion_agent`

Purpose:

- ingest uploaded files and Google document references
- extract structured draft packages
- ask for missing required conversion details
- stage packages
- publish only after approval

It uses the same shared pending-action confirmation model as the other agents, but its internal workflow steps are wrapped as standardized internal tool envelopes.

## State Model

The shared graph state is defined in `app/state.py`.

Important runtime fields include:

- `messages`
- `route`
- `route_reason`
- `route_policy_step`
- `pending_action`
- `execution_contract`
- `assistant_response`
- `routing_decision`
- `skill_invocation_contracts`
- `active_skill_invocation_contracts`
- `skill_execution_diagnostics`
- `tool_invocation`
- `tool_result`
- `tool_execution_trace`
- `requested_agent`
- `requested_skill_ids`
- `resolved_skill_ids`
- `context_paths`

## Knowledge and Data Sources

Knowledge-base content is organized under `knowledge/`.

Current major areas:

- `knowledge/Docs/00_Shared/`
- `knowledge/Docs/10_GameLines/`
- `knowledge/Docs/20_Deployments/`
- `knowledge/Docs/30_Review/`
- `knowledge/Docs/40_Legacy/`
- `knowledge/Docs/50_Templates/`

The project tracker source is Google Sheets, via `tools/project_tracker_google_sheets.py`.

Document conversion can also ingest Google document references through:

- `tools/conversion_google_sources.py`
- `tools/google_workspace_services.py`

## Legacy Cutoff

The following systems are no longer supported as live runtime behavior:

- `pending_interaction`
- regex-only approval or cancel parsing as an agent-local control path
- prompt-only skill routing or delegation

Historical web transcripts may still be normalized on load for readability, but that is history cleanup only and not part of the live runtime control model.
