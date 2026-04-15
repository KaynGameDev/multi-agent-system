# Jade MAS Architecture

## Overview

Jade is a LangGraph-based multi-agent assistant with one production routing story:

1. Interfaces add the user turn into state.
2. The gateway selects exactly one worker agent for the turn.
3. If an active `pending_action` exists, it has priority over fresh-request routing.
4. Otherwise, a lightweight parser converts the user turn into an `AssistantRequest`.
5. The gateway maps that contract to one worker agent and falls back deterministically to `general_chat_agent` when needed.
6. The selected worker agent handles the turn and may enter a tool loop, but the gateway is not re-run inside that same turn.

The runtime no longer uses legacy top-level heuristic text matching for production routing.

## Runtime Model

### One Worker Agent Per Turn

Every turn executes exactly one worker agent after the gateway finishes route selection.

- The gateway writes `route`, `route_reason`, `route_policy_step`, `agent_route_decision`, and `routing_decision`.
- The graph dispatches to the selected worker node once.
- If that worker uses LangGraph tools, the tool loop returns to the same worker agent.

### Pending-Action Priority

`PendingAction` is the only waiting/confirmation model.

At turn start:

1. The gateway checks for an active `pending_action`.
2. If present, `PendingActionRouter` parses the reply into `PendingActionDecision`.
3. If the decision continues the pending action, the gateway routes back to the owning agent.
4. If the decision is `unrelated`, the gateway stops the pending-action path and treats the turn as a fresh request.

This keeps confirmation, selection, clarification, and cancellation flows deterministic across agents.

### Fresh-Request Parser Flow

When there is no active pending-action continuation to honor:

1. `IntentParser.parse_assistant_request(...)` produces an `AssistantRequest`.
2. The contract is validated.
3. `AgentRouter` maps `likely_domain` through the shared domain map.
4. The gateway emits `AgentRouteDecision`.
5. The selected worker agent executes.

Supported top-level domains:

- `general`
- `knowledge`
- `project_task`
- `knowledge_base_builder`
- `document_conversion`

Domain mapping lives in [app/routing/domain_map.py](/Users/kayngame/jade_ai_core/app/routing/domain_map.py).

### Deterministic Fallback

If the top-level parser output is invalid, low-confidence, missing a supported domain, or maps to an unavailable agent, the runtime falls back to `general_chat_agent`.

Fallback is explicit in:

- `route`
- `route_reason`
- `route_policy_step`
- `agent_route_decision.fallback_used`
- `agent_selection_diagnostics`

## Core Contracts

Contracts live in [app/contracts/](/Users/kayngame/jade_ai_core/app/contracts).

### `AssistantRequest`

Used only for fresh top-level requests.

Fields:

- `type = "assistant_request"`
- `user_goal`
- `likely_domain`
- `confidence`
- `notes`

### `PendingActionDecision`

Used only when an active `pending_action` exists.

Fields:

- `type = "pending_action_decision"`
- `pending_action_id`
- `decision`
- `notes`
- `selected_item_id`
- `constraints`

Supported `decision` values:

- `approve`
- `reject`
- `modify`
- `select`
- `unrelated`
- `unclear`

### `AgentRouteDecision`

Produced by the runtime router after parsing.

Fields:

- `selected_agent`
- `reason`
- `fallback_used`
- `diagnostics`

### Other Shared Contracts

The runtime also standardizes:

- `PendingAction`
- `ExecutionContract`
- `SkillInvocationContract`
- `ToolInvocationEnvelope`
- `ToolResultEnvelope`
- `AssistantResponse`

## Parser Responsibilities

The lightweight parser is allowed to:

- interpret the user turn into a routing or pending-action contract
- summarize intent into a normalized `user_goal`
- classify into the supported schema fields
- report low confidence through the contract

The lightweight parser is not allowed to:

- call tools
- execute workflows
- choose tools directly
- branch outside the supported schema
- bypass pending-action rules
- emit arbitrary runtime actions that are not represented by the contracts

## Routing Modules

### Active Runtime Path

- [gateway/agent.py](/Users/kayngame/jade_ai_core/gateway/agent.py): production gateway entrypoint
- [app/routing/agent_router.py](/Users/kayngame/jade_ai_core/app/routing/agent_router.py): top-level parser route resolution
- [app/routing/domain_map.py](/Users/kayngame/jade_ai_core/app/routing/domain_map.py): domain-to-agent mapping
- [app/routing/pending_action_router.py](/Users/kayngame/jade_ai_core/app/routing/pending_action_router.py): pending-action routing entry
- [app/routing/routing_diagnostics.py](/Users/kayngame/jade_ai_core/app/routing/routing_diagnostics.py): structured diagnostics builders
- [app/interpretation/intent_parser.py](/Users/kayngame/jade_ai_core/app/interpretation/intent_parser.py): lightweight parser service
- [app/interpretation/model_config.py](/Users/kayngame/jade_ai_core/app/interpretation/model_config.py): parser thresholds and prompt configuration

### Still Active But Not Top-Level Text Routing

- [gateway/skill_policy.py](/Users/kayngame/jade_ai_core/gateway/skill_policy.py): explicit and automatic skill selection

Skill logic can still influence which instructions run for the selected agent, and explicit internal overrides can still select an agent deterministically, but free-form top-level route selection is contract-based.

## Diagnostics and Hardening

Routing diagnostics are emitted for every turn.

Fresh-request routing diagnostics include:

- parsed `AssistantRequest`
- selected agent
- `policy_step`
- `fallback_used`
- route reason

Pending-action routing diagnostics include:

- pending-action owner
- pending-action id
- pending-action decision
- fallback details when the owner is unavailable

Parser hardening includes:

- confidence thresholds configured in one place via [app/interpretation/model_config.py](/Users/kayngame/jade_ai_core/app/interpretation/model_config.py)
- validation of parser output before execution
- logged parser failures and malformed output fallback
- deterministic fallback to `general_chat_agent`

## Current Worker Agents

- `general_chat_agent`
- `knowledge_agent`
- `project_task_agent`
- `knowledge_base_builder_agent`
- `document_conversion_agent`

## Current Routing Story

The live routing order is:

1. continue active pending action if applicable
2. otherwise honor explicit internal overrides that are already structured state:
   requested agent
   explicit forked-skill delegate/fallback
   explicit inline-skill compatibility
3. otherwise route through `AssistantRequest`
4. otherwise fall back deterministically to `general_chat_agent`

There is no longer a shadow-mode comparison path or a legacy heuristic top-level text router in production.
