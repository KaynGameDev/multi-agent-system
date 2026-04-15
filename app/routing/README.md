# Routing README

## Purpose

`app/routing/` contains the live routing runtime for Jade.

The routing story is intentionally small:

- pending-action routing first
- parser-based fresh-request routing second
- deterministic fallback last

## Modules

### `agent_router.py`

Turns a validated `AssistantRequest` into:

- selected agent
- `policy_step`
- `AgentRouteDecision`
- structured diagnostics

This module does not execute anything. It only maps contracts to route decisions.

### `domain_map.py`

Defines the supported top-level domain map:

- `general -> general_chat_agent`
- `knowledge -> knowledge_agent`
- `project_task -> project_task_agent`
- `knowledge_base_builder -> knowledge_base_builder_agent`
- `document_conversion -> document_conversion_agent`

### `pending_action_router.py`

Handles only turns with an active `pending_action`.

It:

- parses the reply into `PendingActionDecision`
- resolves the shared pending-action contract
- decides whether the turn should stay on the pending action or allow fresh routing

### `routing_diagnostics.py`

Produces structured dictionaries used by the gateway for:

- parsed request diagnostics
- selected route diagnostics

## What The Parser Can Do

The parser can:

- normalize a user turn into `AssistantRequest`
- normalize a pending-action reply into `PendingActionDecision`
- express uncertainty through confidence and notes

## What The Parser Cannot Do

The parser cannot:

- call tools
- choose a tool directly
- execute an action
- invent workflow branches outside the contract schema
- bypass pending-action precedence

## Confidence Thresholds

Parser thresholds are configured in [model_config.py](/Users/kayngame/jade_ai_core/app/interpretation/model_config.py).

`AgentRouter` reads the parser threshold from the parser config by default, so there is one default confidence source for parser-based routing.

## Fallback Rules

Top-level routing falls back to `general_chat_agent` when:

- the parser is unavailable
- the parser output is malformed
- the parser confidence is below threshold
- the parsed domain is missing
- the parsed domain is unsupported
- the mapped agent is unavailable

Pending-action routing never silently converts an unclear reply into execution. It returns an unclear decision and keeps the flow deterministic.
