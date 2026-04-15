# MAS Refactor Status

## Current Status

The Part 2 routing refactor is complete and live.

The codebase now has one production routing story:

- one worker agent per turn
- `pending_action` has priority over fresh-request routing
- fresh requests go through `AssistantRequest`
- pending-action replies go through `PendingActionDecision`
- the runtime emits `AgentRouteDecision`
- top-level fallback is deterministic to `general_chat_agent`

## Completed In Part 2

### Part 2A - Contracts and Lightweight Parser Foundation

Completed:

- shared routing contracts in [app/contracts/](/Users/kayngame/jade_ai_core/app/contracts)
- lightweight parser prompts in [app/interpretation/prompts/](/Users/kayngame/jade_ai_core/app/interpretation/prompts)
- parser service in [app/interpretation/intent_parser.py](/Users/kayngame/jade_ai_core/app/interpretation/intent_parser.py)
- schema and malformed-output tests

### Part 2B - Pending Action Routing

Completed:

- all active pending-action replies route through `PendingActionDecision`
- `app/pending_actions.py` resolves validated contracts instead of raw text matching
- ambiguous pending replies do not silently continue execution

### Part 2C - Top-Level Request Routing

Completed:

- fresh requests route through `AssistantRequest`
- domains map through [app/routing/domain_map.py](/Users/kayngame/jade_ai_core/app/routing/domain_map.py)
- the gateway emits `AgentRouteDecision`

### Part 2D - Shadow Mode

Completed and removed:

- parser-vs-legacy shadow comparison ran during the migration
- shadow-only comparison code has been deleted after switchover

### Part 2E - Production Switchover

Completed:

- parser-based routing is the only production top-level text route-selection path
- legacy heuristic text routing is no longer used by the gateway
- deterministic fallback remains explicit and logged

### Part 2F - Docs, Tests, and Hardening

Completed:

- architecture docs updated to match live routing
- routing README added
- parser failure logging added
- confidence thresholds centralized through parser config
- routing regression coverage kept for knowledge, project task, KB builder, document conversion, pending actions, and fallback paths

## Current Runtime Notes

The parser is allowed to:

- interpret user intent into a routing contract
- interpret pending-action replies into a pending-action contract
- express uncertainty through confidence and notes

The parser is not allowed to:

- call tools
- execute actions
- choose arbitrary workflows
- bypass contract validation

## Follow-On Work

Likely future work after Part 2:

- skill/tool standardization on top of the stabilized routing model
- cleanup of any remaining dead heuristic-era helper modules outside the live path
