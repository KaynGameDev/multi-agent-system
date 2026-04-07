# MAS Refactor Status

## Part 1 - Confirmation / Waiting Cleanup

Removed:
- Legacy `pending_interaction` routing from the gateway and agent follow-up paths.
- Raw-text approval regex fallback from the knowledge-base write gate.
- Local approval/cancel regex handling from document conversion.

Added:
- Shared pending-action selection and confirmation resolution in `app/pending_actions.py`.
- Pending-action follow-up handling for knowledge, project task, KB builder, and document conversion.
- Updated tests covering pending-action selection, KB write approval, and document conversion approve/cancel/clarification flows.

Remains:
- Runtime contract foundation.
- Skill system refactor.
- Tool-calling standardization.
- Gateway routing cleanup.
- Agent migration and deletion.
- Docs, tests, and cutoff cleanup.

## Part 2 - Runtime Contracts Foundation

Removed:
- Builder-local assumptions for pending-action and execution-contract typing.
- Direct final-output dependence on the last message `.content` when a shared assistant response is present.
- Some raw tool payload trust in the knowledge, project task, KB builder, and gateway follow-up paths.

Added:
- Shared runtime contract vocabulary in `app/contracts.py` for assistant responses, pending actions, execution contracts, skill invocations, tool invocation/result envelopes, and routing decisions.
- Contract-aware state fields in `app/state.py`.
- Contract-aware extraction in `app/messages.py`.
- Registry support for skill invocation contracts in `app/skills.py`.
- Assistant response and envelope emission in the general chat, knowledge, project task, KB builder, and document conversion agents.
- Normalized tool-result adapters and new contract tests in `tests/test_runtime_contracts.py`.

Remains:
- Skill system refactor.
- Tool-calling standardization.
- Gateway routing cleanup.
- Agent migration and deletion.
- Docs, tests, and cutoff cleanup.

## Part 3 - Skill System Refactor

Removed:
- Direct agent-side `SKILL.md` body injection as the primary live execution path.
- Hidden prompt-only skill execution semantics spread across general chat, knowledge, project task, KB builder, and document conversion prompt assembly.

Added:
- Shared skill runtime adapter in `app/skill_runtime.py` for active-skill filtering, runtime diagnostics, and runtime-selected `SKILL.md` instruction attachment.
- Richer `SkillInvocationContract` metadata in `app/contracts.py` and `app/skills.py`, including skill name, description, source path, explicit invocation source, and explicit invocation reason.
- Gateway-emitted active skill runtime state in `gateway/agent.py`, including `active_skill_invocation_contracts` and `skill_execution_diagnostics`.
- Prompt builders that consume runtime-selected skill contracts instead of `resolved_skill_ids`.
- Regression coverage proving prompts now read runtime contracts and that skill execution state survives routing and tool loops.

Remains:
- Tool-calling standardization.
- Gateway routing cleanup.
- Agent migration and deletion.
- Docs, tests, and cutoff cleanup.

## Part 4 - Tool Calling Standardization

Removed:
- More agent-local `ToolMessage` parsing assumptions spread across knowledge, project task, and KB builder.
- The document-conversion path fully bypassing shared tool invocation/result envelopes for its main internal workflow steps.

Added:
- Shared tool runtime adapter in `app/tool_runtime.py` for:
  - normalizing model-emitted tool invocations
  - reconstructing `ToolNode` results from `ToolMessage` plus prior tool-call context
  - wrapping internal workflow operations with the same invocation/result envelope model
- Richer tool envelopes in `app/contracts.py`, including tool metadata, execution backend, and tool execution trace records.
- Tool metadata entries for internal document-conversion operations in `app/tool_registry.py`.
- Standardized tool-result consumption in:
  - `agents/knowledge/agent.py`
  - `agents/project_task/agent.py`
  - `agents/knowledge_base_builder/agent.py`
- Internal tool-envelope wrapping for document conversion ingestion, draft extraction, staging, and publishing in `agents/document_conversion/agent.py`.
- Runtime tool execution trace state in `app/state.py`.
- Regression coverage for ToolMessage adaptation and internal publish envelopes in `tests/test_runtime_contracts.py` and `tests/test_document_conversion_confirmation.py`.

Remains:
- Gateway routing cleanup.
- Agent migration and deletion.
- Docs, tests, and cutoff cleanup.

## Part 5 - Gateway Routing Cleanup

Removed:
- The single-file gateway hotspot carrying normalization, routing precedence, tool-intent parsing, matcher orchestration, and skill-selection helpers all in `gateway/agent.py`.
- Any remaining live gateway routing dependence on legacy `pending_interaction`.

Added:
- Focused gateway policy modules:
  - `gateway/routing_policy.py` for routing-order execution, fallback handling, and policy-step diagnostics
  - `gateway/skill_policy.py` for explicit skill resolution, skill eligibility, and deterministic auto-skill selection
  - `gateway/tool_intent.py` for tool-intent metadata detection and routing
  - `gateway/matchers.py` for agent matcher definitions and matcher execution helpers
  - `gateway/text_utils.py` for shared normalization and greeting detection
- Explicit routing policy-step diagnostics via `route_policy_step` state and `routing_decision.policy_step`.
- Regression coverage in `tests/test_gateway.py` that freezes the audited precedence order across:
  - requested agent
  - explicit forked-skill delegates
  - forked-skill fallback
  - explicit inline-skill-compatible routing
  - pending-action owner short-circuit
  - tool-intent routing
  - deterministic matcher routing
  - general fallback
- Contract support for structured routing policy-step reporting in `app/contracts.py`.

Remains:
- Agent migration and deletion.
- Docs, tests, and cutoff cleanup.

## Part 6 - Agent Migration and Legacy Deletion

Removed:
- The unused legacy pending-interaction compatibility module at `app/pending_interactions.py`.
- The legacy-named `tests/test_pending_interactions.py` suite in favor of current-runtime migration coverage.
- Builder-local retry-tool reconstruction helpers and duplicate latest-user-text parsing now superseded by shared runtime helpers.
- The project-task agent's model-dependent tool-result summarization path for task-list results.

Added:
- Shared pending-action tool reconstruction helpers in `app/tool_runtime.py` so agents can replay confirmed tool calls without private adapters.
- Shared message-content extraction usage across knowledge, KB builder, project task, and document conversion via `app/messages.py`.
- Deterministic tool-result rendering for `agents/project_task/agent.py`, including pending-action setup from shared tool envelopes before any LLM fallback path.
- Migration regression coverage in `tests/test_agent_runtime_migration.py` for:
  - knowledge selection follow-ups
  - deterministic project-task rendering and follow-up selection
  - KB builder pending-action confirmation / diff / cancel behavior
  - default registration tool-id consistency

Remains:
- Docs, tests, and cutoff cleanup.

## Part 7 - Docs, Tests, and Legacy Cutoff

Removed:
- Migration-era wording that still described runtime-selected skill instructions as a temporary legacy compatibility adapter.
- Final live references that implied old waiting or prompt-only routing models were still supported runtime behavior.

Added:
- A rewritten [ARCHITECTURE.md](/Users/kayngame/jade_ai_core/ARCHITECTURE.md) that documents:
  - user turn flow
  - routing order
  - pending actions and execution contracts
  - skill invocation contracts
  - tool invocation/result envelopes
  - agent boundaries
  - explicit legacy cutoff
- README runtime-model updates in [README.md](/Users/kayngame/jade_ai_core/README.md).
- Additional regression coverage for:
  - ambiguous knowledge follow-ups blocking execution deterministically
  - ambiguous project-task follow-ups blocking execution deterministically
  - ambiguous KB-builder confirmations blocking execution deterministically
  - finalized skill prompt wording in `tests/test_skills.py`
- Explicit legacy cutoff notes stating that `pending_interaction`, regex-only approval paths, and prompt-only skill routing are no longer supported live runtime systems.

Remains:
- No remaining refactor parts in this execution plan.
