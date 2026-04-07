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
- Shared skill runtime adapter in `app/skill_runtime.py` for active-skill filtering, runtime diagnostics, and the temporary legacy `SKILL.md` prompt-compatibility layer.
- Richer `SkillInvocationContract` metadata in `app/contracts.py` and `app/skills.py`, including skill name, description, source path, explicit invocation source, and explicit invocation reason.
- Gateway-emitted active skill runtime state in `gateway/agent.py`, including `active_skill_invocation_contracts` and `skill_execution_diagnostics`.
- Prompt builders that consume runtime-selected skill contracts instead of `resolved_skill_ids`.
- Regression coverage proving prompts now read runtime contracts and that skill execution state survives routing and tool loops.

Remains:
- Tool-calling standardization.
- Gateway routing cleanup.
- Agent migration and deletion.
- Docs, tests, and cutoff cleanup.
