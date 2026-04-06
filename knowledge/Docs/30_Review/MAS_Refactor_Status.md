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
