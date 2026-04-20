# Jade V2 Task List

## Summary
- Focus the V2 planning tracker on 4 top-level workstreams only: `Context management`, `Memory management`, `Vector database construction`, and `Automations`.
- Keep the document high level for now and break each workstream into smaller tasks later.
- This is a planning tracker only.

## Memory management
- Define the V2 memory model clearly across live graph state, conversation transcripts, compaction summaries, session memory, and durable checkpoints so each layer has a single responsibility.
- Harden the session-memory lifecycle so initialization, incremental updates, reinitialization, invalidation, and transcript-reset behavior stay predictable during long-running conversations.
- Make transcript compaction a first-class memory operation by reusing session memory when safe, falling back to fresh summaries when needed, and preserving only durable continuation context.
- Strengthen rehydration and resume flows so compacted conversations can recover the runtime state they still need after reloads, restarts, missing checkpoints, or checkpoint corruption.
- Add observability and tuning for token thresholds, memory hit rate, stale-memory fallbacks, compaction failures, and checkpoint recovery so memory behavior can be adjusted with evidence instead of guesswork.
- Set explicit boundaries between conversational memory, durable task state, and future vector retrieval so V2 does not mix short-term memory with the knowledge base or retrieval systems.

## Write Location
- Intended Markdown file path: [JADE_V2_TASK_LIST.md](/Users/kayngame/jade_ai_core/JADE_V2_TASK_LIST.md)
