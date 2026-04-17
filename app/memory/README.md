# Memory Subsystem

This folder is scaffolding for Jade's future memory subsystem. It adds names, paths, and contracts only. It does not change runtime behavior yet.

## Folder Layout

- `paths.py`: resolves the future memory work directory and the default paths for session memory, long-term memory, retrieval artifacts, and compaction artifacts.
- `types.py`: shared contracts for session memory snapshots, long-term memory records, retrieval queries/results, and compaction requests/summaries.
- `interfaces.py`: backend protocols for the future session-memory store, long-term-memory store, retrieval layer, and compactor.
- `__init__.py`: convenience exports for the package surface.

## Runtime Concepts

### Session memory

Session memory is short-lived conversation memory keyed to a thread. In Jade, it represents durable continuation context that helps after transcript compaction, but it is still scoped to the current conversation rather than being a permanent knowledge store.

### Long-term memory

Long-term memory is durable memory that survives beyond a single thread, checkpoint, or compacted transcript. It is the right home for promoted facts, decisions, and reusable project context that Jade should be able to retrieve later.

### Retrieval

Retrieval is the lookup layer that searches long-term memory and returns the small set of records relevant to the current turn. Retrieval is separate from storage so the runtime can change backing stores later without changing the turn-time contract.

### Compaction

Compaction reduces a long active transcript into a continuation summary plus a small preserved tail. Compaction is not long-term memory by itself; it is a context-window management step that may reuse session memory when that summary is still valid.

## Default Path Layout

By default, `MEMORY_WORK_DIR` resolves to `runtime/memory/` and future memory assets are expected to live under:

```text
runtime/memory/
  session_memory.json
  long_term/
  retrieval/
  compaction/
```

## Current Repo Mapping

The live runtime still uses the existing modules:

- `app/session_memory.py` for current session-memory behavior
- `app/compaction.py` for transcript compaction
- `app/rehydration.py` for restoring runtime state after compaction or reload
- `app/checkpoints.py` for durable LangGraph checkpoint storage

The new `app/memory/` package is the stable place for future consolidation work.
