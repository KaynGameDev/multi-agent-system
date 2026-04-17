# Memory Subsystem

This folder contains the shared memory subsystem surface for Jade, including the file-based long-term memory format and persistent store helpers.

## Folder Layout

- `paths.py`: resolves the future memory work directory and the default paths for session memory, long-term memory, retrieval artifacts, and compaction artifacts.
- `types.py`: shared contracts for session memory snapshots, long-term memory records, retrieval queries/results, and compaction requests/summaries.
- `interfaces.py`: backend protocols for the future session-memory store, long-term-memory store, retrieval layer, and compactor.
- `long_term.py`: reads, validates, and mutates file-based long-term memory on disk.
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
    MEMORY.md
    topics/
      project_overview.md
  retrieval/
  compaction/
```

## Long-Term Memory File Format

The long-term memory store expects a required root index file plus Markdown topic files:

- `long_term/MEMORY.md`: required index file
- `long_term/topics/**/*.md`: topic files managed by the store

Each file must start with frontmatter containing:

```yaml
---
name: Project Overview
description: Durable context for the current roadmap and release work.
type: project
---
```

Supported `type` values:

- `user`
- `feedback`
- `project`
- `reference`

The `MEMORY.md` body is intentionally short. It acts as a catalog and points to topic files instead of storing the full memory text inline. Entries are rendered like:

```md
## Topics

- [Project Overview](topics/project_overview.md) (`project`): Shared roadmap and release context.
```

The loader/store validates that:

- a `MEMORY` index file exists at the root of the long-term memory directory
- each memory file has frontmatter
- `name`, `description`, and `type` are present and non-empty
- `type` is one of the supported values above
- index entries point to files under `topics/`
- index metadata matches the referenced topic file metadata

## Persistent Store Helpers

`long_term.py` now includes file-store helpers for:

- listing memories from the `MEMORY.md` index
- loading a single memory topic file by id
- creating or updating a memory topic file and rewriting the index
- deleting a memory topic file and removing its index entry

The index stays compact, while full memory content lives only in the topic files.

## Agent-Scoped Memory

Agent definitions can opt into a memory scope:

- `user`: resolves under `long_term/agents/<agent>/users/<user-key>/`
- `project`: resolves under `long_term/agents/<agent>/projects/<project-key>/`
- `local`: resolves under `long_term/agents/<agent>/local/`

Scoped agent memory is exposed through dedicated memory tools rather than raw path input. The tool layer resolves the scope-specific directory from runtime state and only reads or writes inside that directory.

## Current Repo Mapping

The live runtime still uses the existing modules:

- `app/session_memory.py` for current session-memory behavior
- `app/compaction.py` for transcript compaction
- `app/rehydration.py` for restoring runtime state after compaction or reload
- `app/checkpoints.py` for durable LangGraph checkpoint storage

The new `app/memory/` package is the stable place for future consolidation work.
