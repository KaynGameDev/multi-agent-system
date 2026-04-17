# Memory Subsystem

This folder contains the shared memory subsystem surface for Jade, including file-based session memory, file-based long-term memory, and persistent store helpers.

## Folder Layout

- `paths.py`: resolves the future memory work directory and the default paths for session memory, long-term memory, retrieval artifacts, and compaction artifacts.
- `types.py`: shared contracts for session memory snapshots, long-term memory records, retrieval queries/results, and compaction requests/summaries.
- `interfaces.py`: backend protocols for the future session-memory store, long-term-memory store, retrieval layer, and compactor.
- `session_files.py`: creates, reads, and updates per-conversation Markdown session files with a fixed template.
- `long_term.py`: reads, validates, and mutates file-based long-term memory on disk.
- `retrieval.py`: header-first retrieval helpers that rank index entries before opening a few topic files.
- `extraction.py`: conservative post-turn durable-memory extraction helpers that promote a few stable facts into long-term memory.
- `__init__.py`: convenience exports for the package surface.

## Runtime Concepts

### Session memory

Session memory is short-lived conversation memory keyed to a thread. In Jade, it represents durable continuation context that helps after transcript compaction, but it is still scoped to the current conversation rather than being a permanent knowledge store.

The current file-backed session template lives under `sessions/` and gives each conversation a dedicated Markdown summary file with these fixed sections:

- current state
- task spec
- key files
- workflow
- errors/corrections
- learnings
- worklog

### Long-term memory

Long-term memory is durable memory that survives beyond a single thread, checkpoint, or compacted transcript. It is the right home for promoted facts, decisions, and reusable project context that Jade should be able to retrieve later.

### Retrieval

Retrieval is the lookup layer that searches long-term memory and returns the small set of records relevant to the current turn. Retrieval is separate from storage so the runtime can change backing stores later without changing the turn-time contract.

The current file-based retrieval pass is intentionally header-first:

- it scans `MEMORY.md` entries first
- ranks candidates from `name`, `description`, `type`, id, and path metadata
- skips memories already present in `context_paths` or `recent_file_reads`
- opens only the top few topic files after ranking

### Durable extraction

Durable extraction is the post-turn promotion step that looks at a completed conversation turn and writes only a few stable facts into long-term memory. It is intentionally more conservative than session memory or compaction.

The current extractor is heuristic-first and prefers explicit durable user signals such as:

- preferred name
- stable response preferences
- workflow/output-format preferences
- durable behavior corrections
- explicit project context statements

It intentionally skips ordinary ephemeral asks like "what is due today?" and it also skips automatic extraction for a turn if the agent already called `memory.write` directly during that turn.

### Compaction

Compaction reduces a long active transcript into a continuation summary plus a small preserved tail. Compaction is not long-term memory by itself; it is a context-window management step that may reuse session memory when that summary is still valid.

## Default Path Layout

By default, `MEMORY_WORK_DIR` resolves to `runtime/memory/` and future memory assets are expected to live under:

```text
runtime/memory/
  session_memory.json
  sessions/
    web/
      conversation-id.md
  long_term/
    MEMORY.md
    topics/
      project_overview.md
  retrieval/
  compaction/
```

`session_memory.json` remains the compact machine-oriented store used by the live runtime today. The per-session files under `sessions/` are the human-readable companion files for each conversation.

## Session Memory File Format

Each session file is a Markdown document with frontmatter plus a fixed section template. The frontmatter declares the conversation id and template metadata:

```yaml
---
thread_id: web:test-thread
kind: session_memory
template_version: 1
---
```

The body uses this fixed heading layout:

```md
# Session Memory

## Current State

## Task Spec

## Key Files

## Workflow

## Errors/Corrections

## Learnings

## Worklog
```

`session_files.py` validates that:

- the file declares `thread_id` in frontmatter
- `kind`, when present, is `session_memory`
- `template_version` matches the current template version
- all required sections are present
- `Key Files` uses markdown bullet items

The helpers can create a blank template, read a session file from disk, and update only selected sections while preserving the rest of the document.

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

When retrieval is enabled, scoped agents can also inject a compact `Relevant Memories` block into their prompt. That block is built from the agent's own scoped memory root only, so retrieval stays aligned with the same path-scoped permission boundary as read and write operations.

Scoped agents can also opt into automatic durable-memory promotion after a successful turn. That promotion writes into the same scope-resolved long-term memory root, so the automatic path follows the same directory boundary as explicit memory tool use.

## Current Repo Mapping

The live runtime still uses the existing modules:

- `app/session_memory.py` for current session-memory behavior
- `app/compaction.py` for transcript compaction
- `app/rehydration.py` for restoring runtime state after compaction or reload
- `app/checkpoints.py` for durable LangGraph checkpoint storage

The new `app/memory/` package is the stable place for future consolidation work.
