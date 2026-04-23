# Contributing Agents

## Publishing flow

An agent is considered published when its package is merged into `main` after review and green CI.

## Package layout

```text
agents/<agent_id>/
  manifest.yaml
  __init__.py
  agent.py
  prompts.py
  tools.py
  README.md
  tests/
```

## Required rules

- `agent_id` must be a valid Python identifier.
- The folder name must match the manifest `id`.
- `runtime` must be `adk`.
- The manifest `entrypoint` must be `module:function`.
- The entrypoint function must return an ADK `App`.
- `agent.py` should also expose module-level `app` or `root_agent` for `adk web` compatibility.
- Each package must have tests.
- If code reads environment variables with `os.getenv(...)`, `os.environ.get(...)`, or `os.environ[...]`, those variable names must be declared in `required_secrets`.
- Do not import another agent's private files. Shared dependencies belong in `shared/`.

## Recommended workflow

1. Run `mas scaffold agent <agent_id>` or `mas scaffold group <agent_id>`.
2. Keep agent-specific prompts, tools, and implementation inside that folder.
3. Implement the ADK app in `agent.py`.
4. Import reusable helpers only from `shared/`.
5. Let `frontdoor` auto-discover your manifest after merge.
6. Add package tests under `tests/`.
7. Run `mas validate <agent_id>`.
8. Run `mas test <agent_id>`.
9. Open a PR and merge after review.

## Local environment

The `mas` CLI loads `.env` from the repo root automatically before validation, test, and run commands. Keep real secrets only in `.env`, and commit placeholders through `.env.example`.
