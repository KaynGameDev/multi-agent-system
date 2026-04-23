# MAS Platform

`mas-platform` is an engineer-first monorepo for publishing, validating, and running shared Google ADK agents and agent groups.

## What ships in v1

- A merge-to-main agent registry based on `agents/<agent_id>/manifest.yaml`
- A Python SDK and `mas` CLI
- A shared `frontdoor` ADK router that auto-discovers merged agents
- Public shared surfaces under `shared/`
- Validation for manifests, entrypoints, tests, and detectable environment-secret usage
- Two bundled packages:
  - `frontdoor`
  - `general_chat_agent`

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Then fill in the `.env` values for your local model provider before running agents.

List published packages:

```bash
mas list
```

Validate everything in the monorepo:

```bash
mas validate
```

Run the bundled front door:

```bash
mas run frontdoor --message "hello team"
```

Run only one package's tests:

```bash
mas test frontdoor
```

Scaffold a new package:

```bash
mas scaffold agent roadmap_agent --owner platform-team
mas scaffold group release_team --owner platform-team
```

## Package contract

Each published agent lives under `agents/<agent_id>/` and must include a `manifest.yaml` with:

- `id`
- `version`
- `kind`
- `runtime`
- `entrypoint`
- `owner`
- `description`

Optional fields:

- `tags`
- `required_secrets`
- `capabilities`
- `test_paths`

The `entrypoint` function must return a `google.adk.apps.App`.

For compatibility with ADK's built-in `adk web` UI, package authors should also export a module-level `app` or `root_agent` from `agent.py`.

## Team layout

```text
agents/
  frontdoor/
  general_chat_agent/
shared/
  services/
  tools/
  mcp/
  schemas/
```

- Teammates should usually work only inside `agents/<their_agent>/`.
- Shared dependencies belong in `shared/`.
- The `frontdoor` package auto-discovers merged agents and hands routing off to ADK.

## Notes

- v1 is ADK only.
- Publication is merge-to-main after CI passes.
- Agent IDs must be valid Python identifiers so package folders can also serve as import modules.

See [docs/contributing.md](/Users/kayngame/Multi-Agent System/docs/contributing.md) and [docs/runtime.md](/Users/kayngame/Multi-Agent System/docs/runtime.md) for the contributor and runtime details.
