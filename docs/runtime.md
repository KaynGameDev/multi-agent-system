# Runtime Model

## Registry

The platform scans immediate child directories under `agents/`. Every package directory must include `manifest.yaml`.

## Loader

The loader resolves a manifest `entrypoint`, imports the referenced callable, executes it, and verifies the return value is a `google.adk.apps.App`.

Published packages can additionally expose module-level `app` or `root_agent` from `agent.py` so the same package can be discovered directly by `adk web`.

## Front door

`frontdoor` is the shared entrypoint app. It discovers all merged packages except itself, loads their ADK apps, collects each app's `root_agent`, and registers those as ADK `sub_agents`.

That means the routing model for v1 is:

- Git merge acts as publication
- `frontdoor` performs discovery
- ADK performs delegation and handoff between merged agents

## Runner

`mas run <agent-id>` uses:

- `InMemorySessionService`
- `Runner`
- one ephemeral session per invocation

The command sends one user message to the loaded ADK app and prints emitted text events to stdout.

If you want to use ADK Web directly, point it at `agents/` and choose `frontdoor` as the main app.

## Validation

Validation currently enforces:

- manifest schema correctness
- duplicate package IDs
- folder-name and manifest-ID alignment
- entrypoint importability
- entrypoint return type
- package test presence
- detectable undeclared env-secret usage

## Extensibility

The v1 platform does not abstract over multiple runtimes. ADK is the only supported runtime until a later version intentionally introduces another adapter.

Teammates should prefer placing reusable services, tools, schemas, and MCP adapters under `shared/` so individual agent folders remain self-contained and low-conflict.
