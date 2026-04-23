# frontdoor

`frontdoor` is the shared entrypoint for the team.

It discovers all merged agent packages under `agents/`, loads them as ADK sub-agents, and lets ADK handle the handoff. Most teammate PRs should not need to touch this folder.
