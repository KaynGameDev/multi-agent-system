# Rule: Response Language

Apply this rule to all user-facing replies unless a role-specific `AGENT.md` gives a more specific instruction.

## Default Behavior

- Answer in the same language as the user's latest message.
- If the user explicitly asks for a different language, follow that request.

## Preserve Fixed Literals

- Do not translate filenames, paths, IDs, slugs, field names, or command words.
- Keep technical literals exact unless the role-specific prompt explicitly tells you otherwise.
