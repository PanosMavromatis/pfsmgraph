# pfsmgraph — Claude Code context

Claude-Code-specific guidance. At migration time every section of the original
`CLAUDE.md` was tool-agnostic and moved to [`core.md`](core.md); what follows was
added afterwards.

Add content here — not to `core.md` — when it names a Claude Code feature by name:
skills under `.claude/skills/`, slash commands, subagent or `context: fork` patterns,
`/compact` guidance, `MEMORY.md` references, or `@import` conventions.

## `/smart-commit` in this repository

Two of its steps assume a shape this repository does not have. Neither is a bug in the
command; both are places where a single-component assumption meets a five-package family.

- **Never create a `VERSION` file at the repo root.** Its version-bump preflight collects
  every tracked `pyproject.toml` and rewrites the version of each one that differs, to
  match the root `VERSION`. Here that would silently set all five members to one version
  as a side effect of a commit about something else, destroying the per-package scheme
  described in [`core.md`](core.md) under "Versioning". The absence of the file is what
  keeps that step dormant, which makes it an invariant rather than a preference.
- **It will not tag releases here.** It only ever forms `v<VERSION>` from a root
  `VERSION`, so it cannot produce the `pfsmgraph-<pkg>-v<version>` tags this project uses,
  under any configuration. Per-package tags are created by hand in the release commit.

Its Step 3 also specifies conventional-commit subjects (`feat(scope): …`). This repository
does not use them: every commit is an imperative subject with no prefix and a substantial
body explaining why. Follow the repository's convention — the history is the authority, not
the command's template.

Everything else — architecture, commands, conventions, domain invariants — belongs in
`core.md`, which also feeds `AGENTS.md` for other agents.
