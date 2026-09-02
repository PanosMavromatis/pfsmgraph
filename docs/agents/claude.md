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
  under any configuration. Those tags come from `just release` at the repo root instead
  (see [`core.md`](core.md) under "Commands"), which is why `/smart-commit` reporting "no
  tag created" on a release commit is the correct outcome rather than a gap to fill.

Its Step 3 also specifies conventional-commit subjects (`feat(scope): …`). This repository
does not use them: every commit is an imperative subject with no prefix and a substantial
body explaining why. Follow the repository's convention — the history is the authority, not
the command's template.

## A nested `CLAUDE.md` in imported source

Claude Code loads a `CLAUDE.md` found beside files it reads, so importing another project's
working tree into this repository can silently pull that project's agent instructions into a
pfsmgraph session. It happened on the first import into `.scratch/`: the copied tree carried
its own `CLAUDE.md`, which was renamed to `CLAUDE.md.orig` on arrival.

**Presence on disk is the trigger, not tracking** — gitignoring the file does not help, so the
fix has to be a rename. Do the same for any future import, and check for one before reading
anything in a newly imported tree rather than after. The same applies to a nested `AGENTS.md`
for other agents.

Note that the `protect-agent-docs.py` `PreToolUse` hook matches on filename, so it will block
edits to an imported `CLAUDE.md` too — correctly, but for an unrelated reason. Renaming also
takes the file out of that hook's way.

Everything else — architecture, commands, conventions, domain invariants — belongs in
`core.md`, which also feeds `AGENTS.md` for other agents.
