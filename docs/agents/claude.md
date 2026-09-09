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

**This repository's commit convention, stated rather than corrected.** Every commit is a
capitalised imperative subject with no prefix, followed by a substantial body explaining
why. Step 3 of `/smart-commit` used to prescribe conventional-commit subjects
(`feat(scope): …`) and this paragraph existed to override it; since `workflow-claude`
PR #21 the command reads the convention from `git log` instead of asserting one, so the
override is spent. The statement stays, because it is now what the command goes looking
for — and a convention worth following is worth writing down rather than leaving to be
inferred from a sample.

**Both ways that detection can be fooled are absent here, which is why it can be trusted.**
Measured 2026-09-09 over the last 40 non-merge commits: 40 plain imperative, 0
conventional. Three match the plugin's own bookkeeping templates (`Add the branch doc
for …`, `Record the merge of …`, `Remove the branch doc …`), which the rule discounts — and
discounting them changes nothing, because those templates are *also* plain imperative here.
**A template pollutes a history only when its form disagrees with the repository's**; in
`workflow-claude`'s own repository it disagreed, and 19 template-written subjects sat
against 21 human ones. The second failure mode — a subtree import grafting another
repository's convention into the log — cannot arise here either, since every `.scratch/`
import had its `.git` renamed on arrival. See [`../plan/DEFERRED.md`](../plan/DEFERRED.md)
under "the next `workflow-claude` revision" for where that one is live.

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

## `dp-compile` and the root `dp-compile.toml`

`dp-compile` is a Claude Code plugin that guides a dynamic-programming kernel through the
ADR 0016 lifecycle. **It counts five stages where [`core.md`](core.md) counts four phases,
and the two agree** — the plugin puts a `formalization` stage ahead of the four
implementation phases, so its chain reads `formalization → python → cython → cpu_parallel →
cuda`. `core.md` is authoritative for the invariant; the extra stage is the plugin's, and
it is a specification document rather than a backend. The
root **`dp-compile.toml` is its manifest**, and it is the only thing in this repository the
plugin reads to learn the layout: kernel paths per phase, build and test commands, where
backends are registered, and which documents a phase skill must read before writing.
**The plugin has no defaults.** A repository without the file is told so, rather than
resolving paths under a layout it does not have and reporting "algorithm not found" — a
missing file misdiagnosed as a missing algorithm.

Two properties of the manifest matter when editing it:

- **Algorithms are listed, never discovered.** `[algorithms.viterbi]` exists because this
  family's packages are flat, so globbing `_*.py` under `hmm` would return `_numeric.py`
  and `_params.py` beside `_viterbi.py`. Adding a kernel means adding a table.
- **`[commands] benchmark` is deliberately absent.** There is no benchmark infrastructure
  here, and the benchmarking skill must report an unconfigured command rather than invent
  one.

**The plugin ships a blocking `PreToolUse` hook on `Bash`, and the manifest is what arms
it.** Before a `git commit` whose *staged set* touches a kernel path, it runs `[commands]
build` and `test` and blocks on failure. Scoping is what makes blocking tolerable: with
this manifest it arms on exactly four paths — `_viterbi.py` and its three unwritten
siblings — so a commit touching `docs/`, a helper module such as `_numeric.py`, or even a
phase-0 `FORMALIZATION.md` passes silently. Phase 0 is excluded by design: a Markdown
specification compiles to nothing and is imported by nothing, so staging it cannot break a
backend.

**The gate fires inside `/smart-commit`, and that is the design rather than a leak.**
`/smart-commit` ultimately runs `git commit` through `Bash`, so the hook sees it like any
other commit. This is what makes `dp-compile` safe to delegate committing to
`workflow-claude`: the enforcement is deterministic and sits below whichever command
happens to be driving. The two plugins' `PreToolUse` hooks match **disjoint** tool sets —
`dp-compile` on `Bash`, `workflow-claude`'s `protect-agent-docs.py` on
`Write|Edit|MultiEdit` — so no tool call matches both and they stack rather than compete.

`dp-compile` **assumes `workflow-claude` is present** and delegates rather than duplicating:
it runs no `git` command of its own, opens no branch, and writes no plan file. Its commands
detect the companion by reading their own tool list, print one line when it is missing, and
stop at the point where they would have delegated.

**Both plugins are installed from a marketplace, and the declaration is tracked.**
`.claude/settings.json` registers `mavromatis-ai-labs`
(`github.com/PanosMavromatis/claude-plugins`, a monorepo holding both plugins under
`plugins/`) and enables `workflow-claude` and `dp-compile` from it, so a fresh clone
reproduces the setup with no manual step. It is the first tracked file under `.claude/`;
the `tmp/` clones and the `.claude/skills/workflow-claude` symlink that preceded it are
retired. **`dp-compile.toml` is still tracked and the plugin that reads it still is not** —
that half did not change, and it is the point: the manifest belongs to this repository, the
plugin does not.

**`--plugin-dir` is now the development path, not the loading path.** It loads a plugin
directory for a single session with no marketplace involved, which is what to use when
editing the plugins themselves; pointed at the monorepo's `plugins/` directory it loads
both at once, which is what changing their interaction needs. Do **not** register a local
checkout as a marketplace to test it: registration is keyed on the `name` inside
`marketplace.json`, so adding the local copy under the same name silently replaces the
GitHub one for every project on this machine.

Everything else — architecture, commands, conventions, domain invariants — belongs in
`core.md`, which also feeds `AGENTS.md` for other agents.
