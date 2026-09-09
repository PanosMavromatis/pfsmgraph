# chore/revise-plugins

**Status**: merged — PR #19 — 2026-09-09
**Created**: 2026-09-07
**Subgoal**: Meta-work interpolated before revision 02 resumes — registered in
[`../TODO.md`](../TODO.md) under revision 02-hmm-v0.1.0, backlinked 2026-09-09.

## The record is [`_TODO.md`](_TODO.md), beside this file

This file is the resolvable branch plan: small, carrying the merge stamp, and what
`/hitl-step` rung 2 and `/smart-merge` find by name. **The plan itself — every goal, every
inline Q&A, every measurement — is `_TODO.md` in this directory.** Read it by path; the `_`
prefix deliberately keeps it out of the `docs/plan/**/TODO.md` glob, so a finished plan
cannot surface in a disambiguation prompt as a place to do work. That is
`/close-revision`'s `_DO.md` convention applied to a branch plan.

**It lived at `tmp/TODO.md` until 2026-09-09**, at the repository root, because the work
was committed in two *other* repositories — the `dp-compile` (formerly `tokalign-dev`) and
`workflow-claude` clones parked under the gitignored `tmp/` — and the plan sat beside them.
The root `.gitignore` carried a `!tmp/TODO.md` negation to keep that one file tracked; the
negation was retired with the move.

**Why this file was a pointer rather than a copy, and why that reasoning still matters.**
A second real plan here would have drifted from the first the moment work started, while
describing commits this repository does not contain. But the arrangement had a cost that
only appeared at the end: `tmp/TODO.md` was outside `docs/plan/`, so no resolution rung
found it and every invocation had to pass the path — and, more quietly, `/file-plans`
derives a plan's revision from its master-plan backlink, which this branch did not have, so
the directory would have sat flat beside the master plan forever with its content in a
third place. Moving the body in and adding the backlink resolved both. **The lesson is not
"never point" — it is that a plan parked outside the tree needs an explicit plan for
getting back in, decided when it is parked rather than at merge.**

## Tasks

- [x] Work this branch from `tmp/TODO.md` (now [`_TODO.md`](_TODO.md) beside this file).
      Flip this at merge; it is deliberately the only item, and its `> **Done:**` line is
      where the outcome gets summarised for the PR body.
  > **Done:** the plan closed at **33 of 33** across sections A–G, with no `[~]` or
  > `[!]` anywhere. `tokalign-dev` was renamed and generalised to `dp-compile`, its
  > hardcoded layout replaced by a per-repository manifest with no defaults; ADR 0016's
  > five-stage chain implemented end to end including a real phase-4 skill where a ten-line
  > stub stood; a legacy-code entrance added and dry-run against `_viterbi.py`; and the two
  > plugins verified to interoperate, both severities exercised live rather than reasoned
  > about. Landed here: `dp-compile.toml`, the `docs/agents/claude.md` section reconciling
  > the plugin's five stages with `core.md`'s four phases, and this record.
  >
  > **Two things the work found that it did not set out to find.** Auditing
  > `workflow-claude` against its own convention exposed nine sites hardcoding one
  > repository's commit and PR-title form, wrong in opposite directions in both consumers;
  > fixed there as PR #21, where the detection rule proved *self-contaminating* — 19 of the
  > repository's 40 recent subjects had been written by the very templates being removed.
  > And a disagreement between two verification runs traced to the fixture rather than the
  > plugin: `Glob` does not follow symlinks, so a mirrored tree is not the tree it mirrors.
  >
  > Seven pfsmgraph edits collected along the way were cut to `chore-plugin-handback` and
  > registered in the master plan, rather than landed piecemeal here — PR #19.
