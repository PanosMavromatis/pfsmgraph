# chore/revise-plugins

**Status**: merged — PR #19 — 2026-09-09
**Created**: 2026-09-07
**Subgoal**: standalone — meta-work interpolated before revision 02 resumes

## This is a pointer, not the plan

**The plan for this branch is [`tmp/TODO.md`](../../../tmp/TODO.md)**, at the repository
root. Work it with:

```
/hitl-step N tmp/TODO.md
```

**Always pass the path.** `tmp/TODO.md` sits outside `docs/plan/`, so no resolution rung
finds it on its own — and rung 2 finds *this* file, which is why this file exists and says
so rather than being omitted. Declining to create a branch plan at all would not have
avoided the problem: resolution would then fall through to `docs/plan/TODO.md`, the
pfsmgraph master plan, which is also the wrong file and fails less visibly.

It is a pointer rather than a copy because the work is committed in two *other*
repositories — the `dp-compile` (formerly `tokalign-dev`) and `workflow-claude` clones
parked under the gitignored `tmp/`. A second real plan here would drift from the first the
moment work started, while describing commits this repository does not contain.

## Tasks

- [x] Work this branch from `tmp/TODO.md`. Flip this at merge; it is deliberately the only
      item, and its `> **Done:**` line is where the outcome gets summarised for the PR body.
  > **Done:** `tmp/TODO.md` closed at **33 of 33** across sections A–G, with no `[~]` or
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
