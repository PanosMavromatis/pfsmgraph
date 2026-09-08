# chore/revise-plugins

**Status**: active
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

- [ ] Work this branch from `tmp/TODO.md`. Flip this at merge; it is deliberately the only
      item, and its `> **Done:**` line is where the outcome gets summarised for the PR body.
