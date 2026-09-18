# feat/hmm-search-export

**Created**: 2026-09-17
**Base**: main at 73c451a
**Status**: active

## Purpose

Decide whether revision 04 makes the topology search part of `pfsmgraph.hmm`'s public
API, and record the decision either way. As planned, 0.3.0 ships what `core.md` and the
PRD both call "topology search by state merge and split" with `__all__` unchanged at the
ten names it has carried since `viterbi_batch`, and `_search`, `_trials`, `_topology`
and `_mdl` all private and out of contract. That may well be right — but it was never
asked, and two documents already lean the other way.

## Scope

- Assemble the case each way: what a public search would have to promise, against what
  ADR 0023's Open section and PRD §8 leave unresolved.
- Decide, and choose where the decision is recorded so the next revision finds it.
- If it exports: the names and signatures, `__all__`, `docs/api/hmm/search.md` with
  executed blocks per ADR 0013, `docs/api/hmm/README.md`'s public-surface table and
  `packages/pfsmgraph-hmm/README.md` (a PyPI long description under an immutable version).
- If it does not: the sentence that says so with its reason, and close the two documents
  that assume otherwise.
- Sync `docs/agents/core.md` and rebuild `AGENTS.md`.

## Context

- Master plan: `docs/plan/TODO.md:225`, under `## Subgoals — revision 04-hmm-v0.3.0`.
- Found on `feat/hmm-search-log` (PR #50) while asking where the search's training log
  should be documented; `docs/api/hmm/README.md:153` "The public surface" resolved it as
  *nothing*, which is what surfaced the question.
- The two documents leaning the other way: `docs/api/hmm/baum_welch.md:160`, whose
  pointer says the per-move log "belongs with topology search"; and the audit subgoal
  below it in the master plan, which says the API check covers "whatever the search loop
  and training log export".
- The cost of deciding to export: [ADR 0023](../design/adr/0023-topology-search-loop.md)
  Open leaves near-optimum ranking unresolved, and PRD §8 leaves the MDL criterion itself
  an open research question. The seam exists precisely so the criterion can be swapped.
- The cost of deciding not to: 0.3.0 ships a headline feature no consumer can call.
- Ordering: **before** the audit, which checks `docs/api/` against what ships, and
  **before** the release, since `__all__` is frozen by a published version.

## Notes
