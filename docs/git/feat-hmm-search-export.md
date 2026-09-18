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

- **2026-09-18 — decided: no.** The search is not exported at 0.3.0; `__all__` stays at
  the ten names it has carried since `viterbi_batch`. Recorded as
  [ADR 0025](../design/adr/0025-topology-search-not-exported.md), with a `DEFERRED.md`
  trigger keyed on PRD §8 settling or `hmm` 0.4.0 so the question is re-made rather than
  inherited. The decision rests on reversibility, not on a judgement that the current
  surface is wrong: `__all__` may widen later at no cost to any consumer and may never
  narrow, so 0.3.0 is the wrong version to make a promise it cannot withdraw.
- **What settled it beyond the asymmetry**: exporting the search is five names, not one,
  since the whole frozen dataclass graph is reachable from `SearchResult`; and one of
  those five, `TrialResult`, carries `d` and `data_bits`, which assert the two-part
  description length that `_total_description_length` returns a bare `float` precisely to
  avoid asserting (PRD §8). Publishing it would have contradicted a decision written two
  modules away.
- **Scope shrank on contact with the code.** The branch opened saying two documents lean
  toward exporting; one had already been turned by PR #50, which gave
  `docs/api/hmm/baum_welch.md` the paragraph stating the search is private. So the "if it
  does not" arm is one phrase in the master plan's audit subgoal, not two documents.
- **Strongest alternative, rejected on timing rather than merit**: a narrowed public
  result carrying `params`, `total_bits`, `cycles` and `converged` while withholding `d`
  and `data_bits`. It is written into ADR 0025's Open section and the trigger, so whoever
  reopens this finds the argument instead of rebuilding it.
- **The drift found was the mirror image of the drift expected.** The plan said to close
  documents "left asserting a surface that does not exist". One of the two was already
  closed by PR #50. What the code turned up instead was the opposite: the
  *public-surface* paragraph of `docs/api/hmm/README.md` enumerated six private modules
  and omitted all four revision 04 added — `_search`, `_trials`, `_topology` and `_mdl` —
  so the page that defines what is private was silent about the four things this branch
  decided are private. It now lists all ten and says why the last four are, citing
  ADR 0025.
- **Why that sentence could drift at all**, which generalises to every ADR 0013 page:
  `tests/test_api_docs.py` executes the code blocks against pasted output, so the ten-name
  `__all__` listing beneath that paragraph cannot go stale — but the prose around it is
  unexecutable and is checked by nobody. The blocks are verified; the sentences are not.
