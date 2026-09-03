# docs/hmm-migration-plan

**Created**: 2026-09-03
**Base**: main at d1fe295
**Status**: active

## Purpose

Produce the plan for the `pfsmgraph-hmm` migration, and nothing else. `dataseq` is
released and revision `01-dataseq-v0.1.0` is closed, so PRD §11 puts `hmm` next — but the
Lush HMM library has never actually been read. The three reading aids under
`.scratch/hmm-lush/` (`ACCOUNT.md`, `COMPARISON.md`, `translation/`) were all written for
subgoals of `feat/dataseq-merge` and cover `Code/SeqData/` only, the container half.
`Code/HMMlib/` is 2,044 lines across four files and has no account, no comparison against
anything, and no transliteration. Planning the migration before reading it would be
inventing a schedule for code nobody here has looked at.

This branch therefore reads the library, settles the decisions the source forces, records
them where they will be found, and opens revision 02 with subgoals that a later branch can
execute. **No code lands under `packages/pfsmgraph-hmm/src/`.**

## Scope

- Read `Code/HMMlib/` in its own terms and write the account, as `ACCOUNT.md` did for `SeqData`
- Decide the translation strategy and the shape of `pfsmgraph.hmm`, including where it meets `dataseq`
- Settle the scope questions the source forces: the view layer, topology search, and the `.sds` format
- Record the outcome — new ADR(s), a `DEFERRED.md` trigger, and revision 02 opened in the master plan

## Context

- `docs/plan/TODO.md` — revision 01 closed; no revision is open, so this branch is standalone and opening revision 02 is one of its deliverables
- `docs/plan/01-dataseq-v0.1.0/_TODO.md` — how the previous revision was shaped; the model this one should follow
- `.scratch/hmm-lush/ACCOUNT.md` — the template for goal 1, and the evidence that goal 1 is owed: it is scoped to `SeqData` throughout
- `.scratch/hmm-lush/Code/HMMlib/` — `hmm.lsh` (319), `hmm-param.lsh` (386), `hmm-trainer.lsh` (1102), `hmm-trainer-view.lsh` (237)
- [ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md) — the three-phase lifecycle; Baum-Welch is where it first applies
- [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) — the backend matrix, empty until a DP kernel reaches phase 1
- [ADR 0012](../../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md) — why `hmm` is on hatchling, and what the first `.pyx` reopens
- [ADR 0014](../../design/adr/0014-scratch-retention-and-per-package-scoping.md) — `.scratch/hmm-lush/`'s `.gitignore` phase went active 2026-09-01 and is already scoped to the whole live library
- `docs/plan/DEFERRED.md` — `## Trigger: the first .pyx` is the section goal 2 must decide whether to fire

## Notes
