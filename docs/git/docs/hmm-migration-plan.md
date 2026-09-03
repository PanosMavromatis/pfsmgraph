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
anything, and no transliteration.

This branch therefore surveys the library structurally, splits the migration into three
revisions, settles the decisions that split forces, and opens the first of them. **No code
lands under `packages/pfsmgraph-hmm/src/`,** and the source *reading* is not here either:
`ACCOUNT.md` records itself as a subgoal of `feat/dataseq-merge`, so `HMMLIB-ACCOUNT.md`
follows that precedent and belongs to revision 02.

**The migration is three releases, not one.** The trainer spans three problems that fail
differently — a decode, a fixed-topology estimator, and a search over model shapes — and
each raises its own questions about parallelism and data structures. Conflating them would
put the project's first `.pyx`, its first EM loop and its first resizing search in one
revision, where a failure in any of them would be diagnosed against all three.

## Scope

- Draft `docs/plan/planned/0{2,3,4}-hmm-*.md` and register them under `## Planned revisions` in the master plan
- Settle the numpy/`torch` posture, the home of the migrated `Utility` code, and whether the packaging work is isolated
- Write the ADRs those decisions warrant, including that ADR 0002's anti-diagonal claim does not generalize to HMM
- Add the `DEFERRED.md` triggers, and open revision 02

## Context

- `docs/plan/TODO.md` — revision 01 closed; no revision is open, so this branch is standalone and opening revision 02 is one of its deliverables
- `docs/plan/01-dataseq-v0.1.0/_TODO.md` — how the previous revision was shaped; the model this one should follow
- `.scratch/hmm-lush/ACCOUNT.md` — the template for goal 1, and the evidence that goal 1 is owed: it is scoped to `SeqData` throughout
- `.scratch/hmm-lush/Code/HMMlib/` — `hmm.lsh` (319), `hmm-param.lsh` (386), `hmm-trainer.lsh` (1102), `hmm-trainer-view.lsh` (237)
- `.scratch/hmm-lush/Code/Utility/` — `util.lsh` (574) and `mc.lsh` (47); mostly Numerical Recipes transcriptions with zero call sites from `HMMlib`, but it also holds the MDL machinery that scores topology search
- [ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md) — the three-phase lifecycle; Baum-Welch is where it first applies
- [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) — the backend matrix, empty until a DP kernel reaches phase 1
- [ADR 0012](../../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md) — why `hmm` is on hatchling, and what the first `.pyx` reopens
- [ADR 0014](../../design/adr/0014-scratch-retention-and-per-package-scoping.md) — `.scratch/hmm-lush/`'s `.gitignore` phase went active 2026-09-01 and is already scoped to the whole live library
- `docs/plan/DEFERRED.md` — `## Trigger: the first .pyx` is the section goal 2 must decide whether to fire

## Notes
