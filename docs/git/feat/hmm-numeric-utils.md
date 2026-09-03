# feat/hmm-numeric-utils

**Created**: 2026-09-03
**Base**: main at 211723a
**Status**: active

## Purpose

Migrate the numeric Utility code that `pfsmgraph-hmm` 0.1.0 needs out of the Lush
original's `Code/Utility/util.lsh`, private to the package. This is the third subgoal of
revision `02-hmm-v0.1.0` and the first that writes package code rather than settling a
decision: `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/` currently holds nothing but an
empty `__init__.py`.

Six functions migrate — `safe-/`, `int-delta`, `safe-add--log2`, `safe->--log`,
`calculate-entropy`, `rand-p-vector` — plus the stationary-distribution solve. The solve
is the one that cannot be ported by transcription: `(Pᵀ - I)π = 0` is singular by
construction, so a port that hands the homogeneous system as stated to a dense solver
fails outright. The row-replacement trick has to be reproduced, not just its result.

## Scope

- `_numeric.py` and whatever else the module-layout decision names
- The log₂-domain accumulator and comparator, and the fate of the `-1` log-zero sentinel
- The stationary solve, on `numpy.linalg.solve` rather than a translated LU
- `rand-p-vector` and `calculate-entropy`
- `packages/pfsmgraph-hmm/tests/` — the package's first tests
- Negative findings: what was replaced by a library call, and what migrates nowhere
- A widening of `.scratch/hmm-lush/.gitignore` — the generated `util.c` as a Lush-semantics
  reference, and three saved `.hmm` model directories as differential-test fixtures

## Context

- Master plan subgoal: `docs/plan/TODO.md`, revision `02-hmm-v0.1.0`
- `.scratch/hmm-lush/HMMLIB-ACCOUNT.md` §3 (the `-1` sentinel) and §4 (the solve)
- `.scratch/hmm-lush/Code/Utility/util.lsh` — 574 lines; the migrating functions sit at
  44, 246, 349, 389, 417, 431, 439, 448, 523
- [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md) — `state_p` and the
  entropies are cached properties of a frozen parameter value, so the solve and
  `calculate-entropy` are what those properties will call
- [ADR 0015](../../design/adr/0015-arc-emission-mealy-formulation.md) — arc-emission, so
  the transition matrix these operate on is `(S, S)` and the emission is `(S, S, A)`

## Notes

<!-- Running log: decisions made, things tried, things deferred. -->

**2026-09-03 — goal 1.** Four functions get written, not six: `int-delta` dissolves into
`np.eye` and `safe->--log` into `>`, once the `-1` log-zero sentinel is replaced by `+inf`.
That replacement is the branch's first decision, taken because faithfulness to `-1` is
*uncheckable* — there is no Lush runtime here, and the sentinel reaches no persisted
artifact.

The unlooked-for result is that the stationary solve's test got much stronger than planned.
A saved `.hmm` directory holds `transition_p` beside `state_p` and `output_p` beside
`state_entropies`, so it is an input/output pair for both computations this branch ports,
in a plain-ASCII format needing no model code to read. Measured across sizes 1, 5 and 8:
`A = Pᵀ - I; A[0,:] = 1; b = e₀` reproduces every saved `state_p` to 5e-5, the print
format's own rounding. The planned closed-form two-state chain is superseded.

This required amending a recorded decision. `.scratch/hmm-lush/.gitignore` had declined
saved checkpoints as "outputs of the algorithm being translated, not inputs to it, and
unreadable without the model code that wrote them"; both halves are false for these three,
and the declining note now points at its own exceptions.

**2026-09-03 — goal 2.** `_numeric.py` and `tests/test_numeric.py` land: the first code and
the first tests in `packages/pfsmgraph-hmm/`. Suite 94 → 124.

Two functions written, two dissolved. `bits(p)` is unary where `safe-add--log2` was binary,
because the accumulator argument existed only to check the `-1` sentinel and `+inf` absorbs
unaided; it survives as a function at all only because it is the one place numpy's
`log2(0)` warning is suppressed. `safe_divide` is array-aware where the original was scalar
in a loop. `safe->--log` dissolves into `>` and `int-delta` into `np.eye`, both with the
original's behaviour kept as tests so a reinstated sentinel fails rather than passes.

Two things for the reviewer. `safe_divide` has **no consumer in 0.1.0** — all fifteen call
sites are revision 03 or 04 — so it is migrated on the strength of the subgoal naming it,
not on need. And this subgoal's own text offered "replaced by `-inf`"; it is `+inf`, since a
description length is `-log2(p)`. That sign slip is §3's orientation trap in miniature and
is recorded in the plan rather than quietly corrected.

**2026-09-03 — goal 3.** `stationary_distribution` lands; suite 124 → 140. The solve was
already validated in goal 1, so the work was the two things around it.

A **reducible** chain leaves `A` singular even after the row replacement — the trick trades
one redundant equation for the normalization, which rescues nullity 1 and no more.
Measured: a four-state chain with two closed classes is still rank 3 of 4. Revision 04
searches topology by state merge and split, so this is a plausible search *outcome*, not a
malformed input; it raises `ValueError` naming the cause, `from` the `LinAlgError`.

The saved fixtures are **four-decimal prints**, and that broke two tests in unrelated ways:
a tolerance of `5e-5` that a value landed exactly on (true `π₀` = 0.10135, printed
"0.1014"), and a singularity assertion defeated by rows summing to `1 ± 1e-4`, which lifts
the smallest singular value nine orders above `matrix_rank`'s tolerance. Fixed by widening
to `1e-4` with both error sources named, and by renormalising rows before asserting
singularity — restoring the hypothesis rather than loosening the conclusion. Revision 03's
differential tests inherit this hazard.

The fixtures are read from `.scratch/` **in place**. They are tracked, so they exist in
every clone, and `tests/` never ships — the wheel packages only `src/pfsmgraph`. The `.MAT`
reader is a helper in the test module rather than a `conftest.py`, on the same
"one consumer, no invented structure" grounds that kept `_numeric.py` one module.
