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
