# feat/hmm-viterbi-cpu-parallel

**Created**: 2026-09-09
**Base**: main at 2709f43
**Status**: active

## Purpose

Take the Viterbi decode to ADR 0002 phase 3 — Numba CPU-parallel, `@njit(parallel=True)`
with `prange` — the first point in this project where concurrent execution is actually
attempted. Phase 2 was a translation with no open question in it; this phase opens with one
the master plan deliberately held back. [ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md):53
states the anti-diagonal wavefront is "the same transformation for every DP kernel in the
family", and two independent passes have now concluded it does not hold here: the recurrence
is one-dimensional over time with dense `S × S` coupling between states, so it has no
anti-diagonals at all. Phases 1 and 2 are single-threaded and could therefore neither need
that answer nor falsify it; this branch is the first that can. It decides the question
against kernels that exist, records the decision where it is contract, and implements
whichever decomposition it names.

## Scope

- Settle the anti-diagonal question, and decide whether ADR 0002:53 is a **wording fix**
  scoped to the alignment-family kernels it was written from, or a **reversal** warranting
  its own ADR number. Numbers are never reused here, so the call is permanent either way.
- Record the outcome in [`FORMALIZATION.md`](../../design/algorithms/viterbi/FORMALIZATION.md)
  — both the Metadata row and the `Parallel decomposition` section currently read
  **Undetermined**.
- Write `_viterbi_cpu_parallel.py` under `@njit(parallel=True)`/`prange`, entering through
  `/dp-compile:next-phase viterbi`.
- Add the third `_backends.py` row and the differential tests that are this repository's
  only source of equivalence evidence until `align` brings the backend seam.
- Decide what `numba` becomes to `pfsmgraph-hmm`. It appears today only as
  `gpu = ["numba-cuda"]`; phase 3 needs plain `numba>=0.61` at runtime, in a member whose
  0.1.0 release is two subgoals away.

## Context

- Master plan: [`docs/plan/TODO.md`](../../plan/TODO.md), revision `02-hmm-v0.1.0` — the
  phase-3 subgoal and its two children. The anti-diagonal child says in as many words that
  this is "the point at which it is strictly needed" and that "this subgoal still owns the
  call".
- [ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md) inserted this phase on
  2026-09-03. PR #20 replaced its phase-3 Open section's void citation — the `_cuda.py` it
  cited under `.scratch/align-poc/tokalign` does not exist.
- Phase 2 landed as PR #21; its plan is `feat-hmm-viterbi-cython` under `docs/plan/`.
- The `(min, +)` correction of 2026-09-09 matters most here: an associative scan over time
  is the one candidate that would be *built* from the semiring, and building it in max-plus
  inverts every comparison.

## Notes

<!-- running log -->
