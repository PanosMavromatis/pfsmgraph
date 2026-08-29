# 0002. Three-phase algorithm lifecycle: Python, then Cython, then CUDA

- **Status:** Accepted
- **Date:** 2025 (proof-of-concept); formalized 2026-08-29
- **Source:** PRD §1.2, §6 — inherited from the proof-of-concept alignment library

## Context

Sequence alignment and Baum-Welch are dynamic programming. DP is quadratic in sequence
length and embarrassingly awkward to parallelize: each cell depends on its neighbors, so
the naive parallel decomposition does not exist. Making these algorithms fast is
therefore a real engineering project, not a matter of picking a faster library.

At the same time, a fast implementation of the *wrong recurrence* is worthless, and DP
recurrences are easy to get subtly wrong — off-by-one in the traceback, mishandled
affine gap state, an incorrect boundary row. Debugging a wrong recurrence inside a CUDA
kernel is dramatically more expensive than debugging it in Python.

The two pressures pull in opposite directions, and the order in which they are addressed
is the whole decision.

## Decision

**Wherever dynamic programming appears, implementations are written in three phases,
strictly in this order:**

1. **Pure Python — for correctness.** The readable reference implementation. It defines
   the algorithm's semantics and is the oracle every later phase is checked against.
2. **Cython — for performance.** Typed memoryviews, `boundscheck(False)`,
   `wraparound(False)`, no Python object access in the inner loop. Single-threaded, same
   recurrence.
3. **Numba CUDA, anti-diagonal wavefront — for scale.** The DP matrix is traversed along
   anti-diagonals, since all cells on one anti-diagonal are mutually independent and
   depend only on the two preceding anti-diagonals. That is the parallel decomposition
   DP does admit.

No phase is skipped, and no phase begins before the previous one is correct. Every phase
is retained and shipped, not replaced.

## Consequences

### Positive

- **The reference implementation never disappears.** Phase 1 remains the executable
  specification, so "what is this kernel supposed to compute?" always has an answer that
  is code rather than a comment.
- **Optimization bugs are separable from algorithm bugs.** When phase 2 or 3 disagrees
  with phase 1, the algorithm is not in question — only the optimization is. This
  collapses the search space of most debugging sessions.
- **Environment degradation is graceful.** No CUDA device, or no compiler, still leaves
  a working library. This is what allows the GPU dependencies to be optional extras
  ([ADR 0004](0004-gpu-backends-and-optional-dependency-strategy.md)).
- **The anti-diagonal formulation is reusable.** It is the same transformation for every
  DP kernel in the family, so the cost of understanding it is paid once. PRD §11 notes
  the proof-of-concept Needleman-Wunsch `.pyx` is explicitly the reference template for
  future kernels and wavefront passes.

### Negative / costs

- **Three implementations of every kernel must be kept in agreement, forever.** This is
  the dominant cost and it grows linearly with the number of algorithms. It is tolerable
  only because [ADR 0003](0003-one-parameterized-test-suite-per-algorithm.md) makes the
  agreement machine-checked rather than assumed; without that control, this ADR would be
  a liability.
- **A change to a recurrence is a three-place change.** Fixing a traceback bug means
  fixing it three times, in three languages.
- **Time-to-fast is longer.** Phases 1 and 2 must be finished before phase 3 begins,
  even when it is obvious from the outset that the GPU path is the one that matters.
- **The compiled phase drags in build machinery** — meson-python, a C compiler, `ninja`
  — for the packages that reach phase 2
  ([ADR 0008](0008-per-package-build-backends.md)).

## Alternatives considered

- **Write Cython (or CUDA) directly and skip the Python reference.** Rejected: it
  removes the oracle. Every subsequent disagreement becomes an argument about intent
  rather than a test failure, and there is nothing to fall back to on a machine without
  a GPU or compiler.
- **Keep the Python version only as a discarded prototype.** Rejected for the same
  reason: a reference implementation that is not executed is a reference implementation
  that has silently drifted.
- **Use an array library (NumPy vectorization) instead of Cython.** Rejected as
  insufficient for DP specifically — the sequential cell dependency defeats
  straightforward vectorization, which is exactly why the anti-diagonal reformulation is
  needed at all.
- **A row-wise or block-wise GPU decomposition instead of anti-diagonal.** Rejected: it
  does not respect the dependency structure without serializing, whereas the
  anti-diagonal wavefront does so by construction.

## Scope

This lifecycle applies **only where dynamic programming is involved**. It is not a
blanket mandate to optimize everything. Consequently the family is heterogeneous in
build needs — `align` and the Baum-Welch core of `hmm` reach phases 2 and 3, while
`dataseq`, `dl`, and probably `hseg` stay pure Python. That heterogeneity is the direct
cause of [ADR 0008](0008-per-package-build-backends.md).
