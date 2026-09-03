# docs/cpu-parallel-phase

**Created**: 2026-09-03
**Base**: main at 8c2ca6a
**Status**: active

## Purpose

Insert a new phase into ADR 0002's algorithm lifecycle: Numba CPU-parallel (`prange`,
anti-diagonal) between Cython and Numba CUDA, renumbering CUDA from phase 3 to phase 4.
The motivating problem is that CUDA hardware is not reliably available locally (no local
CUDA GPU; GCP VMs are not readily available), which currently means the *algorithmic*
correctness of the anti-diagonal wavefront decomposition can't be checked under real
concurrent execution until a VM is provisioned. Numba's CPU target (`@njit(parallel=True)`,
`prange`) gives real multi-core concurrent execution today, on any machine, using the same
library phase 4 already commits to — de-risking the parallel decomposition with no new
packaging tension and no new class of dependency. A Mac-Metal/PyTorch-MPS route was
considered and rejected: it would pull `torch` into the DP-package side of ADR 0004's "GPU
means two unrelated things" boundary for a purpose that ADR doesn't argue for, and the
measured Metal speed gains for workloads like this have historically been marginal — not
worth the tension for a path this project isn't going to ship anyway.

## Scope

- Draft ADR 0016, amending ADR 0002: insert phase 3 (Numba CPU-parallel, anti-diagonal),
  renumber CUDA to phase 4, record the MPS/torch alternative and why it's rejected.
- Add the README index footnote on ADR 0002's row (mirroring the 0008/0012 precedent).
- Update `core.md`'s "Three-phase algorithm lifecycle" invariant to four phases.
- Update `codex.md`'s phase-ordered review-priority list with the new `_cpu_parallel*.py`
  target.
- Ripple the renumbering into revision 02's Viterbi phase subgoals in `docs/plan/TODO.md`
  (moving "settle the anti-diagonal question" to the new phase 3) and into revision 03's
  draft (`docs/plan/planned/03-hmm-v0.2.0.md`, "phases 2 and 3" → "phases 2 through 4").

## Context

Follows directly from `docs/hmmlib-account` (PR #13, merged 2026-09-03) and the standing
`hmm` migration plan (`docs/plan/TODO.md`, revisions 02–04). Not part of that migration
itself — a lifecycle/tooling amendment revision 02's Viterbi work will build against once
opened.

## Notes

_(running log — filled in as work proceeds)_
