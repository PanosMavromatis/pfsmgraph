# feat/hmm-forward-phases

**Created**: 2026-09-15
**Base**: main at 6404003
**Status**: active

## Purpose

Carry the forward-backward recurrence, and the batched E-step `baum_welch` trains with, through ADR 0016's phases 2–4 (Cython, Numba CPU-parallel, Numba CUDA), starting from a phase-0 `FORMALIZATION.md`. Each phase is reachable through `baum_welch(backend=...)` and is held bit-exact to the numpy reference on one host, as ADR 0020 requires.

## Scope

- Phase 0: `docs/design/algorithms/forward_backward/FORMALIZATION.md`, recovered from `_forward_backward.py` with its SHA-256, including the batched recurrence and the parallel decomposition
- Decide the phase-3 axis: associative scan over time vs batch/state parallelism
- Phase 2: Cython kernel, registered in `_backends.py`
- Phase 3: Numba `prange` kernel
- Phase 4: Numba CUDA kernel, settling numba-cuda's multiply-add contraction behaviour (left open by ADR 0020)

## Context

- Master plan: revision 03-hmm-v0.2.0, forward-recurrence subgoal
- ADR 0020 (scaled probability domain, ordered sums, no fused multiply-add), ADR 0016 (four phases), ADR 0021 (backend table)
- Precedent: `docs/design/algorithms/viterbi/FORMALIZATION.md`; Viterbi phase 3 measured a flat fork/join cost per timestep
- The dp-compile phase skills' registry template doesn't match ADR 0021 (master-plan note); write rows in the ADR's shape

## Notes
