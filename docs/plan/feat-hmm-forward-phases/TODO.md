# feat/hmm-forward-phases

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Carry the forward recurrence through ADR 0002 phases 2 through 4 (revision 03-hmm-v0.2.0)

## Goals

- [ ] Write the phase-0 `FORMALIZATION.md` for forward-backward, recovered from `_forward_backward.py` and the batched E-step, and decide the parallel decomposition (associative scan over time, or batch/state parallelism)
  - [ ] It names ADR 0020's evaluation order and no-fused-multiply-add rule as contract, and carries the source hash
- [ ] Implement phase 2 (Cython) and register it, held bit-exact to the numpy reference
- [ ] Implement phase 3 (Numba CPU-parallel) on the decomposition goal 1 chose, held bit-exact
- [ ] Implement phase 4 (Numba CUDA), settling contraction behaviour, held bit-exact on a device
