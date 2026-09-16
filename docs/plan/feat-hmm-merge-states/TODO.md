# feat/hmm-merge-states

**Status**: active
**Created**: 2026-09-16
**Subgoal**: Implement state merge — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Specify what `merge-states` does to each array before writing code
  - [ ] Transcribe `hmm-param.lsh:218-369` into a per-array statement of the result, including the `old-state-numbers` remap and which assignments later ones overwrite
  - [ ] Identify every `safe-/` call and what each one guards
- [ ] Confirm the `:262` initial-distribution fix under ADR 0022 §1
  - [ ] Establish what `:262` does on the tracked fixtures, as goal 2 of the split did for `:172`
  - [ ] Decide whether ADR 0022 needs an amendment, or already covers the merge
- [ ] Decide the reducible-chain case
  - [ ] Establish when a merge makes the chain reducible, and whether the merge's own `state_p` read can fail first
  - [ ] Decide: reject at proposal time, or allow `stationary_distribution` to raise
- [ ] Decide what a merge does with arrays `HMMParams` rejects
  - [ ] Establish whether a merge can produce an all-zero transition row, and from what inputs
  - [ ] Decide where `safe_divide` is consumed and what a zero weight means
- [ ] Implement `_merge_states` with tests
  - [ ] Write the function in `_topology.py`
  - [ ] Property tests on random models and the tracked fixtures, exact where the arithmetic allows, and mutation-checked
  - [ ] A split-then-merge round trip
  - [ ] Update `core.md` and regenerate `AGENTS.md`
