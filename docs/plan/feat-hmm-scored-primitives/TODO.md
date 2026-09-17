# feat/hmm-scored-primitives

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Port the scored primitives — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Specify what `try-split`, `try-merge` and `suggest-*` do before writing code
  - [ ] Transcribe `try-split` (`hmm-trainer.lsh:749-770`) and `try-merge` (`:858-881`): what is re-converged, with which stopping rule, and how `d` is chosen per trial
  - [ ] Transcribe `suggest-split`, `suggest-merge` and `suggest-move` (`:952-1073`): trial counts, draw order, ranking and tie-breaking
  - [ ] Mark every point where ADR 0022 or an earlier decision makes the port depart from the original
- [ ] Port `try-split`
  - [ ] Decide the signature: the minimum EM budget (ADR 0022 §4) as a parameter, and one `Generator` per (round, state, trial)
  - [ ] Report unrounded data description length beside the total (§5), scoring through `_total_description_length`
  - [ ] Tests
- [ ] Port `try-merge`
  - [ ] Decide the merge weights: reweight from E-step counts, or filter transient pairs with the recurrent mask
  - [ ] Implement and test
- [ ] Port `suggest-split`, `suggest-merge` and `suggest-move` as ranked results
  - [ ] Handle impossible (`+inf`) candidates by comparing totals, never subtracting them
  - [ ] Tests, including the ranking contract
- [ ] Measure and record
  - [ ] Measure the all-pairs merge cost against model size
  - [ ] Update `core.md`
