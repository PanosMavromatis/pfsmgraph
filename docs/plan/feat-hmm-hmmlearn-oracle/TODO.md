# feat/hmm-hmmlearn-oracle

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Validate the numpy reference against an external oracle — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [ ] Add `hmmlearn` as a test-only dependency
  - [ ] Add it to the root `dev` group and `uv lock`; confirm the resolution keeps numpy within the existing `numba-cuda` ceiling
  - [ ] Confirm nothing under `packages/*/src/` imports it and no member declares it
- [ ] Settle the mapping between `CategoricalHMM` and the arc-emission parameters
  - [ ] Derive how `startprob_`, `transmat_` and `emissionprob_` become `init_state_p`, `transition_p` and `output_p` with emission depending only on the destination, including the first step's N+1 geometry
  - [ ] Settle the symbol axis: `hmmlearn` codes from 0 against our user symbols from `USER_BASE`
- [ ] Check the forward-backward against `hmmlearn`
  - [ ] Log-likelihood against `_description_length` (bits) over random reduced models
  - [ ] State posteriors against `_state_posteriors`
- [ ] Check the EM loop against `hmmlearn`'s `fit`
  - [ ] Same start, fixed iteration count on both sides, parameters compared elementwise
  - [ ] Multi-record corpora through `hmmlearn`'s `lengths`
- [ ] Update the docs: test counts in `core.md`, and what the oracle found
