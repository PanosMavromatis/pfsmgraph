# feat/hmm-hmmlearn-oracle

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Validate the numpy reference against an external oracle — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [x] Add `hmmlearn` as a test-only dependency
  > **Q:** How should the dev group bound `hmmlearn`: a floor `>=0.3.3`, an exact `==0.3.3`, or a floor with a `python_version < '3.14'` marker (no 3.14 wheels)?
  > **A:** Floor `>=0.3.3`, like every other dev-group entry. `uv.lock` pins the exact resolution, so a changed oracle shows up as a lock diff under `uv lock --upgrade`.
  - [x] Add it to the root `dev` group and `uv lock`; confirm the resolution keeps numpy within the existing `numba-cuda` ceiling
    > **Note:** 0.3.3 (2024-10-31) is the latest release; it has cp38-cp313 wheels and none for 3.14. It brings `scipy` and `scikit-learn` (plus `joblib`, `threadpoolctl`, `cloudpickle`, `narwhals`), the dev group's first compiled dependencies beyond numpy. The lock diff is 405 insertions and **0 deletions**, so no existing resolution moved; numpy stays at 2.4.6 in the environment. Smoke-tested on numpy 2.4.6 under `-W error`: `CategoricalHMM` with `init_params=''`, `tol=0` runs exactly `n_iter` cycles from supplied parameters, which goal 4 needs.
  - [x] Confirm nothing under `packages/*/src/` imports it and no member declares it
    > **Note:** the only hit under `packages/` is a docstring mention in `_forward_backward.py`; no member's `pyproject.toml` names it. Suite still 509 passed with `scipy`/`scikit-learn` installed.
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
