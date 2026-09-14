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
- [x] Settle the mapping between `CategoricalHMM` and the arc-emission parameters
  > **Q:** Our EM unties the destination-only emission after one cycle, so `_em`'s output cannot be compared with `hmmlearn`'s `fit`. How should goal 4 check EM: an augmented start state with a test-only tied-emission harness, a single E-step's marginalised counts, or a tied option added to `_em`?
  > **A:** Augmented start state plus a tied harness, in the test only: `k` cycles of our `_expected_counts` and `_m_step`, emission counts summed over the source before normalising, against `fit(n_iter=k)` with `lengths`. It checks the E-step and M-step, not `_em`'s loop, which its own tests cover.
  - [x] Derive how `startprob_`, `transmat_` and `emissionprob_` become `init_state_p`, `transition_p` and `output_p` with emission depending only on the destination, including the first step's N+1 geometry
    > **Note:** the first step is where the formulations differ. Our first symbol is emitted crossing s_0 → s_1, and hmmlearn's first state is our s_1, so its `startprob_` is `init_state_p @ transition_p`, with `transmat_ = transition_p` and `output_p[i, j, USER_BASE + k] = emissionprob_[j, k]`. Measured on S=3, A=4, N=40: log-likelihood (`-dl · ln 2`) against `score` to 6.3e-16 relative, `gamma[1:]` against `predict_proba` to 3.3e-16. The map only runs ours → hmmlearn: recovering `init_state_p` from a given `startprob_` needs `init @ A = π`, which may have no stochastic solution.
    > **Note:** our EM does not stay in the reduced family. After one `_m_step`, `output_p[i, j, k]` varies by up to 0.096 over `i`, and the transition rows differ from `transmat_` by 1.8e-3, because our transition counts include the s_0 → s_1 crossing and hmmlearn's do not. This is the arc-emission model having more parameters, not a defect on either side. It is why goal 4 does not compare `_em`.
    > **Note:** a dedicated start state closes the transition gap exactly. With S+1 states, `init = e_S`, row S set to `startprob_`, and column S zero, state S is occupied only at s_0. Its re-estimated row is γ₁, which is hmmlearn's `startprob_`, and rows `:S` count only the crossings hmmlearn counts. Emission is tied by summing counts over the source. Measured over 5 cycles on three records (30, 1, 17): start 1.1e-16, transitions 2.8e-16, emission 1.4e-16, and the log-likelihood history 2.2e-16 relative against `monitor_.history`.
  - [x] Settle the symbol axis: `hmmlearn` codes from 0 against our user symbols from `USER_BASE`
    > **Note:** no decision needed: hmmlearn code `k` is our `USER_BASE + k`, with the reserved fibres zero as `HMMParams` requires. A sequence goes to `fit` as `(codes - USER_BASE).reshape(-1, 1)`.
- [ ] Check the forward-backward against `hmmlearn`
  - [ ] Log-likelihood against `_description_length` (bits) over random reduced models
  - [ ] State posteriors against `_state_posteriors`
- [ ] Check the EM loop against `hmmlearn`'s `fit`
  - [ ] Augmented start state and tied emission (test-only harness over `_expected_counts` and `_m_step`), `k` cycles against `fit(n_iter=k)`: start row, transitions, emission and the log-likelihood history compared elementwise
  - [ ] Multi-record corpora through `hmmlearn`'s `lengths`
- [ ] Update the docs: test counts in `core.md`, and what the oracle found
