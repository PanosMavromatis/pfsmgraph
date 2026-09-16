# exp/hmm-param-representation

**Status**: active
**Created**: 2026-09-16
**Subgoal**: Settle the parameter representation — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [x] Establish what a candidate move costs today, before comparing representations
  > **Q:** Keep the measurement as a tracked script so the result can be re-run, or record the numbers in the plan only?
  > **A:** Track one combined script, `.scratch/hmm-lush/measurements/param_representation_move_cost.py`, following the ADR 0020 and `viterbi_batch_speed.py` convention. The rest of this branch depends on this result, so it should be re-runnable, not just quoted.
  > **Done:** Building a candidate is **about 0.005% of a move's cost** on the `python` backend (build ÷ score between 9.3e-6 and 6.6e-5), and still negligible on Cython, where a single EM cycle costs 32–102 builds and a convergence runs 80–210 cycles. So the two dense options can't be separated on cost: over-allocating and slicing could save at most the ~1 ms build. Both tables reproduce from the tracked script.
  - [x] Build a search-shaped workload without a search loop: candidates one state larger or smaller than the incumbent, built and mostly discarded, at S in {5, 16, 50}
    > **Note:** Workload: from an incumbent trained for 20 cycles, a split-shaped candidate (a partner for state 0, inbound and initial mass halved, outbound fibres randomised at noise 0.1) and a merge-shaped one (state 1 folded into 0, rows weighted by `state_p`), at S in {5, 16, 50}, on `set11a_dInt`: 1449 symbols over 25, the corpus behind the master plan's 62,500-entry figure. It models what split and merge cost, not their exact semantics; that belongs to the split and merge subgoals.
  - [x] Split the cost of each candidate into `HMMParams` construction (validation, freezing, cached properties) and the scoring that follows it: a forward pass, and a `baum_welch` re-convergence. If construction is a negligible fraction of scoring, the dense options are already decided on cost
    > **Note:** On `python`, building (new arrays + `HMMParams` + the `state_p` solve) is 0.17 ms at S=5 and 1.2 ms at S=50. Scoring is `_suggest_d` (0.66–1.82 s) plus a `baum_welch` convergence (6.5–21 s, 80–210 cycles). Within building, `HMMParams`' validation and freezing is about the same as the array operations, and the stationary solve is smaller than either.
    > **Note:** **The numpy forward pass is nearly flat in S**: 35 ms at S=5, 43 ms at S=16, 79 ms at S=50, for 100× more arcs. So on the default backend the time is per-timestep overhead over 1449 steps, not per-arc work, and an edge list, which reduces arcs, attacks the cheaper part. That weakens the edge list's case on the backend the search defaults to.
    > **Note:** **On Cython the per-arc work does start to count.** Speedup over numpy falls from 11.2× at S=5 to 4.7× at S=50 and 1.8× at S=150, where a cycle is 547 ms. So the edge list isn't ruled out at large S, but its case is kernel speed during scoring, not storage cost. Pursuing it means reopening the eight kernel files and both `FORMALIZATION.md` documents, which is goal 2's question to weigh.
    > **Note:** **`_suggest_d` always runs the numpy forward pass**, because `_data_description_length` has no `backend=`, so each of its ~20 Brent probes is a numpy pass whatever backend trains the model. At S=5 one call is 0.80 s, about 100 Cython EM cycles, and the search calls it on every accepted move. Found here by chance; it is about the cost of scoring, not how parameters are stored, and this branch does not change it.
- [ ] Measure the three representations against that workload
  - [ ] Reallocate: today's frozen dense arrays, as the baseline
  - [ ] Over-allocate and slice: say how a view into a larger buffer interacts with ADR 0017's frozen buffers and cached properties
  - [ ] Edge list: what it saves at the densities learned models reach, and what reopening all eight kernel files and both `FORMALIZATION.md` contracts would cost
- [ ] Decide, and record the decision where it binds
  - [ ] Decide whether the result warrants an ADR. Changing the storage contract ADRs 0015 and 0017 left open would; confirming the current layout may only need a note in `core.md`
- [ ] Land the representation split and merge will be written against, or record that nothing changes and why
