# exp/hmm-param-representation

**Status**: active
**Created**: 2026-09-16
**Subgoal**: Settle the parameter representation — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Establish what a candidate move costs today, before comparing representations
  - [ ] Build a search-shaped workload without a search loop: candidates one state larger or smaller than the incumbent, built and mostly discarded, at S in {5, 16, 50}
  - [ ] Split the cost of each candidate into `HMMParams` construction (validation, freezing, cached properties) and the scoring that follows it: a forward pass, and a `baum_welch` re-convergence. If construction is a negligible fraction of scoring, the dense options are already decided on cost
- [ ] Measure the three representations against that workload
  - [ ] Reallocate: today's frozen dense arrays, as the baseline
  - [ ] Over-allocate and slice: say how a view into a larger buffer interacts with ADR 0017's frozen buffers and cached properties
  - [ ] Edge list: what it saves at the densities learned models reach, and what reopening all eight kernel files and both `FORMALIZATION.md` contracts would cost
- [ ] Decide, and record the decision where it binds
  - [ ] Decide whether the result warrants an ADR. Changing the storage contract ADRs 0015 and 0017 left open would; confirming the current layout may only need a note in `core.md`
- [ ] Land the representation split and merge will be written against, or record that nothing changes and why
