# feat/hmm-search-loop

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Design the automatic search loop — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Specify the loop before writing code, and mark it as new work — the original has no such loop (`HMMLIB-ACCOUNT.md` §11)
  - [ ] Read `hmm-trainer-view.lsh`'s thirteen buttons as a requirements list, not a UI
  - [ ] Decide how a move is proposed and when a candidate is re-converged with `baum_welch` before scoring
  - [ ] Decide the acceptance rule against the total description length — totals compared, never subtracted, so two impossible models cannot produce `nan`
  - [ ] Decide the rollback path and the stopping criterion, starting from a one-state model as `model-starting-size` does
- [ ] Decide the parameters the scored primitives left open, measuring where a measurement can settle it
  - [ ] How `d` is chosen: `_suggest_d`'s Brent local minimum or a global integer scan over `[1, 10000]`; a scan must arrive by breaking the pinning test in `test_mdl.py`
  - [ ] The size of the split's minimum EM budget (ADR 0022 §4)
  - [ ] Whether the forward pass inside the total is compiled now or deferred (`merge_round_cost.py`: `_suggest_d` is 53–62% of a trial)
- [ ] Record the decisions as ADR 0023, with a row in `docs/design/adr/README.md`
- [ ] Implement the loop and test it without an oracle
  - [ ] A private search function driving `_suggest_move`, with the draw contract carried from a `SeedSequence` down to `_split_state`, and its docstring marking it as new work
  - [ ] Property and constructed-case tests: an accepted move never raises the total, a rejected move leaves the incumbent exactly as it was, the search terminates, and all-impossible rounds are handled
  - [ ] Sync `core.md`, and write the API documentation if anything becomes public
