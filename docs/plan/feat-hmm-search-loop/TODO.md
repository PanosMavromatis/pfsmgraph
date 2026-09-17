# feat/hmm-search-loop

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Design the automatic search loop — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [~] Specify the loop before writing code, as a port of `Training/hmm-train-new-nw` with named departures *(corrected 2026-09-17: this read "mark it as new work — the original has no such loop (`HMMLIB-ACCOUNT.md` §11)")*
  > **Q:** `Training/hmm-train-new-nw` and `hmm-train-load-nw` run `(repeat N (suggest-move) (keep-model))` headless, so the original does have a loop. Frame ours as a port with departures, as a faithful default with opt-in options, or as new work citing it as prior art?
  > **A:** Port with departures. The `-nw` loop is the original; acceptance, stopping and trajectory selection are recorded as named departures, as ADR 0022 did for the split. §11, `core.md`, the branch doc and the master-plan wording get corrected.
  - [x] Read `hmm-trainer-view.lsh`'s buttons as a requirements list, not a UI
    > **Note:** The view has **fifteen** buttons, not thirteen; §11's list omits `Keep d` and `Reset d`. As requirements: Try Split/Merge → `_try_*`; Suggest Split/Merge/Move → `_suggest_*`; **Continue for N moves** → the loop; Keep/Reset model → commit or roll back one candidate, guarded by `dirty` (no suggestion while a candidate is pending, no keep without one); Suggest/Keep/Reset d and Update DL → `_suggest_d` plus `d` rollback; Add/Converge → `baum_welch`; the history pane → a per-move log (size, move, data DL, model DL, total, `d`); Test Data DL → a held-out score whose field exists but was never wired ("Test Data Set: none", always 0); Update View → nothing.
    > **Note:** §11's "no headless entry point" and "nothing in the tree loops over them automatically" are false. `hmm-train-new-nw` (tracked) is the headless loop, and the view's "Continue for N moves" is the same loop. Its semantics: `suggest-move` scores every merge and then every split (`*min-split-trials*` = 2 per state), keeps the strict-`<` best, and restores the incumbent after each trial. `keep-model` then commits that best **unconditionally**, even when it raises the total. Nothing stops the loop but N. Each keep saves a numbered model with `_total_dl` and a log line, so choosing the best model from the trajectory was left to a person. When every candidate is impossible, `best-size` stays 0 and the model is set to size 0. `m001_0005_005`'s log is one such trajectory: 3198.38 → 2433.99 → 2131.1 → 1774.03 → 1748.9, four splits and each one lower.
    > **Done:** corrected the false no-loop claim where it had spread: `HMMLIB-ACCOUNT.md` §11, §14 and its provenance note; ADR 0017 (two dated amendments — its premise was false, but the accept ratio it called unknown is fixed by the code at 1 in `S(S-1)/2 + 2S`, which favours its decision); the master plan (`:138`, `:205`, `:208-209`); and the branch doc. `core.md` never carried the claim.
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
