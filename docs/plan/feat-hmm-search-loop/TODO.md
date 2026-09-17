# feat/hmm-search-loop

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Design the automatic search loop — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [x] Specify the loop before writing code, as a port of `Training/hmm-train-new-nw` with named departures *(corrected 2026-09-17: this read "mark it as new work — the original has no such loop (`HMMLIB-ACCOUNT.md` §11)")*
  > **Q:** `Training/hmm-train-new-nw` and `hmm-train-load-nw` run `(repeat N (suggest-move) (keep-model))` headless, so the original does have a loop. Frame ours as a port with departures, as a faithful default with opt-in options, or as new work citing it as prior art?
  > **A:** Port with departures. The `-nw` loop is the original; acceptance, stopping and trajectory selection are recorded as named departures, as ADR 0022 did for the split. §11, `core.md`, the branch doc and the master-plan wording get corrected.
  > **Q:** What neighbourhood does each round propose: the full `suggest-move` (every merge pair and every split trial), splits only while growing, or full with a cap on merge pairs?
  > **A:** Full `suggest-move`, as the original. Pruning pairs is the alignment-seeded revision's job.
  > **Q:** Is the accepted winner re-converged again before it becomes the incumbent?
  > **A:** No. The incumbent is exactly the `TrialResult` that won — same params, `d` and total — as the original's "No need to run Baum-Welch or suggest-d on the new model" (`:1071`).
  > **Q:** How are a search's random draws keyed: one entropy keyed by role, or `SeedSequence.spawn()` per round?
  > **A:** One entropy, keyed by role: the starting model draws from `spawn_key (0,)`, round `r` from `(1, r)`, so split trial `(s, t)` of round `r` from `(1, r, s, t)`. Any trial is rebuildable from `(entropy, r, s, t)` with no spawn-counter state.
  > **Q:** What is the acceptance rule: a walk with best-so-far, greedy improvement, or the original exactly (unconditional keep for N moves, returning the trajectory)?
  > **A:** A walk with best-so-far. Each round's best neighbour becomes the incumbent, as in the original. The loop tracks the best model seen by strict `<`, returns it, and stops after `patience` rounds without improving it. `patience=0` is greedy exactly.
  > **Q:** Should improving the best require a margin in bits?
  > **A:** No. Strict `<` with no margin. A tie is not an improvement, so it uses up patience. A margin against `d`'s rounding noise waits until that noise is measured.
  > **Q:** When does the search stop: on patience, `max_rounds` or a dead end; on patience or a dead end only; or on `max_rounds` only?
  > **A:** On whichever comes first: `patience` rounds without a new best; `max_rounds` rounds (the original's N, a required keyword so termination never depends on the data); or a round with no possible move (every candidate impossible, or an empty round). The result records which one fired.
  > **Q:** Where does a search start: from optional start params, or always from the one-state model?
  > **A:** From optional start params. `start=None` builds the one-state model (`hmm-train-new-nw`); `start=HMMParams` resumes from a given model (`hmm-train-load-nw`). Either way the start is converged unfloored and scored before round 1.
  > **Q:** What does the search return: the best plus the winners' trajectory, or also every ranked round?
  > **A:** The best plus the winners' trajectory. A frozen result holds the best `TrialResult`, the start's score, one record per round (the winning `Move`, and whether it set a new best), and the stop reason. Runner-up candidates are not kept.
  - [x] Read `hmm-trainer-view.lsh`'s buttons as a requirements list, not a UI
    > **Note:** The view has **fifteen** buttons, not thirteen; §11's list omits `Keep d` and `Reset d`. As requirements: Try Split/Merge → `_try_*`; Suggest Split/Merge/Move → `_suggest_*`; **Continue for N moves** → the loop; Keep/Reset model → commit or roll back one candidate, guarded by `dirty` (no suggestion while a candidate is pending, no keep without one); Suggest/Keep/Reset d and Update DL → `_suggest_d` plus `d` rollback; Add/Converge → `baum_welch`; the history pane → a per-move log (size, move, data DL, model DL, total, `d`); Test Data DL → a held-out score whose field exists but was never wired ("Test Data Set: none", always 0); Update View → nothing.
    > **Note:** §11's "no headless entry point" and "nothing in the tree loops over them automatically" are false. `hmm-train-new-nw` (tracked) is the headless loop, and the view's "Continue for N moves" is the same loop. Its semantics: `suggest-move` scores every merge and then every split (`*min-split-trials*` = 2 per state), keeps the strict-`<` best, and restores the incumbent after each trial. `keep-model` then commits that best **unconditionally**, even when it raises the total. Nothing stops the loop but N. Each keep saves a numbered model with `_total_dl` and a log line, so choosing the best model from the trajectory was left to a person. When every candidate is impossible, `best-size` stays 0 and the model is set to size 0. `m001_0005_005`'s log is one such trajectory: 3198.38 → 2433.99 → 2131.1 → 1774.03 → 1748.9, four splits and each one lower.
    > **Done:** corrected the false no-loop claim where it had spread: `HMMLIB-ACCOUNT.md` §11, §14 and its provenance note; ADR 0017 (two dated amendments — its premise was false, but the accept ratio it called unknown is fixed by the code at 1 in `S(S-1)/2 + 2S`, which favours its decision); the master plan (`:138`, `:205`, `:208-209`); and the branch doc. `core.md` never carried the claim.
  - [x] Decide how a move is proposed and when a candidate is re-converged with `baum_welch` before scoring
    > **Note:** Most of this was already fixed by `_trials.py`: every candidate is re-converged inside its own trial (a split with a `min_cycles` floor, a merge unfloored), then gets a fresh `_suggest_d` and a call to the total. What the loop adds is the ends of a round. The original's constructor (`hmm-trainer.lsh:58-100`) converges the random starting model unfloored, runs `suggest-d` and keeps it as model 1 before any move, so the incumbent always has a converged, scored total to compare against. A winner is accepted as its trial left it. So the loop runs EM only on the starting model; every other EM cycle happens inside a trial.
  - [x] Decide the acceptance rule against the total description length — totals compared, never subtracted, so two impossible models cannot produce `nan`
    > **Note:** A walk with best-so-far contains greedy exactly, not approximately. Round `r` draws from `spawn_key (1, r)`, so a walk and a greedy run with the same entropy pass through identical models until greedy stops. The walk's best-so-far is therefore never worse than greedy's result. The departure from the original is narrow: the walk itself is the original's, and only choosing the best model from the trajectory (left to a person there) and stopping on patience (a fixed N there) are new. The rule has two places where `inf` can appear and neither subtracts: a move with no result can never become the incumbent, and `winner.total_bits < best.total_bits` is `False` whenever the winner is `+inf`.
  - [x] Decide the rollback path and the stopping criterion, starting from a one-state model as `model-starting-size` does
    > **Note:** Rollback needs no mechanism of its own. ADR 0017 makes parameters frozen values, so a rejected candidate is dropped and the incumbent is a reference, where the original's `reset-model` copied a working copy back. With a walk, the end-of-run rollback is returning the best model instead of the last.
    > **Note:** At `S = 1` the random start cannot affect the search. Every position's posterior is 1 on the only state, so the E-step counts don't depend on the parameters, and the first M-step lands on the corpus's symbol frequencies from any start. The original's `init-random 0.001` draws (init, the one transition row, then the fibre, in that order, from `spawn_key (0,)`) are reproduced for fidelity, not because they matter. A dead end at `S = 1` needs `min_split_trials=0`, since there are no merge pairs.
- [ ] Decide the parameters the scored primitives left open, measuring where a measurement can settle it
  - [ ] How `d` is chosen: `_suggest_d`'s Brent local minimum or a global integer scan over `[1, 10000]`; a scan must arrive by breaking the pinning test in `test_mdl.py`
  - [ ] The size of the split's minimum EM budget (ADR 0022 §4)
  - [ ] Whether the forward pass inside the total is compiled now or deferred (`merge_round_cost.py`: `_suggest_d` is 53–62% of a trial)
- [ ] Record the decisions as ADR 0023, with a row in `docs/design/adr/README.md`
- [ ] Implement the loop and test it without an oracle
  - [ ] A private search function driving `_suggest_move`, with the draw contract carried from a `SeedSequence` down to `_split_state`, and its docstring marking it as new work
  - [ ] Property and constructed-case tests: an accepted move never raises the total, a rejected move leaves the incumbent exactly as it was, the search terminates, and all-impossible rounds are handled
  - [ ] Sync `core.md`, and write the API documentation if anything becomes public
