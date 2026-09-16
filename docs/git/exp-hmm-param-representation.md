# exp/hmm-param-representation

**Created**: 2026-09-16
**Base**: main at 68afee4
**Status**: active

## Purpose

Settle how `pfsmgraph.hmm` stores model parameters before state split and merge are written against a layout. The search builds a new candidate model for every move it tries and rejects most of them, so this is the one revision where the data structure, not the recurrence, may be the cost. Three options: reallocate on every accepted move (what `HMMParams` does today, as a frozen value), over-allocate and slice, or keep no dense array and use an edge list. Measure first; the answer may well be "keep the dense frozen arrays", which is why this is `exp/` rather than `feat/`.

## Scope

- Measure what a move costs today, separating `HMMParams` construction from the re-convergence that scores a candidate
- Measure the three representations against a search-shaped workload
- Decide, and record the decision where it binds
- Land whatever representation split and merge will be written against, possibly nothing

## Context

- Master plan: revision 04-hmm-v0.3.0, subgoal moved ahead of state split on 2026-09-16 so split and merge are written once
- ADR 0015 fixes the model's semantics and deliberately not its storage; ADR 0017 makes parameters a frozen value and already settled rollback
- ADR 0020 and both `FORMALIZATION.md` documents specify kernels built on a host-built `(S, S, U)` dense table
- Measured at opening: learned topologies are sparse in emissions (3.7% and 5.7% of entries live) far more than in arcs (32% and 50%)
- An edge list would reopen eight `dp-compile`-gated kernel files and both specifications; the dense options touch none

## Notes

**2026-09-16 — what a move costs.** Building a candidate (new arrays, `HMMParams`, the `state_p` solve) is 0.17 ms at S=5 and 1.2 ms at S=50. Scoring it (`_suggest_d` plus a `baum_welch` re-convergence) takes 6.5–21 s, so building is **about 0.005% of a move** on `python`, and still negligible on Cython, where one EM cycle costs 32–102 builds. Reallocate and over-allocate-and-slice therefore can't be told apart on cost. The numpy forward pass is nearly flat in S (35 ms at S=5, 79 ms at S=50), so on the default backend an edge list attacks the cheaper part; on Cython the per-arc work starts to count at large S (speedup over numpy falls to 1.8× at S=150), which leaves the edge list as a question of kernel speed, not storage. Also found: `_suggest_d` always runs the numpy forward pass, about 0.8 s per call at S=5 whatever the training backend. Evidence: `.scratch/hmm-lush/measurements/param_representation_move_cost.py`.
