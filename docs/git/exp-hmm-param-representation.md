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
