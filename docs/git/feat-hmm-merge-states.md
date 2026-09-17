# feat/hmm-merge-states

**Created**: 2026-09-16
**Base**: main at 32634c5
**Status**: active

## Purpose

Port `merge-states` (`hmm-param.lsh:218-369`), the second of revision 04's two topology moves, as a free function over `HMMParams` beside `_split_state` in the private `_topology.py`. It builds a candidate only: remove the higher-indexed state of the pair, remap indices through an `old-state-numbers` table, sum the inbound transition mass, and average the pair's outbound transitions and emissions weighted by their stationary probabilities. Three things make this a branch rather than a transliteration: the `:262` initial-distribution defect, which ADR 0022 §1 has already decided should be fixed; what a merge should do when it produces a reducible chain, whose `stationary_distribution` raises; and the arrays a merge can produce that `HMMParams` rejects.

## Scope

- Read `merge-states` line by line and state what it does to each of the three arrays, before writing any code
- Confirm the `:262` fix ADR 0022 §1 already decides, and whether it needs an ADR amendment
- Decide the reducible-chain case: reject at proposal time or allow to raise, given that the merge itself consumes `state_p`
- Decide what a merge does with arrays `HMMParams` rejects (an all-zero transition row), and where `safe_divide` gets its first consumers
- Fix `stationary_distribution`, which solves some reducible chains silently, with a structural closed-class helper in `_numeric` that the merge reuses for its weights (goal 3)
- Implement `_merge_states` with property tests, including a split-then-merge round trip; update `core.md`

## Context

- Master plan: revision 04-hmm-v0.3.0, "Implement state merge" (`docs/plan/TODO.md`)
- `feat/hmm-split-state` (PR #45) landed `_topology.py` and [ADR 0022](../design/adr/0022-state-split-initialisation.md). Its goal 6 moved `try-merge`/`suggest-merge` to the scored-primitives subgoal, so this branch delivers the move alone
- `HMMLIB-ACCOUNT.md` §5 describes the surgery and records the `:262` defect beside `:172`
- `stationary_distribution` raises `ValueError` on most reducible chains, though goal 3 found 18 of 254 random ones solved silently, and `HMMParams.state_p` is a `cached_property`, so the failure surfaces on attribute access
- `HMMParams` rejects a zero transition row with no exemption and validates no fibre on a dead arc
- `safe_divide` has had no consumer since 0.1.0; the original's `safe-/` calls in the merge are among its fifteen call sites
- `_topology.py` is already in `meson.build`'s `install_sources`, so no build change is expected

## Notes

<Running log.>
