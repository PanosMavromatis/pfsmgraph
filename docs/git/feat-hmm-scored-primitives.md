# feat/hmm-scored-primitives

**Created**: 2026-09-17
**Base**: main at c9b252c
**Status**: active

## Purpose

Port the scored primitives of revision 04's topology search: the per-trial wrappers `try-split` (`hmm-trainer.lsh:749-770`) and `try-merge` (`:858-881`), and the ranking primitives `suggest-split`, `suggest-merge` and `suggest-move` (`:952-1073`). A trial builds a candidate with `_split_state` or `_merge_states`, re-converges it with `baum_welch`, and scores it by **calling** `_mdl._total_description_length`, never by assembling the score from its halves. The suggest primitives return candidates ranked. The branch delivers ranked moves, not the search loop: acceptance, rollback, stopping and the global `d` scan belong to the next subgoal. The suggest primitives are reviewable line by line against the original; `try-split` is not wholly, because ADR 0022 departs from it on purpose, and the plan must say where the port stops being faithful.

## Scope

- Read `try-split`, `try-merge` and `suggest-*` line by line: what is re-converged, how `d` is chosen per trial, and the draw order
- Port `try-split` with ADR 0022's obligations: a minimum EM budget before `run-converge` may stop (§4, a parameter here, sized by the loop), unrounded data description length beside the total (§5), and one `Generator` per (round, state, trial)
- Port `try-merge`, and decide the merge weights: reweight from E-step counts, or filter transient pairs with the recurrent mask
- Port `suggest-split`, `suggest-merge` and `suggest-move` as ranked results, handling impossible (`+inf`) candidates without subtracting totals
- Measure the all-pairs merge cost against model size, record it, and update `core.md`

## Context

- Master plan: "Port the scored primitives", revision 04-hmm-v0.3.0, and its two notes (the `try-*` wrappers moved here on `feat/hmm-split-state` goal 6; the merge-weight question from `feat/hmm-merge-states` goal 4)
- [ADR 0022](../design/adr/0022-state-split-initialisation.md) §4, §5 and its Resolved draw contract
- `_mdl.py`: `_total_description_length`, `_suggest_d` (Brent, local minimum kept on purpose)
- `_topology.py`: `_split_state` (PR #45), `_merge_states` (PR #46)
- `baum_welch` and its `run-converge` stopping rule
- Out of scope: the search loop, its acceptance rule and the choice between Brent and a global `d` scan (next subgoal)

## Notes
