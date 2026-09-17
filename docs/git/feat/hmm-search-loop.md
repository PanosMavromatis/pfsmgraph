# feat/hmm-search-loop

**Created**: 2026-09-17
**Base**: main at 1eca86b
**Status**: active

## Purpose

Design and implement the automatic topology search: starting from a one-state model, propose
state merges and splits, re-converge and score each candidate by total description length, accept
or roll back against the incumbent, and stop. **This is a port with named departures.** The
Lush original's driver is `Training/hmm-train-new-nw`, `(repeat N (suggest-move) (keep-model))`,
which keeps every round's best neighbour unconditionally and stops only at N; the view's
`Continue for ` button is the same loop. `HMMLIB-ACCOUNT.md` §11 said no such loop existed,
and was corrected on this branch. Acceptance against the incumbent, a stopping rule and
trajectory selection are departures from that loop, and the plan, the docstrings and ADR 0023
must name them as such. There is still no numeric oracle for a trajectory: the only tracked
one is `m001_0005_005`'s training log, whose splits used the original's randomness, which
ADR 0022 changed.

## Scope

- Specify the loop before writing code: move proposal, when a candidate is re-converged, the
  acceptance rule (totals compared, never subtracted), rollback, the stopping criterion, and the
  one-state start
- Decide the parameters the scored primitives left open: how `d` is chosen (Brent's local minimum
  or a global scan), the split's EM floor, and whether the forward pass inside the total is compiled
  now or deferred — measuring where a measurement can settle it
- Record the decisions as ADR 0023
- Implement the loop privately on top of `_suggest_move`, tested by properties and constructed
  cases rather than an oracle
- Sync `core.md`, and the API docs if anything becomes public

## Context

- Master plan: `docs/plan/TODO.md`, revision `04-hmm-v0.3.0`, "Design the automatic search loop"
- PR #47 (`feat/hmm-scored-primitives`): `_try_split`, `_try_merge`, `_suggest_split`,
  `_suggest_merge`, `_suggest_move` in `_trials.py`; ranking by stable sort on the total, impossible
  candidates as `Move(result=None, total_bits=inf)`, split trials seeded by
  `SeedSequence(entropy, spawn_key=round_key + (s, t))`
- [ADR 0022](../../design/adr/0022-state-split-initialisation.md) §4: whatever re-converges a split
  owes it a minimum EM budget
- `feat/hmm-mdl`: `_suggest_d` keeps the original's Brent local minimum (`d = 29` where `d = 13` is
  10.46 bits cheaper on the one-state model); a global scan must break the pinning test in
  `test_mdl.py`; `inf - inf = nan` inside Brent
- `.scratch/hmm-lush/measurements/merge_round_cost.py`: a merge round costs its pair count, 1.3 to
  1.7 s per pair on a 4-vCPU host, with `_suggest_d` taking 53–62% of each trial
- `hmm-trainer-view.lsh`: fifteen buttons, read as a requirements list rather than a UI
- `Training/hmm-train-new-nw`, `hmm-train-load-nw`: the headless loop being ported

## Notes
