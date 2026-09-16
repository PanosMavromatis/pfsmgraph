# feat/hmm-split-state

**Created**: 2026-09-16
**Base**: main at 883c345
**Status**: active

## Purpose

Port `split-state` (`hmm-param.lsh:142-215`), the first of revision 04's two topology moves, as a free function over `HMMParams`: add one state at index `old-size`, halve the initial and inbound transition mass between the original and its new partner, copy the original's inbound emissions to the partner, and randomize every outbound emission fibre at noise width `0.1`. Two decisions land with it and are the reason this is a branch rather than a transliteration: what to do about the `hmm-param.lsh:172` defect, which seeds every uninvolved state's initial probability from its *stationary* probability, and whose `numpy.random.Generator` a split draws from and where it enters the public API, which the trial budget in `suggest-split` makes load-bearing.

## Scope

- Read `split-state` line by line and state what it does to each of the three arrays, before writing any code
- Decide the `:172` defect: reproduce, fix, or fix and renormalize
- Settle the reproducibility contract: whose `Generator`, where it enters, and in what order the fibres draw from it
- Implement the split with tests, including constructed cases the fixtures cannot exhibit
- Decide where `try-split` and `suggest-split` belong: here, or in the scored-primitives subgoal

## Context

- Master plan: revision 04-hmm-v0.3.0, "Implement state split" (`docs/plan/TODO.md`)
- `exp/hmm-param-representation` (PR #44) settled the shape: split builds fresh arrays and returns `HMMParams(init_state_p, transition_p, output_p, params.vocabulary)`, as `baum_welch` does after each M-step. Its `split_arrays` measurement helper is a workload shape, not a port, and should not seed this one
- `HMMLIB-ACCOUNT.md` §5 describes the surgery and records the `:172` / `:262` defect, provenance unknown; `merge-states` carries the same defect at `:262`, so whatever is decided here binds the merge subgoal too
- `rand_p_vector(size, noise_width, rng)` (`_numeric.py`) already takes a required `Generator` and returns a new array, because the original assigns rather than perturbs
- `HMMParams` requires the six ADR 0011 reserved fibres to be exactly zero, and validates no fibre on a dead arc
- Revision 02 recorded the δ-seeding degenerate case (`init_p == 0` on a state that can emit `begin`) as masked by learned topologies and unmasked by exactly this operation
- `try-split` (`hmm-trainer.lsh:749-770`) re-converges with `run-converge` and chooses `d` with `suggest-d`; `suggest-split` (`773-843`) tries every state `*min-split-trials*` + `int(*split-trials-per-state-p* × state_p)` times, 2 + 0 in the shipped settings

## Notes

**2026-09-16 — read at opening, not yet checked against the generated C.**

- **Every one of the partner pair's outbound fibres is randomized, including the ones the "copy inbound emissions" loop just wrote.** The inbound loop writes `new-output-p j last-state` for every `j < old-size`, which includes `j = state-n`, so `state-n → last-state` is copied and then overwritten by the self-or-partner block. The net effect matches §5's description, but the order of assignments is where a port would go wrong.
- **`state-n → last-state` gets its transition from the inbound loop, not the outbound one**: `0.5 × transition-p[state-n, state-n]`. Both new rows still sum to one, but only because of that interaction.
- **`rand-p-vector` over `n-symbols` would violate `HMMParams` as ported.** The original randomizes the whole symbol axis; here the six reserved fibres must stay exactly zero, so the draw is over the user symbols, `n_symbols - USER_BASE`. This also fixes how many draws each fibre consumes, which is part of the reproducibility contract. *(Sharpened at goal 1: the count does not change. Lush's alphabet holds `begin` and `end` as ordinary codes, and they map to user symbols here rather than BOS/EOS, so a Lush whole-alphabet fibre and a user-symbol fibre have the same length.)*
- **The draw order is observable.** The original draws `2 × old-size + 2` fibres in a fixed order (per `j`: `state-n → j`, then `last-state → j`; then `state-n → last-state`, then `last-state → last-state`). A port that draws them in array order produces a different model from the same seed.

**2026-09-16 — goal 2 discussion: the split departs from the original on two axes, to be recorded in one ADR.**

- **Axis 1, decided: `:172` is a defect, and the fix carries `init_state_p`.** The original author confirms that `state_p`, a quantity derived after training to estimate entropies, was never meant to seed a trainable parameter. The split state and its twin share `½·init[s]`, every other state keeps `init[i]`, and `merge-states` `:262` gets the same fix. This restores intent rather than departing from the algorithm. Verbatim reproduction fails `HMMParams` on 13 of the 14 fixture splits.
- **Axis 2, proposed, pending measurement: break the twins' symmetry on inbound transitions, not outbound emissions.** Each predecessor gets its own perturbation of the halving of its arc into the split state. Outbound transition rows are copied, and every emission fibre is copied in both directions, so no learned parameter is discarded. That replaces the original's near-uniform redraw of outbound fibres (`:190-207`), which the Purpose above still describes. Constructed cases show the perturbation carries a real signal only when the twins have more than one distinct predecessor row. Where they have one, which includes every split of the one-state model the search starts from, symmetry breaks only through floating-point rounding magnified at an unstable fixed point (about 100 cycles) or not within `run-converge`'s budget at all. A deliberate emission seed for those splits is therefore a measured arm. Measured in `.scratch/hmm-lush/measurements/split_symmetry_breaking.py`: rounding alone left the one-state `set11a_dInt` split unseparated in 3 of 3 seeds at `w = 0.1`, and a 1% emission seed fixed it; and `run-converge` stops inbound splits early, so on `m008` they separated in 100% of splits and beat the redraw on mean data length only with 200 forced cycles. The branch plan holds the measurement that decides the width and how such states are treated.
