# 0022. A state split preserves what was learned, and breaks symmetry on inbound arcs

- **Status:** Accepted
- **Date:** 2026-09-16
- **Source:** none in the PRD — postdates it. Decided on `feat/hmm-split-state`, revision
  `04-hmm-v0.3.0`, goals 2 and 3 of that branch's plan, with the original author of the
  Lush library.

## Context

Revision 04 searches HMM topology by splitting and merging states and keeps whichever model
has the shortest description length. The Lush original's `split-state`
(`hmm-param.lsh:142-215`) adds a twin `p` for a state `s` at index `S`:

- **Initial probabilities.** `s` and `p` each get `½·init[s]`.
- **Inbound arcs.** Every arc into `s` is halved between `s` and `p`, and its emission
  fibre is copied to `p`.
- **Outbound arcs.** Both twins copy `s`'s outbound transition row. `s`'s self-loop is
  halved like any inbound arc, so `p→s` and `p→p` each get `½·T[s,s]`.
- **Outbound emissions.** Every fibre leaving either twin is **redrawn** near-uniform with
  `rand-p-vector … 0.1`.

Two parts of that do not survive review.

**`:172` is a defect.** Its loop seeds every uninvolved state's initial probability from
`state_p`, the stationary distribution, instead of `init_state_p`:
`(new-init-state-p i (state-p i))`. `state_p` is a derived quantity, computed after training
to estimate state entropies. The original author confirms it was never meant to seed a
trainable parameter. `run-add` marks every `init-state-p` line `*** new for init-state-p`
(`hmm-trainer.lsh:599-637`), which suggests the slot was added after `split-state` and this
line was missed. Reproduced verbatim, the vector sums to `Σ_{i≠s} state_p[i] + init[s]`.
That fails `HMMParams`' `SUM_TOL` in **13 of the 14** splits the tracked fixtures allow,
missing by 0.10 to 0.84. It also replaces a trained, concentrated `init` such as
`[0, 1, 0, 0, 0]` with stationary mass spread over every state. EM re-estimates `init`
within one cycle, so no scored model ever carried the defect, but it moved EM's starting
point. `merge-states` repeats it at `:262`.

**The outbound redraw discards learning.** Trained fibres are sharp: on `m008_0001_008`,
state 5's outbound fibres are all `[1, 0, 0, 0, 0, 0]`. The redraw replaces them with vectors
near `[1/6, …, 1/6]`, which EM must relearn. It is the only thing in the original that tells
the twins apart. Without it, twins with identical outbound rows and fibres have identical
backward variables, and EM leaves them identical.

## Decision

### 1. Initial probabilities are carried, and `:172` / `:262` are fixed

Every uninvolved state keeps `init[i]`. The twins share `init[s]` in halves, as `:179-180`
already intend. A merge sums the merged pair's `init` and every other state keeps its own.
This restores the author's intent. It is not a departure from the algorithm.

### 2. Transitions into the split state are perturbed, per predecessor

Each predecessor `j` with `T[j, s] > 0`, including `s` itself, splits its arc as

    T'[j, s] = (½ + w·u_j) · T[j, s],    T'[j, p] = T[j, s] − T'[j, s],    u_j ~ U(−½, ½)

with `w = 0.01`, one independent `u_j` per predecessor. Computing the twin's share as the
remainder makes each predecessor's row sum exactly what it summed before. Zero arcs stay
zero, so the topology does not change. Both twins copy `s`'s outbound transition row, so
the twin's row carries `s`'s self-loop split unchanged.

The perturbation must be **independent per predecessor**. EM can separate the twins only
through the entry ratio `α_t(s)/α_t(p)`. A single shared split would hold that ratio
constant, and the twins would stay a fixed-ratio copy of one state.

### 3. Emission fibres are copied, then lightly seeded on every split

Every fibre into and out of both twins starts as a copy. Each fibre leaving either twin on a
live arc then becomes

    O'[t, j, USER_BASE:] = 0.99 · O'[t, j, USER_BASE:] + 0.01 · rand_p_vector(n_symbols − USER_BASE, 0.1, rng)

for `t ∈ {s, p}`. Fibres on dead arcs are copied unchanged. The mixture sums to one without
renormalising, keeps the ADR 0011 reserved block at exactly zero, and can give mass to a
symbol the learned fibre excluded.

This seed is needed only where the twins have a **single distinct predecessor row**, which
includes every split of the one-state model a search starts from. There the entry ratio is
constant, and the inbound perturbation carries no information. The seed is applied to every
split anyway, because it measured no harm elsewhere and one rule is simpler than two.

### 4. A split needs a minimum re-convergence budget, set where splits are tried

An inbound split starts at almost exactly the incumbent's likelihood. The seed moves it
slightly, and nothing else does. EM's first batches therefore change the description length
by less than `run-converge`'s 0.1-bit threshold, and the default stop fires before the twins
separate. Whatever re-converges a split candidate (`try-split`, and the master plan's search
loop) must run a minimum number of EM cycles before that rule may stop it. The number and
the mechanism are decided there, not here.

### 5. Split trials report unrounded likelihood beside description length

Wherever split candidates are compared or logged, the unrounded data description length,
`_corpus_description_length` of the unquantized parameters, is reported beside the total.
The total moves with `d`'s rounding and `_suggest_d`'s local minimum by amounts comparable to
the differences being judged.

## Consequences

### Positive

- **No learned parameter is thrown away.** Every transition and every emission starts from
  its trained value. Before the seed, a split is an exact reparameterisation: every sequence
  keeps the probability the incumbent gave it, so EM starts from the incumbent's fit rather
  than from partly random fibres.
- **The twins separate along history, which is what a split is for.** With several
  predecessors, `s` gathers counts from the contexts after the predecessors that favour it,
  and `p` from the others.
- **The candidate is always a valid `HMMParams`.** Axis 1 makes `init` sum to one, the
  remainder form keeps rows exact, and the mixture keeps live fibres normalised and reserved
  fibres zero.

### Negative / costs

- **Splits cost more EM.** The budget in §4 is a real cost per candidate: 200 forced cycles
  against a median of about 30-40 under the default stop in the measurement.
- **The draw order changes.** The original drew `2S + 2` emission fibres. A split now draws
  one `u_j` per live predecessor plus one fibre per live outbound arc of each twin. Their
  order is fixed under **Resolved**.
- **Rounding still plays a part where the seed is small.** A 1% seed separated every
  single-predecessor split measured. Whether it always does so within the budget of §4 is
  measured only on the corpora below.
- **This is a departure from the original on axis 2**, and results from the Lush search are
  not reproducible bit for bit. That was already true, since the generators differ.

## Alternatives considered

- **Reproduce `:172` verbatim.** Rejected: `HMMParams` refuses the result on 13 of 14 fixture
  splits, and the author confirms it is a defect.
- **Seed uninvolved states from `state_p`, then renormalise.** Rejected: it passes
  validation but still replaces a trained vector with a derived one. EM repairs it within
  one cycle, which is no reason to introduce it.
- **The original's outbound redraw.** Rejected on the reasoning in Context. On the tracked
  8-state model it separated twins more often under the default stop (62% against 12-38%).
  With a minimum budget, the inbound design separated in 100% of splits, and its mean
  data-length change was −10.8 bits against the redraw's +30.0.
- **Perturb outbound transitions or emissions instead of inbound transitions.** Rejected:
  outbound perturbation separates the twins in a random direction, where inbound perturbation
  separates them by predecessor history, which is the structure a split should find.
- **Seed only single-predecessor splits.** Measured, and equivalent on those splits. Not
  chosen: seeding every split measured no harm and removes a structural test from the
  contract.
- **Skip single-predecessor states as split candidates.** Rejected: the search could never
  leave its one-state start.
- **Rely on floating-point rounding.** Rejected: on `set11a_dInt`'s one-state split,
  `w = 0.1` without a seed left the twins together in 3 of 3 seeds, and `w = 0.01` in 1 of 3.
  Symmetry breaking that depends on evaluation order is not reproducible across hosts or
  within `torch`'s tolerance (ADR 0020).

## Evidence

Scripts are tracked under `.scratch/hmm-lush/measurements/`, run on `cython`, with 3 seeds
per arm.

- **Axis 1.** All 14 splits of the three `set02a_200` fixtures, measured in the branch plan's
  goal 2.
- **Constructed cases** (goal 3; synthetic data, EM with the stop rule disabled).
  - **Several predecessors:** twin gap `8e-4` after 1 cycle, `4e-1` by cycle 100, and `0`
    throughout with exact halves.
  - **One state:** gap at rounding level (`2e-15`), reaching `1.0` by cycle 200, and `0`
    with exact halves. The perturbation acts only through its rounding footprint.
  - **Single predecessor with stochastic outbound:** `6e-11` after 400 cycles.
- **`split_symmetry_breaking.py`**, splitting every state of the three `set02a_200`
  fixtures and of `set11a_dInt` (1449 symbols over 25) models trained at 1, 4 and 8 states.
  - **`set11a` one-state split:** `inbound(0.1)` unseeded separated in 0 of 3 seeds; every
    seeded or budgeted arm separated in 3 of 3, with data change −403 to −446 bits against
    the redraw's −388 to −445.
  - **`m008_0001_008`:** separation 12-38% under the default stop and 100% with 200 forced
    cycles.
  - **Seed on every split against `inbound(0.01)`**, mean data change:
    `m008` +42.1 vs +42.8; `set11a` S=4 −133.6 vs −131.2; S=8 −90.5 vs −81.7;
    S=1 **−433.0 vs −292.8**, with separation 100% vs 67%. Seeding does not fix the early
    stop: `m008` separation stays at 12%.
  - **Totals are noisy.** The best single split swings by 100+ bits across seeds, and totals
    moved against the data half, e.g. `set11a` S=4 total +660 vs +593 with a better data
    fit. That is the case for §5.
- **One difference from §3.** The script seeds a twin's fibres on dead arcs too. EM never
  moves a zero transition, so this cannot change a result, and the contract copies them to keep
  every fibre summing to one or zero.
- **Caveat.** The `set11a` models at 4 and 8 states score worse in total than its one-state
  model, so they stand in for the multi-state candidates a search would try; a search would
  not choose them as its current model.

## Open

- **The minimum EM budget after a split**: its size, and whether it is a cycle count or a
  change in the stop rule. It belongs to `try-split` and the search loop (§4).
- **Whether `w` and the 1% seed should be exposed as parameters.** They are fixed here; the
  measurement did not separate `w = 0.01` from `0.1` beyond rounding luck.

## Resolved

- **The draw order and whose generator a split uses** (2026-09-16, `feat/hmm-split-state`
  goal 4). **The number of draws depends only on `S`, in array order, and every trial gets
  its own generator.**

  The split takes a required keyword `rng: numpy.random.Generator`, with no default and no
  module state, as `rand_p_vector` does. It draws, in this order:

  1. `u = rng.uniform(-0.5, 0.5, size=S)`: one value per predecessor `j = 0..S-1`, whether
     or not `T[j, s] > 0`.
  2. For twin `s`, one `rand_p_vector(n_symbols - USER_BASE, 0.1, rng)` per destination
     `j = 0..S`.
  3. The same for twin `p`, over `j = 0..S`.

  Values drawn for dead arcs are discarded. A split therefore consumes exactly
  `S + 2·(S+1)·(n_symbols − USER_BASE)` uniforms. Drawing only for live arcs would make the
  count depend on the topology, so removing one arc anywhere would shift every later draw,
  and two nearly identical models would get unrelated candidates from the same seed.

  The search also takes a `Generator`, and gives each trial its own child tied to the
  trial's identity (round, state, trial number) rather than to execution order. That keeps
  trials reproducible if they run in parallel or are reordered. **`Generator.spawn(n)` alone
  does not guarantee this**: measured on numpy 2.4.6, a child does not depend on how much its
  parent has drawn, but it does depend on how many times the parent has already spawned.
  The search must therefore spawn in a fixed structure before any trial runs (one call per
  round, one child per trial, in canonical (state, trial) order), or build each child from
  `SeedSequence(entropy, spawn_key=(round, state, trial))`. Which one is the search loop's
  choice.
