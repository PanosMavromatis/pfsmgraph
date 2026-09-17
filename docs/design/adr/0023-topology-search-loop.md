# 0023. The topology search walks the original's loop and returns the best model it visits

- **Status:** Accepted
- **Date:** 2026-09-17
- **Source:** none in the PRD — postdates it. Decided on `feat/hmm-search-loop`, revision
  `04-hmm-v0.3.0`, goals 1 and 2 of that branch's plan.

## Context

Revision 04 searches HMM topology by merging and splitting states, scored by total
description length. The scored primitives already exist in `_trials.py`. `_try_split` and
`_try_merge` build one candidate, re-converge it with `baum_welch`, choose its `d` and score
it. `_suggest_split`, `_suggest_merge` and `_suggest_move` run every trial of one round and
rank the results. What was left undecided is the loop that calls them: which moves it
proposes, what it accepts, when it stops, and the parameters the primitives left to it.

**The original has that loop, although this project believed otherwise until this record.**
`HMMLIB-ACCOUNT.md` §11 said there was no headless entry point and that nothing looped over
`suggest-move`. It had been written from the two GUI scripts alone. `Training/hmm-train-new-nw`
and `hmm-train-load-nw`, tracked beside them, run

```lisp
(repeat (val (nth 4 argv))
  (==> trainer suggest-move)
  (==> trainer keep-model))
```

and the view's `Continue for ` button runs the same loop. Its semantics
(`hmm-trainer.lsh:58-100`, `:677-691`, `:952-1073`):

- **The start.** The trainer's constructor converges the new model (`model-starting-size`
  defaults to 1, `init-random 0.001`) under `run-converge`, runs `suggest-d`, and keeps it as
  model 1.
- **A round.** `suggest-move` tries every merge pair, then `*min-split-trials*` = 2 splits
  per state. It keeps the strict-`<` best and restores the incumbent after each trial. The
  winner becomes the model as its trial left it: "No need to run Baum-Welch or suggest-d on
  the new model".
- **Acceptance.** None. `keep-model` commits the winner even when its total exceeds the
  incumbent's.
- **Stopping.** Only the move count `N`.
- **Selection.** Each keep saves a numbered model with `_total_dl` and a training-log line,
  so choosing a model from the trajectory was left to a person.
- **An all-impossible round** leaves `best-size` at 0 and sets the model to size 0.

So this is a **port with named departures**, as ADR 0022 is for the split. The departures
are what the original left to a person: choosing the best model visited and deciding when to
stop. The original also chose `d` by a local search, left a split's EM budget to
`run-converge`'s effective 30-cycle floor, and scored with a numpy forward pass. Each of
those was measured on this branch before being changed.

## Decision

### 1. A round proposes the whole `suggest-move` neighbourhood, and its winner is not re-trained

Each round calls `_suggest_move` on the incumbent: every merge pair except pairs of two
transient states, then `min_split_trials + int(split_trials_per_state_p · state_p[s])`
split trials per state. Every candidate is re-converged inside its own trial. The round's
winner, the first `Move` with a finite total, becomes the incumbent exactly as that
`TrialResult` holds it: the same parameters, `d` and total, with no further EM. The only EM the loop runs
itself is on the starting model.

Pruning merge pairs is the alignment-seeded revision's job (`DEFERRED.md`), not this one's.

### 2. One entropy, keyed by role

The search takes a `numpy.random.SeedSequence`, and every generator it builds is keyed from
that seed's entropy by role, never by `spawn()`:

- the starting model draws from `spawn_key = (0,)`;
- round `r` passes `SeedSequence(entropy, spawn_key=(1, r))` to `_suggest_move`, whose split
  trial `(s, t)` then draws from `(1, r, s, t)`.

Any trial is therefore rebuildable from `(entropy, r, s, t)` alone. This settles the choice
ADR 0022's **Resolved** section left to the search loop.

### 3. The walk keeps the original's unconditional move and returns the best model visited

Each round's winner becomes the incumbent whatever its total, as `keep-model` did. Beside the
incumbent, the search keeps the **best** model visited: a winner replaces it only when
`winner.total_bits < best.total_bits`. The comparison is strict, has no margin, and never
subtracts, so a tie is not an improvement and `+inf` compares `False`. A `Move` with no
result, or with a result that is impossible at every `d` and so scores `+inf`, can never
become the incumbent. The search returns the best, not
the last.

With `patience = 0` this is greedy improvement exactly: round `r` draws from `(1, r)`
whatever came before, so a walk and a greedy run with the same entropy visit the same models
until greedy stops, and the walk's best is never worse than greedy's result.

### 4. The search stops on patience, a round cap, or a dead end

The search stops at whichever comes first:

- **`patience`**: a round that makes `patience + 1` consecutive rounds without a new best, so
  `patience = 0` stops at the first round that does not improve. It is required, since
  nothing has measured a good value;
- **`max_rounds`** rounds, also required, so termination never depends on the data (the
  original's `N`);
- **a dead end**: a round with no move of finite total, because every candidate is
  impossible or the round is empty (at `S = 1` with `min_split_trials = 0`).

`start` is required and is either a `Vocabulary`, over which the one-state model is built
as `hmm-train-new-nw` does, or an `HMMParams` to resume from, as `hmm-train-load-nw` does.
Either way the start is converged under the default rule and scored before round 1.

The search returns a frozen result: the best `TrialResult`, the start's score, one record per
round (its winning `Move`, and whether it set a new best), and which stop fired. Runner-up
candidates are not kept. That record is what the view's history pane showed.

No rollback mechanism exists or is needed. ADR 0017 makes parameters frozen values, so a
rejected candidate is dropped and the incumbent is a reference, where the original's
`reset-model` copied a working copy back.

### 5. A trial chooses `d` by a bounded exact scan, not by Brent's method

`_try_split` and `_try_merge` choose `d` by scanning `d = 1, 2, …` up to 10000. The scan
stops at the first `d` where

    data_bits + int(S) + int(d) + (1 + S)·comb(d, 1 + S) + [d >= S]·S·comb(d, 1 + A)  >=  best

holds, `best` being the lowest total found so far. `data_bits` is the trial's unrounded data
length (ADR 0022 §5), `A` is the number of user symbols, and the right-hand terms are a lower
bound on the model half that is monotone in `d`. The last term applies once `d >= S`,
because every transition row's largest entry is at least `1/S`, so rounding at `1/d` cannot
zero it, which leaves at least `S` live arcs. The scan returns the integer `d` with the
lowest total: the global minimum on `[1, 10000]`, provided the assumption below holds.

**The scan assumes that rounding a converged model never lowers its data length below the
unrounded one.** At `S = 1` this is exact: the one-state likelihood is concave, EM reaches
its global maximum, and no rounded model can beat it. At `S >= 2` it is measured, not proven
(Evidence). If a candidate broke it by `v` bits, the scan could stop early, and its choice
would then be at most `v` bits above the true minimum.

`_suggest_d` stays, unused by the search, as the checked reproduction of `suggest-d`: it
still returns the three tracked models' stored `d` (29, 3, 4). Its pinning test in
`test_mdl.py` is rewritten to assert that the scan beats it on `m001_0001_001`, so the switch
is visible in the tests.

### 6. A split's minimum EM budget defaults to 200 cycles

The search's `split_min_cycles` defaults to 200, and a caller may override it. This sizes the
budget ADR 0022 §4 obliges the search to give a split. Merges keep `merge_min_cycles = 0`,
since a merge does not start at the incumbent's likelihood.

### 7. The total's forward pass takes the search's backend

`_total_description_length`, `_data_description_length` and `_corpus_description_length`
take `backend=`, resolved through `_backends._TABLE["forward_backward"]`. The trials and
the search take it as `score_backend=`, beside the `backend=` that runs EM: they are separate
because `forward_backward` has no `torch` phase, and ADR 0021 forbids substituting one
backend for another. No kernel is written: all four phases already exist and are
held bit-exact to numpy (ADR 0020), so no score changes with the backend.

**Extended by [ADR 0024](0024-search-compiled-work.md) §2** (2026-09-17). There was a fourth
forward pass this section could not reach: `baum_welch` checks convergence by computing the
corpus description length itself, so no `score_backend=` threaded through the trials touched
it, and it ran on numpy under every backend. It now takes a `score_backend=` of its own,
which the trials and the search pass through, so one name governs every forward pass a
search runs.

## Consequences

### Positive

- **The search is reviewable against an original.** Everything but §3's best-so-far and
  §4's stops is the `-nw` loop, and those two automate what a person did with its output.
- **Every trial and every round can be replayed from the seed alone**, including in parallel
  or out of order.
- **The one-state start is scored at its best `d`.** Brent's `d = 29` there is 10.46 bits
  above the global minimum at `d = 13`, and that start is the first model every run compares
  against.
- **The tests can be exact.** The incumbent is the `TrialResult` that won, so "an accepted
  move never raises the best" and "a rejection leaves the best unchanged" are `==`
  assertions, not tolerances.
- **A trial costs about half as much.** Choosing `d` was 53–62% of a trial
  (`merge_round_cost.py`), almost all of it the numpy forward pass, and the Cython pass is
  120–330 times faster on the corpus.

### Negative / costs

- **The search's results are not the original's.** ADR 0022's split already made that true;
  §5's `d` and §6's floor add to it.
- **Splits cost more EM.** At `S = 8` a separated split runs 350–450 cycles, against 40
  under the original's stopping rule alone.
- **§5 rests on an assumption that is measured, not proven, at `S >= 2`.** A violation
  would go unnoticed, although its cost is bounded by its size.
- **The walk can revisit a model.** A split followed by the merge of its twins returns close
  to where it started, and only `patience` or `max_rounds` ends such a cycle.
- **Every round's cost grows with `S²`** through its merge pairs, at about 1.3–1.7 s per pair
  on a 4-vCPU host before §7 (`merge_round_cost.py`).

## Alternatives considered

- **Greedy improvement: accept only a lower total, stop on the first rejection.** Not a
  separate rule. It is §3 at `patience = 0`, and it stops at the first local minimum.
- **The original exactly: unconditional keep for `N` moves, returning the trajectory.**
  Rejected as the default: it leaves the choice of model to the caller and pays for every
  one of `N` rounds. §3 keeps its walk.
- **A margin in bits before a winner counts as an improvement.** Deferred, not rejected: see
  **Open**.
- **Splits only while growing.** Rejected: cheaper early, but a merge could then never win a
  round where the original's would have.
- **A cap on merge pairs per round.** Rejected here: it needs a heuristic nothing in this
  revision measures, and pruning pairs is the alignment seed's job.
- **Polishing the winner with further EM and `d`.** Rejected: its total would then differ
  from the one it won on, and the comparison would have been made against a number the
  incumbent no longer has.
- **`SeedSequence.spawn()` per round.** Rejected: children depend on how many were spawned
  before, so replaying round `r` needs the whole history.
- **Keeping Brent's method for `d`.** Rejected on measurement: on the one-state start it is
  10.46 bits above the minimum. It also used more evaluations than the scan on all three
  tracked models.
- **An unbounded scan of `[1, 10000]`.** Rejected: up to 10000 evaluations per trial where
  the bounded scan needs 7–21.
- **The bounded scan with a safety margin.** Not needed on the evidence: the assumption held
  on every model measured, and a violation's cost is bounded by its own size.
- **Replacing `_suggest_d` with the scan.** Rejected: it would delete the checked
  reproduction of `suggest-d` along with the Brent transliteration, and hide the switch.
- **A split floor of 100 or 400 cycles, or none by default.** 100 separated no split on
  `m008`. 400 changed no measured outcome, since the stopping rule already carried every
  separated run past 350. No default would leave every caller a choice the measurement made.
- **Deferring the compiled forward pass.** Rejected: it is plumbing over existing bit-exact
  kernels, and the numpy pass dominated every trial.

## Evidence

- **The first rounds reproduce the original's.** From the one-state start on
  `set02a_200` (`cython`, seed 20260917, `max_rounds = 3`), the search split state 0, then
  0, then 2, to totals of 2434.01, 2130.66 and 1774.03 bits. `m001_0005_005`'s training log
  records the same three moves at 2433.99, 2131.1 and 1774.03. Only the start differs: 3187.91
  at `d = 13` against the original's 3198.38 at `d = 29` (§5). So the log is a partial
  oracle for the moves chosen, although not for the candidates' bits, since ADR 0022 changed
  the split's draws.

Scripts are tracked under `.scratch/hmm-lush/measurements/` and use the three `set02a_200`
fixtures, with the corpus as one record (N = 1268), on a 4-vCPU Xeon.

- **The loop.** `Training/hmm-train-new-nw`, `hmm-train-load-nw` and
  `hmm-trainer-view.lsh`, read in `feat/hmm-search-loop` goal 1. The view has fifteen
  buttons, not the thirteen §11 listed at the time (§11's list was corrected on
  `feat/hmm-search-log`, 2026-09-17, and now reads fifteen). `m001_0005_005`'s training log is one kept
  trajectory: 3198.38 → 2433.99 → 2131.1 → 1774.03 → 1748.9, four splits, each lower.
- **`d_choice.py`** (§5).
  - `total(d)` has **23 local minima on `[1, 300]` at `S = 1`, and one at `S = 5` and `S = 8`**.
  - Past `d ≈ 300` the data half is flat to 0.03 bits, while the model half keeps rising
    (`m008`: 2097 bits at `d = 300`, 3640 at 10000).
  - The smallest `data(d) − unrounded` over `d ≤ 300` was **+0.0078, +0.0009 and +0.0007
    bits**, so the assumption held on all three.
  - Brent chose `d` = 29, 3, 4 in 20, 28 and 26 evaluations. The bounded scan chose 13, 3, 4
    in **16, 7 and 21**. At `d = 13` the one-state total is 3187.91 bits, against 3198.37
    at 29.
- **`split_em_floor.py`** (§6). Every split trial `(s, t)` of the three models, re-converged
  under floors 0, 30, 60, 100, 200, 400 and 800 with one generator per trial.
  - `m008` (`S = 8`): **floors ≤ 100 separated 0 of 16 trials, and 200 separated 16 of 16.**
    At ≤ 100 EM stopped at 40–110 cycles, 10–27 bits above the 800-cycle data length. At
    200 the stopping rule carried each run on to 350–450 cycles, ending 0.06–2.4 bits from
    the 800 run.
  - `m001_0001_001` and `m001_0005_005`: 60 cycles already matched 800 exactly.
  - Floors 0 and 30 at `S = 5`: split `(2, t)` stopped 9 bits short on data, yet scored
    **3.3 and 13.3 bits lower** in total, so an unseparated split can win a round on
    rounding.
- **`total_forward_cost.py`** (§7). One forward-backward over the corpus: numpy 27.5 ms at
  `S = 1` and 34.2 ms at `S = 8`; Cython 0.11 and 0.23 ms. `cpu_parallel` took 23.6 and
  18.3 ms and `cuda` about 500 ms, the per-timestep overhead `core.md` records at small `S`,
  so Cython is the backend that pays at this size. The scale factors were bit-identical to
  numpy on all four backends.
- **`merge_round_cost.py`** (`feat/hmm-scored-primitives`). Within a merge trial, `_suggest_d`
  took 53–62% and `cython` EM 35–44%.

## Open

- **Totals closer than one step of `d` are ranked by where EM stopped.** Measured on
  `m008` (`split_em_floor.py`, `split_total_steps.py`), one split candidate's total moved by
  up to 72 bits between EM floors of 200, 400 and 800 while its unrounded data length moved
  by at most 2.4. The cause is not Brent: under §5's scan all 16 trials were bit-identical to
  the Brent run. It is the integer `d` itself. As EM moves the parameters slightly, the best
  `d` steps (6 to 5, 5 to 4), and each step changes the model half by 58-71 bits at nine
  states while the rounded data length moves by 2-3. **§3 therefore has no margin, and
  deliberately**: a margin only changes which of two such totals is kept as best, and cannot
  make either less sensitive. On `set02a` the accepted rounds improved by 300-750 bits, far
  above the step. The remedy, if one is needed, belongs to the criterion (PRD section 8) or to
  convergence, not to acceptance.
- **§5's assumption at `S >= 2`**, measured on three trained models and not on the
  candidates a search produces. The implementation's tests check it on constructed
  candidates.
- **The floor needed depends on the model**: 60 cycles on `set02a`'s one- and five-state
  models, 200 on its eight-state one. 200 was not measured on other corpora.
- **A held-out score.** The view has a Test Data DL field that was never wired ("Test Data
  Set: none", always 0). Selecting a model on held-out data is not part of this record.
