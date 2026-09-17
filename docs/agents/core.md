# pfsmgraph

Shared project knowledge for any coding agent working in this repository.

## Current state

**`dataseq` and `hmm` are implemented and released — `dataseq` at 0.1.0 (2026-09-02) and `hmm` at 0.2.0 (2026-09-16, after 0.1.0 on 2026-09-13); the other three members are still empty scaffolding.** In place: the uv workspace root `pyproject.toml` (virtual — no `[project]` table), `uv.lock`, all five `packages/*` members with their own `pyproject.toml`, the `meson.build` files for `align` and `hmm` (`align`'s extension block still dormant, `hmm`'s live since the first `.pyx` landed 2026-09-09), and an empty `pfsmgraph/<pkg>/__init__.py` for the three members that have no code yet (plus `dl/rnn/` and `dl/transformer/`). The ADRs in `docs/design/adr/` are authoritative for the decisions they cover — the twelve initial records from the PRD, plus 0013 (how this family documents its public surfaces) and 0014 (how imported migration source is retained), both added 2026-09-01; the PRD remains the narrative design document.

**What `dataseq` now contains.** Six modules under `packages/pfsmgraph-dataseq/src/pfsmgraph/dataseq/` (the container landed 2026-08-31, the encoder API 2026-09-01) and 74 tests — the first tests in this repository, and 74 of the suite's 3088 today; 2962 are `hmm`'s and the remaining 52 are the repo-root backend-matrix, API-docs, release-runbook and meson-source tests. `_reserved.py` hard-codes the ADR 0011 block as module constants, with no class or parameter that could relocate it; `_vocabulary.py` holds the `Vocabulary` protocol and `SymbolTable`, a frozen first-appearance-ordered implementation that encodes strictly and decodes *totally*, reserved codes included; `_record.py` and `_dataset.py` are the ragged container, whose records carry true lengths and never padding; `_collate.py` is `pad_collate`, where padding is introduced and always returned with its mask. The container imports neither torch nor pandas — verified in a subprocess — and its one runtime dependency is `numpy`.

**What `hmm` now contains.** Fifteen Python modules, two Cython kernels, and 2962 tests. The
fifteenth is `_search.py`, written 2026-09-17 on `feat/hmm-search-loop` under
[ADR 0023](../design/adr/0023-topology-search-loop.md): `_search(records, *, start, seed,
max_rounds, patience, ...)`, **a port of `Training/hmm-train-new-nw`'s headless
`(repeat N (suggest-move) (keep-model))`**, a loop `HMMLIB-ACCOUNT.md` §11 had said did not
exist. `start` is a `Vocabulary`, over which the one-state model is drawn in `init-random`'s
order, or an `HMMParams` to resume from; either is converged and scored first. Each round takes
`_suggest_move`'s first move with a *finite* total as the incumbent whatever its total, as
`keep-model` did, and keeps the best model visited by strict `<` beside it; that and the stops
(`patience` rounds without a new best, a required `max_rounds`, a round with no finite move)
are the named departures. Round `r` is seeded `seed.spawn_key + (1, r)` and the start `+ (0,)`,
so every trial is rebuildable from its identity. It returns a frozen `SearchResult(start, best,
rounds, stop)`. **Its first three rounds on `set02a_200` choose the moves the Lush training log
of `m001_0005_005` records** (split 0, 0, 2, to within a bit of each logged total), which makes
that log a partial oracle for moves, though not for bits. `tests/test_search.py` holds its 37
tests, mostly against a scripted `_suggest_move`, mutation-checked against eight defects. **One
limit is measured and deliberately not engineered around**: near the optimum a candidate's total
is a step function of its converged parameters, because the best integer `d` steps as EM runs on
and each step moves the model half by 58-71 bits at nine states, so rankings closer than that
are decided by where EM stopped; acceptance has no margin because a margin cannot fix it (ADR
0023, Open). The
fourteenth is `_trials.py`, opened 2026-09-17 on `feat/hmm-scored-primitives` with
`_try_split(params, records, state, *, rng, min_cycles, ...)`, the port of `try-split`: split,
re-converge under a minimum EM budget, choose `d` (by `_scan_d` since ADR 0023, by `_suggest_d`
before), and score by *calling* `_total_description_length`, returning a `TrialResult` that carries the unrounded data length
beside the total (ADR 0022 sections 4 and 5). `_try_merge(params, records, first, second, *,
min_cycles=0, ...)` shares the scoring tail, as the two Lush methods share every line but the
surgery, and keeps stationary-mass weights: on every tracked model, trained flat or per record,
no state is transient and occupancy agrees with `state_p` to 0.003, because the corpus's
`begin`/`end` symbols make it one long run of the chain. **A merge can still make a record
impossible**, but only through a transient state's zero-weighted arcs, and then `_try_merge`
raises `ImpossibleSequenceError` rather than scoring `+inf`. **`_suggest_split`, `_suggest_merge` and
`_suggest_move` return every candidate as a `Move`, ranked**, where the original kept a strict-`<`
argmin: candidates are enumerated in the original's order (states then trials, pairs
lexicographically, merges before splits) and `sorted` on the total alone, whose *stability* is the
tie-break -- sorting on `(total, kind)` matches only because `"merge" < "split"`, a coincidence a
test pins against. An impossible candidate is a `Move` with no result at `+inf`, ranked last, so an
all-impossible round is an ordinary list rather than the original's crash; a pair of two transient
states is not a candidate and is filtered by `closed_classes` first. Split trial `(s, t)` draws from
`SeedSequence(entropy, spawn_key=round_key + (s, t))`, so its candidate is rebuildable from its
identity alone. Acceptance is `_search`'s. Every trial also takes `score_backend=` beside `backend=`: it names the `forward_backward` phase of the scan's passes and, since ADR 0024, of `baum_welch`'s convergence check too, and it is separate from `backend=` because `forward_backward` has no `torch` phase and ADR 0021 forbids substituting one. **A merge round
costs its pair count, and scoring dominates each trial** (`merge_round_cost.py`, 2026-09-17,
4-vCPU host, `set11a_dInt`): 0.7 to 200 s for 1 to 120 pairs at `S = 2..16`, a log-log slope of
1.17 against pairs, 1.3 to 1.7 s per pair. Within a trial `_suggest_d` takes 53-62% and `cython`
EM 35-44%, because `_suggest_d`'s Brent evaluations run the numpy forward pass and take no
`backend=`. Pruning pairs, as the alignment seed would, and a compiled forward pass inside the
total are therefore separate savings, and the second roughly halves every trial. The second has
since landed: the description-length functions take `backend=`, and the Cython forward pass is
120-330 times faster than numpy's on the corpus with bit-identical scale factors. Its 36 tests pin the composition, not the parts,
and **their first floor was vacuous**: the case converged in 180 cycles unfloored, so a floor of
20 left a dropped `min_cycles` passing everything; assert that a threshold binds before trusting
a test that passes it through. The
thirteenth is `_topology.py`, opened 2026-09-16 on `feat/hmm-split-state` with
`_split_state(params, state, *, rng)`: the candidate-building half of a topology move, private and
not a transliteration, since [ADR 0022](../design/adr/0022-state-split-initialisation.md) fixes the
Lush `:172` initial-distribution defect and replaces the original's outbound emission redraw with a
per-predecessor inbound perturbation and a 1% emission seed. It neither re-converges nor scores a
candidate; `_trials.py` does. Its random draws follow a fixed
contract whose count depends only on the state count. `tests/test_topology.py` holds 645 tests for it:
properties rather than a differential oracle, since the split departs from the original on purpose, exact
`==` wherever the arithmetic allows, mutation-checked against five defects including the `:172` one. Twelve of them build the case revision 02
could not exhibit, a zero-init state that can emit `begin`: a split does produce it, but through the 1% seed
alone it never makes the old δ-seeding defect win, so only a constructed model whose zero-init state emits
`begin` at 0.9 fails when that fix is reverted. The draw contract is pinned by rebuilding each candidate from an
independent generator in the contract's order, which a same-seed comparison alone would not catch.
**Beside it since 2026-09-17 is `_merge_states(params, first, second)`** (`feat/hmm-merge-states`),
the other topology move, which follows `merge-states` closely and departs in three places: it carries
`init_state_p` where `:262` read `state_p` (ADR 0022 section 1); its outbound weights are stationary
mass that is exactly zero on transient states, where the original weighted by a solve's rounding;
and it refuses a pair of two transient states, from which the original built an all-zero row.
**Weighting by stationary mass makes a merge an exact lumping of the stationary chain**: the merged
state's `state_p` is the pair's sum, and the stationary flow on every arc and symbol adds up. That is
the property its 741 tests lean on, since it checks the inbound, outbound and self-loop blocks
together without restating them, and it caught three of six mutants on its own. Two consequences are
worth knowing before the search loop: a transient but occupied state contributes nothing to a merge
(occupancy weights need records; `try-merge` kept stationary mass on measurement, see `_trials.py` above), and a transient state's split
cannot be undone, because both twins are transient and the merge refuses them. The twelfth is `_mdl.py`, the minimum description length criterion revision 04 scores topology
moves with, opened 2026-09-16 on `feat/hmm-mdl` with its two code-length primitives:
`_int_code_length`, Rissanen's universal prior, and `_comb_code_length`, the cost of a
composition. **Two things the reading settled are not in `HMMLIB-ACCOUNT.md` §8 and are
easy to get wrong from the paper alone.** The original codes `n + 1` rather than `n`
(`util.lsh:467`), which is what makes the code of zero finite — `log2(2.865)` alone —
and matters because the search starts from a one-state model; and the iterated logarithm
runs while the *value* exceeds 1, so it adds every term above **zero**, typically one
more term than §8's "while the term exceeds 1" describes. The generated C
(`util.c:637-671`) settled both, which is the use `core.md` claims for it. The binomial
is never formed — the original already sums `log2(sum+i) - log2(i)` term by term — and
the port reduces the differenced terms with one ascending `np.add.accumulate`, agreeing
with the original's interleaving to 5.7e-14 bits and with an `lgamma` closed form to
4e-11, so it is deliberately **not** claimed bit-identical. Negative input raises where
the original returned `0.0`, because zero is the *cheapest* code and would make an
impossible model win; `m <= 1` still returns `0.0`, which is arithmetic rather than a
guard. The original accumulates in single precision (`flt` locals widened to `real` only
at the return) and this does not, worth 2.1e-4 bits on an 8-state model description
length — two orders below the 0.009 bits the data half already differs from the
original's own logged value. It also owns the data half since the move described below:
`_quantize` (`round-using`, half up, then an ascending renormalisation),
`_corpus_description_length`, and `_data_description_length`, which matches the three
tracked models' logged `data-dl` at their stored `d` to 0.009 bits where the unrounded
value misses by 1.5 to 29 bits. **The model half landed 2026-09-16**, and two of its
choices are decisions rather than transcription. `n-states-r` is `(to-float n-states)`
(`hmm-trainer.lsh:405`) and **not** a rounded quantity, four lines from a
`transition-p-r` where the same suffix means rounded. And the symbol axis is the **user**
symbols, `n_symbols - USER_BASE`: `HMMParams` requires the six ADR 0011 reserved fibres to
be exactly zero, so those codes carry no information, and the count then coincides exactly
with Lush's `:model:alphabet-size` on all three fixtures. Taking `output_p.shape[-1]`
literally adds 12, 25 and **105 bits** to the three tracked models -- the last 24% of
`m008`'s model cost, against a whole training run that moves 58 to 142 bits over four
accepted splits -- and it inflates the per-arc term alone, so it scales with the
transition count and biases the search toward sparsity for a reason that is pure encoding
artifact. `comb-code-length(d, 1 + n_states)` is reproduced although the combinatorics
call for `m = n_states`; which the original did is settled by the oracle, whether it is
right is PRD section 8's question. Transitions are counted **after** quantization, which
is what makes `d` price the topology rather than merely round it. All three logged
`model-dl` values are reproduced to within the log's six-figure print. **The total,
`_total_description_length(params, records, d)`, is the one function the search calls**,
and it returns a bare `float` on purpose: PRD section 8 leaves open whether this project
ends up with a refined one-part code, which has no data/model split, so a return type
carrying those fields would assert the two-part structure inside the seam that exists to
make the swap cheap. A test pins `type(...) is float`; the training log calls the halves
itself. The original's `1e100` sentinel dissolves with no code, since the data half already
returns `+inf` and it absorbs -- and the difference is observable in exactly one place: a
finite sentinel ranks two *impossible* models against each other by the cost of describing
them, where `+inf` ties them. Each model's stored `_total_dl` is a fourth oracle, at four
decimals rather than the log's `%g` and the only one constraining both halves jointly;
reproduced to 0.013 bits, the largest part being the data half's known `d = 29` offset.
**`_suggest_d(params, records)` chooses `d` as the original does, by Brent's method**:
`suggest-d` is `minimize-int` over the total on `[1, 10000]` from 3821, and `minimize`
(`util.lsh:118-227`) is Numerical Recipes' `brent` without `ITMAX`, transliterated statement
for statement because the fixtures pin the probe *path* -- a tighter tolerance or a
different start changes the answer, which is why a library `brent` would not do. It
returns all three stored `d` exactly (29, 3, 4). **It keeps the original's local
minimum on purpose**: the data half is not monotone in `d`, so the total has several
basins, and on the one-state model every search starts from `d = 13` is 10.46 bits
cheaper than the stored 29. The search will not use it: [ADR 0023](../design/adr/0023-topology-search-loop.md)
§5 (2026-09-17) chooses `d` by a bounded exact scan, which stops once the unrounded data
length plus a monotone lower bound on the model half reaches the best total, and found
13, 3 and 4 in 16, 7 and 21 evaluations against Brent's 20, 28 and 26. It rests on
rounding never lowering the data length below the unrounded one, exact at `S = 1` and
measured at `S >= 2`. `_suggest_d` stays as the reproduction, and its pinning test now
asserts both Brent's 29 and the scan's 13, so the switch shows in the tests. **It also qualifies the `+inf` decision above**: Brent
*subtracts* scores, `inf - inf` is `nan`, and a `nan` step quietly collapses to `-tol1`,
which drove a synthetic search to an impossible point and changed the chosen `d` on a
real model with a rare symbol. So `+inf` stays for anything that compares totals, and
the optimizer's objective alone maps it back to the original's `1e100`. `minimize` was
never compiled, so unlike the code-length primitives it ran in doubles.
`tests/test_mdl.py` holds its 121 tests, whose oracle is
arithmetic rather than the original, there being no Lush runtime here to run: powers of
two where the iterated logarithm is exact, `math.comb` where the binomial is still
representable, and an `lgamma` closed form sharing no code with the loop under test. The
eleventh is `_forward_backward_cuda.py`, forward-backward's ADR 0002 phase 4, which completes
its lifecycle the same day and makes `baum_welch(backend="cuda")` public, held byte for byte
on an NVIDIA L4. **NVVM fuses multiply-add by default, and that shaped the kernel**: measured
with numba-cuda 0.30.4, a `c + a*b` kernel returned the exactly rounded `fma` in 20,000 of
20,000 cases, without `fastmath` and still at `opt=False`, and `cuda.jit` exposes no `fma=0`.
So no launch both multiplies and adds a product: products go to device memory in one launch
and are summed in the next, which is about eight launches per timestep. Fusing them back into
one kernel failed 38 of its 55 bit-exact tests, TC-21 among them. `backend="cuda"` takes only
`device=None`, since it runs on numba-cuda's current device and `device=` names a torch one.
The tenth module is `_forward_backward_cpu_parallel.py`, forward-backward's ADR 0002 phase 3,
written the same day: `prange` over each timestep's `(record, state)` cells with every
reduction a serial local of one cell, under the `cpu-parallel` extra, so
`baum_welch(backend="cpu_parallel")` is public too, and held byte for byte at 1 thread and
at 4. **A thread-count comparison is the only check that sees a reassociated sum**: turning
the scale factor's sum into a `prange` reduction failed 17 bit-exact tests at 4 threads and
none at 1, where numba's single chunk folds to the same bits. The
second kernel is `_forward_backward_cython.pyx`, forward-backward's ADR 0002 phase 2,
written 2026-09-15 on `feat/hmm-forward-phases` and registered under both
`forward_backward` and `baum_welch`, so `baum_welch(backend="cython")` is public. **It is
held to the reference byte for byte, never to `torch`'s tolerance**: every sum is the fold
`np.add.accumulate` defines, ξ is `(α·w)·β`, and the extension alone is built with
`-ffp-contract=off`. `tests/test_forward_backward_backends.py` (213 tests) compares on
`tobytes()`, and that comparison is load-bearing: a `-march=native -ffp-contract=fast` build
failed 36 of its tests and a reversed forward sum 29, while all 64 `cython` cases in the
tolerance-based `test_baum_welch_backends.py` passed both. The
ninth is `_baum_welch_torch.py`, `baum_welch`'s torch backend, written 2026-09-14 on
`feat/hmm-torch-backend`: the E-step as reverse-mode gradients, float64 on CPU or on a device the caller names with
`baum_welch(..., device=)` (`feat/hmm-batched-training`, measured on an L4 within the same
tolerance; vectorised over a padded batch with a leaf per record, so one backward pass
yields per-record counts), behind the
`torch` extra, held to the reference within a measured tolerance. **Two properties of it are
fixes, not style, and each crashed `baum_welch` in `HMMParams` before it was made**: the
scale factors are detached, since differentiating through them cancels near-equal terms
and left counts as low as `-3e-19` once EM drove parameters to `1e-50`; and the leaf is the
arc table `w` (counts `w · ∂w`, transitions their sum over symbols), since transition and
emission gradients taken separately round a subnormal unit apart on an arc of `8e-322` and
the re-estimated fibre summed to `1.002`. Random models reach neither case.
`tests/test_baum_welch_backends.py` (469 tests, the kernel half also on `cython`, `cpu_parallel`, `cuda` and `torch-cuda`) pins both, with constructed subnormal arcs and
EM trajectories re-estimated from every state; counts agree within `N·eps·max(1, count)` and
bits within `N·eps·max(1, bits)`, measured worst 1.00 and 0.21, which is the shape ADR 0020's
"a few ulps" should have had. The
eighth is `_backends.py`, the [ADR 0021](../design/adr/0021-runtime-backend-selection.md)
backend table and selection, written 2026-09-14 on `feat/hmm-backend-seam` (see the ADR 0003
paragraph below). The seventh is `_baum_welch.py`, written 2026-09-14 on `feat/hmm-em-loop`: the private
`_m_step(init_counts, transition_counts, emission_counts) -> (init_state_p, transition_p,
output_p, out_counts)`, `run-add`'s stage 5 translated with `safe_divide` and ascending
`accumulate` reductions, and not a DP kernel, so it has no lifecycle phases and the
`dp-compile` gate does not arm on it. **A state with zero outgoing count comes back with an
all-zero row, which `HMMParams` rejects**, and EM constructs one from a valid model in a
single cycle whenever a state is occupied only at the last position; the function returns
`out_counts` so the EM loop can restore that state's previous row *and* fibres, which any
row maximises and which leaves the likelihood bit-identical to the original's zero row.
Beside it is `baum_welch(params, records, *, backend="python", score_backend="python",
batch_size=None, ...)`, the EM loop, public
since 2026-09-14 (the private `_em` until `feat/hmm-torch-backend`), whose `backend=` selects
the E-step, whose `score_backend=` selects the `forward_backward` phase of the convergence
check, and whose result is `BaumWelchResult`. **That check was the last forward pass no
backend could name, and it cost 84% of a search round.** `_corpus_description_length` was
called with no `backend=` at the end of each batch and in `_finish`, so the numpy reference
ran whatever `backend=` said, and `BaumWelchResult`'s docstring stated that as a contract --
"the reference forward pass on every backend", added in `58942fe` as a description and never
argued. Since the four phases are bit-identical the keyword changes no result, which is
exactly why an equality test cannot see it being dropped: the tests pin the phase with a spy
on `_mdl._resolve`, once per check. `"torch"` is refused, because `forward_backward` has no
such phase and ADR 0021 forbids substituting one
([ADR 0024](../design/adr/0024-search-compiled-work.md) section 2,
`feat/hmm-convergence-backend`, 2026-09-17). **Its `log=` takes a text stream**
(`feat/hmm-training-log`), writing a header, a row per convergence check and a stop line,
flushed per line; it is `run-converge` made observable, not the Lush training log, which is
per-topology history for revision 04, and whose `save-training-log` is model persistence
(`DEFERRED.md`). Each backend row names a **batched** kernel,
`_e_step_batch`, fed `pad_collate`'s padded `(B, L)` codes `batch_size` records at a time
(`feat/hmm-batched-training`); **it returns counts per record, never summed**, because
`(c₀ + c₁) + c₂` differs from `c₀ + (c₁ + c₂)` in about a fifth of random triples, and the
loop adds them in record order, so training is bit-identical at every `batch_size`. The numpy
row is bit-exact with the per-record `_e_step` row by row. Its padding mask is **redundant on
any valid model**, since `PAD`'s fibre is zero and padding already weighs 0; the tests pass
raw arrays with `PAD` emittable to make it load-bearing. The loop sums counts over **records** before each M-step, so the
original's flat concatenated stream is the one-record case, and it stops by
`run-converge`'s rule (batches of 10 cycles, 3 in a row moving under 0.1 bits). **That rule
can stop on a symmetric saddle**, since it watches only the size of each batch's change: an
exactly uniform start, which `rand_p_vector(noise_width=0)` returns, is one, so whatever
initialises a model must break symmetry. **`min_cycles=` holds the rule off** (since
2026-09-17, for ADR 0022 section 4's split twins): the loop runs `while unchanged < patience or
cycles < min_cycles`, so the floor takes effect at the next check and a batch that changes
beneath it still resets the count; `0` is the original rule bit for bit. The Lush-trained `m008` fixture, run through it
as one record, moves 0.002 bits and stops after 30 cycles.
**The data description length moved out of this module on 2026-09-16** and now lives in
`_mdl.py` with `_quantize` and `_corpus_description_length`; `_check_codes` went to
`_params.py` at the same time, so `_baum_welch` imports the corpus length from `_mdl`
rather than the reverse. The direction is the point: EM's convergence quantity and the
topology search's data half are now the *same function*, not two that agree. The search's
acceptance rule compares a re-converged candidate against the incumbent, and re-converging
runs `baum_welch`, so two implementations could drift and a candidate would converge under
one definition while being scored under another. That the share is possible at all is owed
to `_corpus_description_length` taking *arrays* rather than an `HMMParams`, which it does
because `_quantize` can round a row to all zeros -- a rounded model need not be a valid
one. `tests/test_baum_welch.py` holds its 89 tests. **`tests/test_hmmlearn_oracle.py` checks the
forward-backward and the E- and M-steps against `hmmlearn`'s `CategoricalHMM`** (21 tests,
2026-09-14), on the destination-only emission case where arc emission reduces to state
emission: hmmlearn's `startprob_` is our `init_state_p @ transition_p`, since its first
state is our `s_1`, and its posteriors are our `gamma[1:]`. Log-likelihood and posteriors
agree to 3e-15 against its `scaling` implementation. **`baum_welch` itself cannot be compared**,
because one M-step estimates a separate emission per arc and so leaves the reduced family;
the EM tests instead run `_corpus_step` and `_m_step` with a dedicated start state (whose
re-estimated row is hmmlearn's `startprob_`) and emission tied over the source, and match
`fit(n_iter=k)` to 1.2e-14 over up to 200 cycles. That leaves `_m_step`'s per-arc emission
division and `baum_welch`'s zero-count restore to `test_baum_welch.py`. The sixth is `_forward_backward.py`, the project's second dynamic-programming kernel, written 2026-09-14 on
`feat/hmm-forward-backward`: the private `_forward_backward(init_state_p, transition_p,
output_p, codes) -> (alpha, beta, scale)` in the scaled probability domain, under
[ADR 0020](../design/adr/0020-scaled-probability-domain-forward-backward.md), not exported.
Beside it are `_description_length`, `_state_posteriors` and `_expected_counts`, the last
returning `run-add`'s three count arrays while building ξ one position at a time rather than
storing `(N, S, S)`. `tests/test_forward_backward.py` holds 108 tests in four layers:
identities that hold for *any* model; **exact enumeration of every state path in
`fractions.Fraction`**, which is what catches a kernel correct for the wrong model (an arc
table transposed everywhere passes every identity about values); constructed cases (an
impossible sequence under `np.errstate(all="raise")`, an exactly uniform model asserted with
`==`); and a bit-for-bit pin against scalar ascending loops. Its reductions are
`np.add.accumulate(...)[-1]`, which is bit-identical to an explicit ascending loop because it
returns every partial sum; do not "simplify" them to `np.sum` or `@`, whose order is theirs
to choose. **Agreement is not permission**: `terms.sum(axis=0)` matches the loop on a
C-ordered array and disagreed in 50 of 50 trials on a Fortran-ordered copy at `S >= 16`, so
it survives a mutation test as an equivalent mutant while depending on memory layout. Of the others, the
fourth module is `_viterbi_cpu_parallel.py`, the ADR 0002 phase-3 backend, landed
2026-09-10, and the fifth is `_viterbi_cuda.py`, the phase-4 CUDA backend, written
2026-09-13 on `feat/hmm-viterbi-cuda` and registered as the fourth backend row. It keeps
phase 3's decomposition but takes every `-log2` on the host with numpy, because device
`log2` (libdevice) differs from numpy's by one ulp in about a quarter of inputs; the kernel
is therefore bit-exact with phase 1. **All four backends are bit-exact with one another on a
given host since 2026-09-14, and phases 2-3 were not before, contrary to their own
docstrings**: they called libm's scalar `log2`, which rounds one ulp away from numpy's
vectorised one in roughly 0.07-0.19% of probability-shaped inputs on an AVX-512 host, most
often on high-probability arcs. `fix/hmm-viterbi-log2` moved their logarithms onto the host
into phase 4's `(S, S, U)` table, so **the contract is numpy's `log2` through `bits`, and
every kernel performs only `+` and `<`**. That is agreement on one host, not identical bits
across machines, since numpy's SIMD dispatch varies. The old suite could not tell the
difference, because an arc-level ulp rarely survives into `total_bits`: 1 generated model in
200 reached it. `_numeric.py` is the numeric
Utility code migrated from the Lush original, landed 2026-09-03 and complete for 0.1.0 at
five functions; `_params.py` is `HMMParams`, the ADR 0017 frozen parameter value, landed
the same day; `_viterbi.py` is the decode, landed 2026-09-04 and **the project's first
dynamic-programming kernel**. Ten names are exported from
`pfsmgraph/hmm/__init__.py` — `HMMParams`, `viterbi`, `ViterbiPath` and
`ImpossibleSequenceError` since 0.1.0, `backends`, `BackendStatus` and
`BackendUnavailableError` since 2026-09-14, when `viterbi` gained a keyword-only `backend=`,
`baum_welch` and `BaumWelchResult` since later that day, and `viterbi_batch` since
`feat/hmm-batched-decode`. **`viterbi_batch(params, records, *, backend, batch_size,
on_impossible)` returns `list[ViterbiPath | None]`, row `i` bit-exact with `viterbi` on
`records[i]`**, over `_viterbi_batch`, a padded `(B, L)` kernel beside `_viterbi` with its
own `_TABLE["viterbi_batch"]` key (all four phases; the Cython batch finishes one record before starting the next, the numba batch runs `prange` over each timestep's `(record, state)` cells, the CUDA batch launches one device thread per such cell per timestep, and all three are held to the numpy batch exactly). It carries δ unchanged at a padded
step and reads the final argmin from column `L` for every row, which is what makes the
carry observable: a padded step taken failed 11 of the 28 tests it had at phase 1, and 12 of 21 on Cython, while ψ written at padded
steps fails none and cannot, since each backtrace starts at its record's own length.
`bits(p)` is `-log2(p)`; `safe_divide(num, den)` yields `0.0` wherever the denominator
is zero, matching the original for `0/0` and `x/0` alike;
`stationary_distribution(transition_p)` is the solve behind the original's `state-p`;
`entropy(p, axis=-1)` is Shannon entropy in bits; and `rand_p_vector(size, noise_width,
rng)` builds a near-uniform random probability vector for parameter initialisation.
**The quantities `bits` builds are
description lengths, not probabilities** (`HMMLIB-ACCOUNT.md` §3): they *grow* as the
probability falls, so **Viterbi over them is a min-sum, not a max-product**, and a port that
reaches for `max` inverts every comparison. This is the second fact after arc-emission most
likely to be lost in translation, and the original's own naming hides it — its `data-p` and
`result-p` hold bits despite a `-p` suffix that means "probability" everywhere else in that
library; nothing on this side carries that suffix. The original's `-1` log-zero sentinel is
**not** reproduced: `bits(0)` is `+inf`, which absorbs under addition and sorts where it
means, so `safe->--log` dissolves into plain `>` and `int-delta` into `np.eye`. Faithfulness
to `-1` was declined as uncheckable — there is no Lush runtime in this repository, and the
sentinel reaches no persisted artifact. One live loose end: `safe_divide` has **no consumer
in 0.1.0**, since all fifteen of its call sites are in the forward pass, the M-step, or the
topology surgery, which arrive in revisions 03 and 04.

**The stationary solve carries the second fact a port loses.** `state_p` is the stationary
distribution of `transition_p`, and `(Pᵀ - I)π = 0` is singular *by construction* — that is
what makes π an eigenvector — so the original replaces row 0 with the normalization `Σπ = 1`
before solving. A port that hands the homogeneous system to a dense solver fails outright,
and `(I - Pᵀ)` has the same null space and needs the same fix. The trick supplies **exactly
one** equation, so it rescues a one-dimensional null space and no more: a **reducible** chain
(two closed communicating classes) has nullity 2 and stays singular, which
`stationary_distribution` reports as a `ValueError` naming the classes. **Since 2026-09-17
that is decided structurally, by `closed_classes`, before any solve**, because the solve's
own singularity turned out not to be a reliable witness: a probability that is not
representable (`[[1,0,0],[0,⅔,⅓],[0,1,0]]`) lifts an exactly singular block off singularity by
an ulp, and `numpy.linalg.solve` then returns one of the stationary distributions silently,
in 18 of 254 random reducible chains. The same labels give transient states an exact `0.0`
in `state_p`, where the solve left rounding up to `1.8e-14`, some of it negative; the state
merge weights by that mass and refuses a pair with none (`feat/hmm-merge-states`, goals 3-5). That matters because revision 04 searches topology by state merge and split, so
a disconnected component is a plausible *search outcome*, and ADR 0017 makes `state_p` a
cached property, so the failure surfaces on an attribute access. The Lush `LU-solve` /
`LU-decomposition` / `LU-back-substitution` trio is replaced by `numpy.linalg.solve` rather
than translated, and that is a deliberate behaviour change: `LU-decomposition` substitutes
`TINY = 1e-20` for a zero pivot and returns a silently perturbed answer where numpy raises.

**`hmm`'s tests read differential fixtures from `.scratch/` in place, and that is the first
test in this repository to read any file at all.** The three saved `.hmm` directories hold
`transition_p` beside `state_p`, so the solve is checked against numbers the original itself
produced. This is not the `.notebooks/`/`.data/` prohibition being bent: those are barred
because their contents exist on one machine only, which is exactly what tracking removes, and
the `.scratch/` rule is about *importing* Python. `tests/` reaches no wheel either — every
member's `[tool.hatch.build.targets.wheel]` packages only `src/pfsmgraph` — so "the fixture is
absent in an installed wheel" is not a scenario here. **The fixtures are four-decimal prints,
and that is a trap worth knowing before revision 03 writes more of these**: it cost two test
failures in unrelated places on first run. A tolerance of `5e-5` is both incomplete (the
*input* is rounded too, and propagates through the solve) and exactly attainable — the 5-state
model's true `π₀` is `0.10135`, printed `"0.1014"`, so the residual is `5.000000000000837e-05`
and an `abs=5e-5` bound fails by less than an ulp. And rounding destroys exact singularity:
the 8-state model's rows sum to `1 ± 1e-4`, lifting the smallest singular value of `(Pᵀ - I)`
from `6.6e-17` to `1.2e-5`, nine orders above `matrix_rank`'s tolerance, so it reports *full*
rank. Renormalise the rows before asserting anything that assumes row-stochasticity — that
restores the hypothesis rather than loosening the conclusion.
The reader now lives in `packages/pfsmgraph-hmm/tests/_lush_fixtures.py`, shared by both test
modules; its `load_params` is what a differential test of anything model-shaped should go
through, because **a saved model is not loadable without the ADR 0011 renumbering**. Lush's
alphabet starts its user symbols at code 2, so the fixtures' `(S, S, 6)` `output_p` becomes
`(S, S, 12)` here, placed at `[..., USER_BASE:]`; every derived quantity is invariant under
that, since `state_p` never reads `output_p` and zero-padding a symbol axis adds only
`0·log2(1) = 0` terms to an entropy. A saved model's own `_alphabet` holds Lush pointer
addresses (`#$11F50E0`) rather than names, **but the names are not lost — they are in the
corpus**, at `set02a_200.sds/_alphabet`, which reads `0→begin, 1→end, 2→c, 3→d, 4→b, 5→a`.
*(Corrected 2026-09-03. The earlier wording, "the symbol names are gone either way", was
written from the model directory alone and is true only of it.)* The sharpened claim is the
better evidence for the same conclusion: persistence split the mapping from the model, so
the model alone stops denoting and has to be rejoined to the corpus that trained it, which
is precisely ADR 0001's cost clause and `DEFERRED.md`'s serialization trigger.

**The port has a decode oracle, and the Viterbi kernel was validated against it before it
was written.** `save-viterbi-path` (`hmm-trainer.lsh:712-727`) wrote a `<model>.vpath.xls`
beside each saved model — one `Output / States / Entropy` row per position, plus a leading
`-` row for the state before the first symbol, which is ADR 0015's N+1 geometry printed to
disk. All three tracked models have one, and they were tracked on 2026-09-03 together with
the corpus. Concatenating the corpus's 200 `.seq` files reproduces each file's `Output`
column exactly, and the `HMMLIB-ACCOUNT.md` §3 min-sum recurrence reproduces its `States`
column at **1269/1269 positions on two of the three models**. So a differential test of the
decode does not have to be invented, and — more to the point — **this branch cannot
accidentally validate its kernel against itself**, which was the standing risk in porting a
defect. The `.gitignore` widening that made this possible qualified the same
`Training/<all others>` note a second time: "output of the algorithm being translated" is
not a reason to decline a file when its inputs are tracked beside it, because that pairing
is what a differential test is made of.

**The δ-seeding defect is fixed, not reproduced, and the divergence is exactly one
position.** `HMMLIB-ACCOUNT.md` §7 records that `update-viterbi-path` seeds δ with raw
`init-state-p` into a bit-domain accumulator; here `delta[0] = bits(init_state_p)`. Measured
against the three oracles: the corrected seed agrees at 1269/1269 on two models and
1268/1269 on `m008_0001_008`, differing only at position 0, where the original prefers
state 0 (`init_p` 0.3665) to state 5 (0.6335) — the two best outgoing arcs differ by 0.004
bits, so the seed alone decides it and the original decides it backwards. **The defect
reaches nothing downstream**, which is what makes the fix cheap: `run-add` is genuine
Baum-Welch over α/β/ξ, the MDL score comes from `update-data-dl` (a clone of the *forward*
pass), and `path-states`/`path-entropy` are read only by `save-viterbi-path` and
`seq-state`'s printer — so the decode is annotation-only and neither revision 03's training
nor revision 04's search moves. **The degenerate half of the defect is masked by the learned
topology, and that is not a guarantee**: every state with `init_p == 0` in these models also
cannot emit `begin` on any outgoing arc (0 of 4 reachable in `m001_0005_005`, 0 of 6 in
`m008_0001_008`), so `+inf` absorbs before the δ = 0.0 seed can win. Revision 04's
`split-state` halves initial probabilities without halving a topology, which decouples them
— so the case the fixtures cannot exhibit is the one the next revision constructs. Note the
contrast with the `-1` sentinel above, which was declined as *uncheckable*: that reasoning
was sound when written and would have needed re-examining had the `.vpath.xls` files been
found a revision earlier. §7's other defect, `psi` as a float matrix round-tripping state
indices, is likewise not reproduced — `psi` is `np.int64`.

**The decode landed 2026-09-04, and three of its properties are decisions rather than
mechanics.** `_viterbi(init_p, transition_p, output_p, codes)` is the private kernel and
`viterbi(params, record) -> ViterbiPath` the checked wrapper. **The kernel neither validates
nor raises**: an impossible sequence comes back as `total_bits == inf` and the *wrapper*
turns it into `ImpossibleSequenceError`, because every later ADR 0002 phase implements the
same signature and a CUDA device function cannot raise a Python exception — keeping the
kernel purely numeric is what leaves phases 2–4 a transliteration. `ImpossibleSequenceError`
subclasses `ValueError` so `except ValueError` still works, and exists as a distinct type
because revision 04's topology search decodes many sequences against many candidate
topologies, where an impossible one is an ordinary *search outcome* rather than a malformed
input. **`ViterbiPath` carries no per-position entropies** even though the original has a
`path-entropy` slot and the oracle prints the column: it is
`params.state_entropies[path.states]`, so ADR 0017's "computed, never stored" applies — and
deriving it in the test turns the oracle's third column into a fourth independent check.
The `(S, S, A)` precompute goal 2's probe used was **refused** in the kernel: it is
`O(S²·A)`, and it *reads* like the hoisted emission factor ADR 0015 forbids while not being
one.

**The fixtures cannot exercise the cases the next revision constructs, and that has now
happened twice.** Mutation-testing the decode found that reversing the tie-break breaks
nothing — there are **0 exact ties in 3804 positions**, because learned float parameters do
not collide — so the differential test alone would accept a last-wins port. The property
still matters: `rand_p_vector(size, noise_width=0)` returns an exactly uniform vector, which
ties at *every* position, and that is how revision 03 initialises. Same shape as the
δ-seeding degenerate case above, which the learned topology masks and revision 04's
`split-state` unmasks. **Construct those cases in tests rather than waiting for them**; a
green differential suite is evidence about the corpus, not about the algorithm. The converse
holds too and is worth not over-correcting: making `psi` float64 again breaks no test
either, and *should not*, since every index below 2²⁴ is exactly representable — which is
precisely why the master plan judged that defect harmless.

**`entropy` deliberately does not reuse `bits`, and reuse would be a bug.** Entropy is
`Σ p·bits(p)`, so the refactor looks obvious; but `bits(0)` is `+inf`, which is *correct* for
a description length — an impossible event costs infinitely many bits — and *wrong* for an
entropy term, where the zero is a weight as well as an argument and `0·inf` is `nan` rather
than the 0 the `0 log 0 = 0` convention requires. The two functions disagree about zero
because they ask different questions of it. `entropy` substitutes `1.0` at the zeros before
the logarithm (`log2(1) = 0`), masking on `!= 0` rather than `> 0` so a *negative* input
still reaches `log2` and still goes `nan` loudly, matching `bits`. Its `axis` defaults to the
last, so the `(S, A)` marginal
`p_i(k) = Σⱼ transition_p[i,j]·output_p[i,j,k]` — `np.einsum("ij,ijk->ik", …)`, and
model-shaped, so it belongs with `HMMParams` rather than here — yields `S` entropies in one
call.

**`rand_p_vector` takes a required `numpy.random.Generator`**, with no default and no module
state, so reproducibility is structural rather than merely available. ADR 0017 makes
parameters a frozen *value*, which is hollow if the value cannot be re-derived, and ADR 0002
commits to `prange` and CUDA phases where a shared generator is a data race rather than a
style choice. It takes a `size` and returns a new array because the original *assigns* rather
than perturbs, so an out-parameter would carry no information. Two guards the original lacks:
`noise_width` must be in `[0, 1)`, since at 1 or above an element can go negative and
normalizing still yields a vector summing to 1 — undetectable downstream; and `size` is
type-checked before it is compared, because passing the array to fill is the natural porting
mistake and `size < 1` on an array raises numpy's "truth value is ambiguous".

**`HMMParams`'s symbol axis spans the whole vocabulary, and the six reserved fibres are
required to be exactly zero.** `output_p` is `(S, S, vocab.size)`, so a code indexes it
directly — `output_p[i, j, codes[t]]`, no offset anywhere, which is what leaves ADR 0002's
phases 2–4 with no index arithmetic to port. The alternative, sizing the axis to the user
symbols alone and subtracting `USER_BASE`, **fails silently**: `encode(...,
on_unknown="unk")` is a documented `dataseq` path that puts `UNK` (code 1) into a record,
and `1 - USER_BASE` is `-5`, a negative index numpy accepts without complaint, so the
decode returns a confident path built from some other symbol's emission probabilities.
Sized to the whole vocabulary the same record reaches a zero, `bits(0)` is `+inf`, and the
path is reported impossible instead of wrong — emission of a reserved symbol becomes
impossible by *arithmetic* rather than by convention. The cost is `6·S²` dead entries,
120 KB at `S = 50`, scaling with the state count rather than with the corpus.
**Two validation rules there are decisions, not mechanics.** A zero `transition_p` row is
rejected with no exemption, because `merge-states` divides with `safe-/` and so revision 04
can construct one (`HMMLIB-ACCOUNT.md` §5) — it should learn that at construction, where it
has to decide what an unreachable state means, not downstream in a stationary solve that has
no answer. An emission fibre on a **dead** arc is conversely not checked at all, since
`bits(0)` on the transition absorbs the path; that exemption is load-bearing rather than
theoretical, because the original's own saved models are full of all-zero fibres on
zero-probability arcs and a blanket rule rejects the fixtures outright. `SUM_TOL` is `1e-5`,
and it is the **lower** bound that binds: `float32` eps is `1.19e-7`, so a vector normalised
in float32 drifts past `1e-6` over a symbol axis of a few dozen, and ADR 0017's own Negative
section anticipates exactly that consumer in revision 03's `torch` backend. Finally, the
**cached** arrays are frozen too — a `cached_property` returns the same object every time, so
freezing only the inputs would leave `state_p` writable and reintroduce, one level out, the
stored-and-stale failure ADR 0017 claims becomes unrepresentable.

**`hmm` declares `numpy>=2.1` where `dataseq` declares `>=1.24`, and the divergence is
deliberate** — it tracks the pure/compiled split, not drift. `hmm` and `align` are the
meson-python members and both already spell `numpy>=2.1` in their commented-out
`build-system.requires`; a C extension built against numpy 2.x headers will not run on 1.x,
so the runtime floor follows the build floor. Reviewed 2026-09-03 against the first numpy
code in the package: `_numeric.py` uses nothing newer than numpy 1.20 (`broadcast_shapes` is
the newest name), so the bound is justified by the compiled future rather than today's API —
do not "correct" it downward on that basis, since it would only have to be undone at the
first `.pyx` and raising a published lower bound is the breaking direction. Editing it left
`uv.lock` byte-identical, confirming the workspace footgun below.

**The ADR 0003 reporting mechanism lives at the repo root, and the table it reports on
lives in the package.** `conftest.py` carries `pytest_report_header` and nothing else;
`_backends.py` beside it is **policy only** since 2026-09-14 — the header, the escalation
rule and `PFSMGRAPH_REQUIRE_BACKENDS` — and reads each distribution's table (`TABLES`, today
`pfsmgraph.hmm._backends`) through that module's private `_status`, the same cached probe
behind the public `hmm.backends()`. So the header and a user's call report one reason for one
absence; `tests/test_backends.py` asserts they match. **The conftest must stay at the rootdir** —
`pytest_report_header` is a *startup* hook while conftest files under `packages/*/tests/`
are loaded during collection, so a hook sited there is registered too late and discarded
with no warning at all (measured on pytest 9.1.1: of two conftests each defining it, only
the root one printed). `tests/test_backends.py` asserts the placement precisely because
that failure is silent. Fixtures have no such constraint, which is why the `backend` fixture
lives in `packages/pfsmgraph-hmm/tests/conftest.py`. **The matrix stopped being empty on
2026-09-04**, when `hmm/_viterbi.py` reached ADR 0002 phase 1, and gained Viterbi's phases 2,
3 and 4 on 2026-09-09, -10 and -13; `forward_backward`'s phase 1 joined the header with the
move. A run opens with `backends: viterbi python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓ |
viterbi_batch python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓ | forward_backward python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓ | baum_welch python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓ · torch ✓` on a machine with a GPU and `… · cuda ✗ (no CUDA device
detected) | …` without one. **The `cuda` probe can fail only because the module makes it**:
`@cuda.jit` decorates lazily, so `_viterbi_cuda` raises `ImportError` at import when
numba-cuda reports no device — otherwise the import would succeed on a GPU-less machine and
report a backend that fails on first call. `tests/test_backends.py` derives that row's
expectation from numba-cuda directly, never from the module under test. Each row names the
*kernel module*, not the package, because `import pfsmgraph.hmm` succeeds with or without a
decode in it and a probe that cannot fail is not a probe.

**One field, `needs`, lets one table serve two policies** ([ADR 0021](../design/adr/0021-runtime-backend-selection.md)
section 3). Each `_Row` records what its absence is attributable to: nothing (`python`), the
compiled extension (`cython`), numba (`cpu_parallel`) or a CUDA device (`cuda`). The root
policy escalates the first two (`ESCALATED_NEEDS`) — nothing external is needed to run pure
Python, and a committed `.pyx` whose extension is missing is a missing or stale build — and
reports the other two as skips. **The public API treats the second differently, and
correctly**: an install may be a pure wheel, so there `backend="cython"` raises
`BackendUnavailableError` with the pure-wheel remedy. The field has had three names:
`hardware` until 2026-09-10, when phase 3 showed an absence can be legitimate without being a
*device*, then `optional_on` on the root `Backend` row, and `needs` since the table moved.
The package's probe also distinguishes *which* import failed — the extension itself absent is
a pure install, any other `ImportError` from inside it is reported as itself — and
`test_backend_selection.py` checks that with a synthetic stale module. `EMPTY_HEADER` stays
under test — the branch is still live and ADR 0003 requires that an empty matrix say so in as
many words. `dp-compile.toml`'s `[backends] registry` names the package's table; its rows are
`_Row` entries under `_TABLE`, not the phase skills' flat `Backend` template.

**The API documentation lives in a repo-level `docs/api/`, and that layout is now binding on all five members.** [ADR 0013](../design/adr/0013-api-documentation-layout-and-tooling.md) settles it: one subdirectory per distribution (`docs/api/dataseq/`, and `docs/api/hmm/` since 2026-09-13 — a member gets its subdirectory when it gets code; `hmm`'s `baum_welch.md` documents training, the tolerance a `torch` result is held to, and since 2026-09-17 the `score_backend=` that names the convergence check's phase, and its `backends.md` documents runtime backend selection with examples chosen to print the same output on every machine, since which backends can run differs by host), hand-written Markdown rather than a generator, no build step and no addition to the `dev` group. Two rules divide the labour and are the reason the choice is sustainable: **docstrings are normative for signatures**, since that is where an editor and `help()` look, while **`docs/api/` is normative for contracts** — the invariants, the reasons behind them, and the seams between distributions, which are contracts even when stated nowhere else. And **every code block is executed and its output pasted from the run**, error messages and tracebacks included; it is the only guard against drift that a hand-written layout has. Sphinx is deferred rather than refused (the docstrings already speak reST, so the migration is mostly configuration), and mkdocstrings is refused outright, because its Google/NumPy style expectation would force rewriting all six modules' docstrings to satisfy a tool.

**The encoder API is settled (2026-09-01), and three parts of it are contracts rather than choices.** `SymbolTable(symbols)` builds a frozen, first-appearance-ordered table, with `from_sequences(sequences)` for a corpus; the name is deliberate, since `Alphabet` implies single characters and this family's symbols are words. Encoding is strict by default and the opt-in is spelled per call — `encode(symbols, on_unknown="raise" | "unk")` — so one mapping serves curated training data and uncurated inference without needing two tables; the value is validated *before* the loop, so a misspelled policy cannot behave as `"raise"` until the first unseen symbol shows up in production. Decoding is total over `range(size)`, reserved codes included, because a padded batch is the array most likely to be decoded. And the symbol→code mapping is **public**: `code(symbol)` plus a `sym_to_code` property returning a `MappingProxyType` — a live read-only view, not a copy — because `pfsmgraph-align` builds an `(size, size)` scoring matrix from the whole mapping at construction and must not reach into a private attribute across a distribution boundary. Persistence and a frequency reordering were deliberately left out; see `docs/plan/DEFERRED.md` for both triggers.

**All five members are on meson-python as of 2026-09-04, and three of them contain no compiled code and never will.** The history is kept because the conclusion is counter-intuitive and the reasoning that produced it was wrong twice. meson-python's editable-install import hook injects a `sys.meta_path` finder that claims the entire `pfsmgraph` PEP 420 namespace and shadows any distribution left on a plain `.pth`, so while `align`/`hmm` alone were on meson-python, `import pfsmgraph.dataseq` (and `hseg`, `dl`) failed after `uv sync`. Neither package has compiled code yet — the `meson.build` extension blocks are dormant `if fs.exists()` guards — so the switch to meson-python was deferred to when the first `.pyx` lands, at which point the namespace/editable interaction must be solved (non-editable install of the compiled members, a single combined compiled distribution, or an upstream fix). **The second of those is refuted**, measured on `exp/meson-python-namespace` 2026-09-04: two meson-python finders *chain* rather than conflict — a finder that does not recognise a submodule returns `None` and the import falls through — so one finder is not the problem and any finder is, and a combined compiled distribution would still shadow the three pure-Python members. The boundary is meson-python versus plain `.pth`, which makes a fourth option visible that ADR 0012 does not list: put all five members on meson-python, so every one has a finder. Those revert recipes are gone now, spent; every member carries a `meson.build` instead. This qualifies the PRD §6.1 note, whose "namespace is fine" evidence was gathered with *setuptools* editable, which composes; meson-python's finder does not. **That deferral's own premise is falsified, and the resolution is scheduled *before* the first `.pyx` rather than with it** (decided 2026-09-04). ADR 0012 holds that choosing between the candidates without a compiled kernel "would be guessing"; but the finder is injected by the *editable install*, not by compilation, so it appears identically with the extension blocks dormant — which is how candidate 2 was refuted and the fourth option found. The ADR names the right mechanism in its Context and then reasons from the wrong one in its Alternatives, treating the deferred cost as compilation. What a real `.pyx` still adds — whether rebuild-on-import works, whether a stale `.so` can appear, how much dev-loop friction each option costs — bears on the *quality* of the surviving candidates, not on which are viable. **The choice is made: all five members on meson-python** (2026-09-04). Both surviving candidates were measured and both work — the compiled members installed non-editable (no finder exists at all, suite green at 271) and all five on meson-python (five finders chaining, suite green at 280). It turned on *how each fails*, not on elegance. Non-editable is the cleaner model and retires the baked-`ninja` footgun entirely, but `editable = false` must be repeated at every `[tool.uv.sources]` declaration site — a member-level declaration beats the workspace root's — and omitting it at any one site resurrects a finder that breaks **all five** members; separately, without `[tool.uv] cache-keys` a source edit is served stale with no error, since uv's default cache key for a local path is its `pyproject.toml` rather than its sources. Both failures are silent. Under all-five-on-meson-python the characteristic failure is a module missing from `install_sources`, which is a `ModuleNotFoundError` at import and is already caught by `tests/test_meson_sources.py` — that test went 7 → 16 by itself when the three new `meson.build` files appeared, which is where 271 → 280 comes from. The dev loop agrees and the gap widens with compiled code: editing a `.py` is visible with **no sync at all** under an editable finder, against a rebuild-and-reinstall for a non-editable copy that becomes a full recompile at the first `.pyx`. Note this inverts ADR 0012's implicit cost model, which treated non-editable installation as the cheap fallback and rebuild-on-import as the luxury being surrendered. Two consequences for the superseding ADR: it overrides ADR 0008's per-package build backends, and no wheel this project has published was built by meson-python -- `pfsmgraph-dataseq` 0.1.0, the only real release so far, went out from hatchling. (Until 2026-09-13, when `pfsmgraph-hmm` 0.1.0 became the first meson-python release.) The four-file release invariant must therefore be re-verified against an actual built wheel, and **that lands on `pfsmgraph-hmm` 0.1.0**, the next release the master plan schedules; `dataseq` has no next release scheduled, so the burden is not waiting on it. **Re-measured 2026-09-13 on `chore/release-hmm-0.1.0`**, and the findings inverted: the meson sdist is a byte-identical `git archive` of `HEAD` (so `just build` cannot see uncommitted work), while the wheel is reproducible only with `SOURCE_DATE_EPOCH`, which the `build` recipe now sets; the four files were verified in a clean venv. `docs/ops/release.md` carries the detail. **Landed 2026-09-04.** Verified from a deleted venv and a plain `uv sync` — not `--reinstall`, which reuses an environment that already has `ninja` on disk and so cannot test the question: uv reported "Prepared 5 packages without build isolation", all seven import paths resolve, five finders sit on `sys.meta_path`, and the suite is green at 280. `pfsmgraph.__path__` is still a single synthetic loader entry, which is the point stated positively — the fix is not to stop the finder replacing `__path__`, it is to leave no member relying on `__path__`. Two things the landing turned up that the evaluation had not: the scope is **five members, not two**, so `dataseq`/`hseg`/`dl` needed comments written from scratch where `align`/`hmm` had a recipe to follow; and the `dev` group needs **`numpy`** as well as `meson-python`/`cython`/`ninja`, because it is a *build* requirement of `align`/`hmm` that `build-system.requires` cannot supply once isolation is off. **Recorded as [ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md)** on the same day, which supersedes 0012 and overrides 0008; that record, not this paragraph, is authoritative, and it is where the two rejected alternatives are argued rather than merely named. 0008 and 0012 are the only `Superseded` records in the directory — the sequence 0008 → 0012 → 0018 is the one worked example here of a decision surviving two revisions of its own evidence.

**`.scratch/` holds imported source that is not ours, together with our own writing about
it.** It is where the three existing `dataseq` implementations were read side by side before
the merge into `packages/pfsmgraph-dataseq/`, together with the proof-of-concept
alignment library whose `Alphabet` is the *encoder* ancestor. That distinction is why ADR
0010 still says "three implementations" while also requiring the `Alphabet` reconciliation:
four imported sources, three of them containers. So the scaffolding note above is a
claim about the three members that have no code yet, and the test count is a claim about
`packages/`: `.scratch/` does contain Python, Cython and `tests/` directories
belonging to other projects, and now also Python of our own — a runnable transliteration of
the Lush original under `.scratch/hmm-lush/translation/`, written as a reading aid for the
merge, and since 2026-09-14 the measurement scripts behind ADR 0020 under
`.scratch/hmm-lush/measurements/`, joined 2026-09-15 by `viterbi_batch_speed.py`, the per-phase batch timings, which unlike the ADR 0020 scripts imports the public `pfsmgraph.hmm`. `param_representation_move_cost.py` followed on 2026-09-16, splitting a topology move's cost into building the candidate and scoring it; building is about 0.005% of the total, which is what settles the parameter representation's dense options on cost. `split_symmetry_breaking.py` joined the same day on `feat/hmm-split-state`, comparing how a split state's twins are told apart: the original's outbound redraw against a per-predecessor inbound perturbation. It is the evidence for [ADR 0022](../design/adr/0022-state-split-initialisation.md). `merge_round_cost.py` joined on 2026-09-17 on `feat/hmm-scored-primitives`, timing a `suggest-merge` round and its phases against model size. Three more joined the same day on `feat/hmm-search-loop` as ADR 0023's evidence: `d_choice.py` (the shape of `total(d)` and Brent against the bounded scan), `split_em_floor.py` (how many EM cycles a split needs before its score settles) and `total_forward_cost.py` (the forward pass inside the total, per backend), followed by `split_total_steps.py`, which found why a candidate's total moves far more than its data length with EM's budget. `search_round_profile.py` joined on `feat/hmm-convergence-backend` as ADR 0024's evidence: one `_suggest_move` round before and after the convergence check took `score_backend`, 9.38 against 1.70 s, with the numpy check rebuilt by a patch so both run in one process. `torch_search_path.py` joined it the same day and answered 0024's other Open item: with `score_backend="cython"` on both sides, a torch search does **not** leave the serial path on the tracked fixtures -- all 66 candidates of three rounds and the 3-round search from the one-state start chose the same `d`, took the same EM cycles and scored bit-identical totals, because the criterion quantizes at `d` and absorbs the 1e-11 bits their unrounded data lengths differ by -- while costing 350-400 times more on the CPU, since `_search` takes no `device=` and so reaches numba-cuda but never a torch device. Those are tracked so a decision record's evidence can
be re-run rather than only read. They sit outside `tests/` because they measure the
alternatives the record rejected.

**It is retained across branches, and that changed on 2026-08-31.**
[ADR 0014](../design/adr/0014-scratch-retention-and-per-package-scoping.md) is
authoritative for this. `.scratch/` was created as a temporary `dataseq` working area to be
deleted by the last goal of the `feat/dataseq-merge` plan; it is not, because the same
imports are the migration source for `hmm` and `align` 0.1.0, and `hmm-lush` is under no
version control anywhere else. What is re-scoped per package is not the contents but the
**`.gitignore` policies**: each import's rules surface the files relevant to the package
being migrated, so the tracked set follows the work. `.scratch/align-poc/.gitignore` is the
only phased policy of the four — `dataseq` done, `hmm` **active since 2026-09-01** and empty
by design, `align` inert but for a single file advanced early — and advancing it is an
uncomment rather than a re-derivation. That advance added no files, which is recorded as the
finding rather than left looking like an oversight: nothing in tokalign is HMM-related. The
`align` phase took its one exception on 2026-09-01: `_python.py` is tracked ahead of that
migration because the reserved-block renumbering had to show that the alignment code
hard-codes no gap index, and a negative finding needs a signature to be checkable.
`_cython.pyx` stays untracked so `DEFERRED.md`'s "the first `.pyx` lands" trigger keeps one
meaning — a `.pyx` under `.scratch/` belongs to no distribution and must not appear to fire
it. The other three state their
forward judgement in their own headers, and `hmm-lush`'s was **wrong in an instructive
way**. It read "already scoped to the whole live HMM library, needs no widening when the
`hmm` branch opens"; the `hmm` numeric migration widened it twice on 2026-09-03, and both
additions came from outside the library. `Code/Utility/C/util.c` — the DH compiler's
generated C — is tracked because with no Lush runtime anywhere in this repository it is the
only *machine-checked* statement of what a primitive means, and it settled two questions a
reading could only have guessed at. Three saved `.hmm` model directories are tracked as
differential-test fixtures, because each holds the **inputs and the outputs** of a
computation being ported. So the judgement was right about the library and wrong about the
boundary: a migration needs not only the code it translates but whatever makes the
translation checkable, and that is rarely in the same directory. `dl` and `py-rudimentary`
remain spent imports whose tracked sets are final.

**The tracked set can only widen, never narrow.** `.gitignore` is consulted only for files
git does not already track, so an ignore rule added over a tracked path is silently inert —
narrowing a policy would take a `git rm --cached`, which is a deletion commit and re-opens
the squash-merge hazard that retention stands down. Track deliberately: the cheap direction
is available only once, at import. The matching failure is that a file written under
`.scratch/` *without* a negation is invisible to `git status` rather than an error, so the
work looks committed and is not; every policy carries a `!/*.md` negation for our own
writing because of it. `git check-ignore -v <path>` names the rule that matched.

**Four imported directories, six source trees**: `.scratch/py-rudimentary/` holds two
repositories — `segalign/` (the implementation) and `SegAlign-Draft/` (the predecessor it was
refactored from, tracked at one file because what it contributes is the *absence* of a
sequence abstraction) — and `.scratch/align-poc/tokalign/` carries two more nested inside it.
`.scratch/align-poc/` is the proof-of-concept alignment library that PRD §1.2 describes and
that ADRs 0001–0004 derive from, so unlike the other three it is a *source* of this project's
invariants rather than only evidence about them. Nothing there is part of any distribution
and nothing outside it may import from it. The leading dot is
load-bearing — it matches pytest's default `norecursedirs` entry `.*`, which is why `uv run
pytest` still collects zero items with those files present; and the directory sits outside
`packages/`, so the workspace glob never claims it. `.scratch/README.md` states the rest,
including why an imported repository's `.git` must be renamed before its contents can be
committed here.

**`.notebooks/` and `.data/` are a local workbench, and are not migration source.** Added
2026-09-01, they exist so a new feature can be exercised by hand against real data before
release -- `uv sync` installs all five members editable, so `pfsmgraph.dataseq` resolves to
`packages/*/src/` and there is nothing to publish or reinstall. Each owns a deny-by-default
`.gitignore` negating only itself and a `README.md`, so *everything else written there is
ignored*; the two tracked files are what make the directories survive a clone, since git
tracks no empty directory. That is also why the policies are not root `.gitignore` entries:
git never descends into an ignored directory, so a nested policy there would be dead text.
They take the leading dot for the same reason `.scratch/` does -- `.*` is pytest's default
`norecursedirs`, so a scratch `test_*.py` cannot join the suite (verified 2026-09-01, when the suite stood at 74: it passed with one
on disk) -- and sit outside `packages/`, so the workspace glob never claims them. **Nothing
under `packages/` may import or read from either**: their contents exist on one machine only,
so a distribution reaching in passes locally and fails in every clone. Unlike `.scratch/`,
neither is evidence about anything; do not add negations to commit data through them.

Still to do, in PRD order (§11): `hmm` (Lush translation), then `align`, then `hseg`. **The `hmm` migration is planned as three revisions rather than one**, because the Lush trainer spans three problems that fail differently: `02-hmm-v0.1.0` (Viterbi and the `dataseq` interface, carrying the project's first DP kernel, first `.pyx`, first non-empty ADR 0003 backend matrix; the meson-python resolution ADR 0012 deferred to it was in the event settled *ahead* of it, on `exp/meson-python-namespace`, since the finder turned out to be injected by the editable install rather than by compilation — [ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md)), `03-hmm-v0.2.0` (Baum-Welch on a fixed topology, with an optional `torch` autograd backend held against the numpy reference), and `04-hmm-v0.3.0` (topology search by state merge and split, scored by minimum description length). **All three are now open**, and `docs/plan/planned/` is empty in consequence: 02 and 03 are closed, their subgoals and the branch plans that executed them filed under `docs/plan/02-hmm-v0.1.0/` and `docs/plan/03-hmm-v0.2.0/`, and 04 was opened on 2026-09-16 with its subgoals in the master plan. A revision drafted ahead of its opening still goes under `planned/` as a splice source; the convention is stated in `docs/plan/TODO.md` rather than in a directory git does not track. `dataseq` is finished and released: 0.1.0 is on PyPI and tagged `pfsmgraph-dataseq-v0.1.0`, as of 2026-09-02. `hmm` 0.1.0 followed on 2026-09-13, tagged `pfsmgraph-hmm-v0.1.0`: the first release built by meson-python, published as a pure `py3-none-any` wheel. **`hmm` 0.2.0 followed on 2026-09-16, tagged `pfsmgraph-hmm-v0.2.0`, and is this family's first release of platform wheels** — twenty of them, over linux x86_64/aarch64, macOS arm64 and Windows x86_64 on cp310–cp314, built by GitHub Actions and published through PyPI Trusted Publishing with PEP 740 attestations. It is the first version whose public API reaches a compiled kernel, which is why the pure wheel could not continue: `backend="cython"` is public, so a pure install would report `cython ✗` to every pip user. The sdist reproduces byte for byte across machines — the archive this repository builds from the tagged commit has the same sha256 as the one CI built and PyPI serves, with no `SOURCE_DATE_EPOCH` set by the sdist job at all. The container half of the merge (three existing implementations, `dl` version as base — §3.5) has landed.

## Commands

Toolchain: **uv** (workspace) + **pytest**. Requires `uv` and Python ≥ 3.10.

- `uv sync` — create/refresh the venv; installs all five members editable (plain `.pth`) plus the `dev` group (`pytest`).
- `uv run pytest` — run the suite (3088 tests: 74 in `packages/pfsmgraph-dataseq/tests/`, 2962 in `packages/pfsmgraph-hmm/tests/`, and 52 in the repo-root `tests/` — 23 covering the ADR 0003 backend matrix, 11 executing documented code blocks against their pasted output per ADR 0013, 2 checking that `docs/ops/release.md` names only recipes the root `justfile` defines, and 16 asserting each `meson.build`'s `install_sources` matches the package on disk). That last figure was 7 until 2026-09-04: it is parameterised over `packages/*/meson.build`, so it grew by itself when the three pure members got theirs, and that growth **is** the 271 → 280 — no test was written for the change that occasioned it. That verifier reads `docs/api/*/*.md` **and `packages/*/README.md`**: a member README becomes a PyPI long description under an immutable version, so it is the one documentation surface where drift cannot be corrected in place. Every run opens with the backend header. Two narrow skips are by design: `test_torch_interop.py` verifies the `DataLoader` integration and skips when torch is absent, since `dataseq` does not depend on torch; and the 229 numba-cuda tests (59 in `test_viterbi.py`, 25 in `test_viterbi_batch.py`, 55 in `test_forward_backward_backends.py`, 88 in `test_baum_welch_backends.py`, and 1 each in `test_trials.py` and `test_search.py`: every shared test's `cuda` parameter plus each file's launch-geometry tests) skip without a CUDA device, with the header reading `cuda ✗ (no CUDA device detected)` — set `PFSMGRAPH_REQUIRE_BACKENDS=cuda` where a GPU is expected, so a lost device fails the run instead.
- `uv build --package pfsmgraph-<pkg>` — build one member's sdist + wheel.
- `uv lock` — refresh `uv.lock` (committed; one lockfile for the whole family).

The repo-root `justfile` wraps the release path — `just release <version> [package]` runs
test → build → `twine check` → preflight → upload → tag, defaulting to `default_package`,
the member under development, so an omitted argument can never reach a published package,
and taking any member as its second argument. It requires `just` (`brew install just`) and
is the only place per-package release tags are formed. `just` alone lists every recipe;
[`docs/ops/release.md`](../ops/release.md) is the runbook. Two properties of it are
deliberate and easy to break: a guard must be a *prerequisite* of `release`, never a body
line, because every body line runs after `publish` — the irreversible step; and `clean` is
a prerequisite of `build` because `dist/` is shared by all five members while the publish
glob carries no version, so removing it lets a stale version of the same package be
uploaded.

**Since 2026-09-16 there are two release paths, and `just` refuses the wrong one in both
directions.** A `ci_built_packages` variable names the members whose shipped artifact this
machine cannot build -- today `pfsmgraph-hmm` alone -- and a member joins it when its first
public call reaches a compiled kernel. `release` stops for such a member and points at
`release-ci`, which asserts the tree is clean and pushed and that `pyproject.toml` declares
the version being released, then pushes the `pfsmgraph-<pkg>-v<version>` tag that triggers
the workflow's publish job. `release-ci` stops for a member CI does *not* build, since that
would push a tag no workflow matches and report success having released nothing -- a guard
that points one way leaves the other direction silently wrong. Both refusals are
prerequisites, by the rule above. One consequence is easy to misread as a regression:
`default_package` is `pfsmgraph-hmm`, so a bare `just release` now refuses outright rather
than failing later at `preflight`, which is the stronger form of the same promise.
**`preflight`'s `py3-none-any` assertion was examined for platform wheels and deliberately
not relaxed**: no member publishes platform wheels from here, so it stays correct for every
member still on the local path, and relaxing it would have removed the last check between a
local `pfsmgraph-hmm` release and an upload. Its tree half -- the dirty check and the
`git push origin HEAD` -- is extracted into `_tree-pushed`, which `release-ci` reuses.
**Trusted Publishing cannot be driven from here at all**, which is what makes this two paths
rather than one with a flag: it is OIDC, minted by GitHub Actions for a specific repository,
workflow and environment. The CI artifacts *are* downloadable (`gh run download <run-id>`),
so the reason not to publish them locally is the signing identity, not access.

All five members build through meson-python (landed 2026-09-04), so the root `dev` group carries `meson-python`, `cython`, `ninja`, `numpy`, `numba` **and, on Linux only, `numba-cuda[cu13]` with a `numpy<2.5` cap**. `numpy` is there because it is a build requirement of `align`/`hmm` that `build-system.requires` cannot supply with build isolation off; `numba` was added 2026-09-10 so that ADR 0002 phase 3's backend actually runs under a plain `uv sync`. `numba-cuda` followed on 2026-09-13 for phase 4, for a sharper form of the same reason: `uv run` syncs before executing, so a GPU stack installed outside the lock is uninstalled by the next bare `uv run pytest`, and by the `dp-compile` gate's own build and test commands. The `cu13` extra is load-bearing, since bare `numba-cuda` ships no NVRTC, nvJitLink or NVVM and so compiles nothing; the Linux marker is because it publishes no macOS wheels. Consumers got no numba from `pfsmgraph-hmm` at all in 0.1.0, which withheld its extras while no backend was selectable; they are declared again since ADR 0021's `backend=` (2026-09-14), so the dev group is still the only thing here that exercises those backends, and the two halves are not redundant: an extra is a promise to consumers, a dev group is a promise to this repository, and only the second is checked by anything here. A parallel backend nobody exercises is worse than none. **`hmmlearn>=0.3.3` joined on 2026-09-14 as a test-only oracle**, not a backend: revision 03 checks the numpy forward-backward and EM against its `CategoricalHMM` on the destination-only emission case, where arc emission reduces to state emission (ADR 0015). No member declares or imports it. It brings `scipy` and `scikit-learn` along, and its lock diff was pure insertion, so no existing resolution moved; 0.3.3 is the latest release and ships no cp314 wheels. **`torch>=2.13` joined the same day** for `baum_welch`'s torch backend, on numba's reasoning: `pfsmgraph-dl` already brought torch as a hard dependency, but a backend exercised only through a sibling's resolution is an accident, not a promise. A C compiler is needed **now**: `hmm` got the first `.pyx` on 2026-09-09 -- `_viterbi_cython.pyx`, the ADR 0002 phase-2 Viterbi kernel -- so this repository compiles something for the first time. `align`'s extension block is still dormant. **The dev group alone is not sufficient**, measured 2026-09-04: the generated editable loader bakes an *absolute* `ninja` path at build time and never consults `PATH`, so under uv it points into the build-isolation directory uv deletes once the build finishes, and every import then dies `FileNotFoundError` before the namespace shadowing is even reachable. Those members must also be built without build isolation — `[tool.uv] no-build-isolation-package`.

**This repository has CI as of 2026-09-16, and it is the first.** Two workflows under `.github/workflows/`. `test.yml` runs `uv sync` and `uv run pytest` on every push to `main` and every pull request, with `PFSMGRAPH_REQUIRE_BACKENDS=cython,cpu_parallel,torch` -- the three the `dev` group actually installs, so a skip among them means an import broke rather than a modest environment; `cuda` is deliberately absent, because no GitHub-hosted runner has a device and its 229 tests skip there exactly as on a GPU-less laptop. `release.yml` builds `pfsmgraph-hmm`'s twenty platform wheels with cibuildwheel (linux x86_64/aarch64, macOS arm64, Windows x86_64 × cp310–cp314) and the sdist with `pipx run build`, and publishes through PyPI Trusted Publishing **only** when the ref is a `pfsmgraph-hmm-v*` tag. **Its per-wheel test is narrow on purpose**: only `_viterbi_cython.pyx` and `_forward_backward_cython.pyx` differ between those twenty wheels, so `cpu_parallel`, `cuda` and `torch` are left uninstalled and skip, while `cython` -- whose `needs` is in `_backends.ESCALATED_NEEDS` -- aborts the session if the `.so` did not ship. **The release path itself has now moved too** (2026-09-16): the justfile carries the two paths described above, and `just release` refuses `pfsmgraph-hmm` outright. **The wheel half is measured, though** -- run 35049292620 on 2026-09-16 built all twenty wheels and the sdist from a temporary branch trigger, since removed, with `cython ✓` against each installed wheel and `publish` correctly skipped. Two things it settled are worth carrying. The manylinux floor is **glibc 2.17** (`manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64`), against the `manylinux_2_39` an `auditwheel` repair on a current host is limited to -- which is the whole argument for building in CI rather than locally. And **`release.yml` has no equivalent of the justfile's `preflight`**: nothing in it compares the pushed tag to the version actually built, so the assertion whose comment reads "would publish 0.1.0 and tag it v0.2.0. Both halves of that are irreversible" is absent from the CI path, and the dry run made that concrete by producing 0.1.0 platform wheels whose filenames PyPI would accept onto the already-published 0.1.0. This fires `docs/plan/DEFERRED.md`'s "CI existing" trigger, of whose three items two land here; the agents-docs sync check stays deferred, and the GPU half of `PFSMGRAPH_REQUIRE_BACKENDS` stays deferred with it, since no hosted runner can discharge it.

## Branches and pull requests

**A branch and a PR are for substantial work — a revision subgoal — not for a cleanup.**
Reformatting, a typo, a stale sentence, a link fix: commit those straight to `main` and
push. `main` is not protected, so the direct path is always available, and routing a
two-line fix through a branch, a PR body, a merge and two branch deletions costs more
attention than the change is worth and buries the substantial PRs among trivial ones.

The line is the work, not the file count: a change that a reviewer would have an opinion
about wants a PR, and a change whose whole content is visible in its diff does not.
Recorded 2026-09-03, after three trivial PRs in one session.

## Architecture

`pfsmgraph` is a family of five independently publishable Python packages sharing one PEP 420 namespace, developed in a single repo as a **uv workspace** under `packages/`.

```
                  dataseq          (base — no intra-family dependencies)
                     │
                   align
                  ╱  │  ╲
              hseg  hmm  dl
```

| Distribution | Import | Role |
|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; base layer, PyTorch `Dataset`-compatible |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment (DP-heavy, compiled) |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Baum-Welch, topology search via state merge/split; translated from an existing Lush implementation |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models; `rnn` and `transformer` are plain submodules of one distribution |

The family coheres because `align`/`hseg` are **interpretability instruments for the outputs of `hmm`/`dl`**, and because alignment is a training accelerant for HMM topology search — not merely because the topics are adjacent.

**Implementation order** (`dataseq` → `hmm` → `align` → `hseg`) deliberately differs from **release order**, which must follow the dependency graph since a package cannot publish before its dependencies exist on PyPI. Release order follows *declared* edges, not the drawn graph ([ADR 0019](../design/adr/0019-declared-dependencies-follow-imports.md)): a member declares a sibling when a module in it first imports that sibling, so `hmm`, which imports only `dataseq`, releases before `align`, and re-declares `align` with the alignment-seeded topology revision.

## Invariants

These constrain any code written here. They are inherited from the proof-of-concept and are non-negotiable without amending the PRD.

- **No `pfsmgraph/__init__.py` anywhere.** The `pfsmgraph` level is a PEP 420 implicit namespace; each distribution contributes exactly one regular subpackage beneath it. An `__init__.py` at that level breaks every other package's imports.
- **Encode at the boundary.** Multi-character string symbols are mapped to integers at the entry point of every public call; all inner computation is integer-only; results decode back to strings at exit. This is what makes Cython and CUDA backends mechanical to write — they never touch string types.
- **Fixed reserved symbol block** in `dataseq`, not configurable: `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6. `PAD` must be 0 because PyTorch's zero-fill idioms (`pad_sequence`, `torch.zeros()` buffers) would otherwise silently mean something other than "absent". Encoding is **strict by default** — unseen symbols raise; `UNK` fallback is explicit opt-in.
- **The HMM is arc-emission (Mealy), not state-emission.** A symbol is emitted while *crossing* a transition, so the emission parameter is `output_p[i, j, symbol]` — indexed by source state, destination state and symbol — and never `B[state, symbol]`. A path over *N* symbols visits *N+1* states, which is why `dataseq`'s `seq-state` carries state arrays one longer than its symbol array. Every textbook, and every library (`hmmlearn`, `pomegranate`), is the other formulation, so this is the single fact most likely to be lost in translation; the emission factor also **cannot** be hoisted out of a Viterbi or forward inner loop, because it depends on both endpoints. See [ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md).
- **Four-phase algorithm lifecycle** ([ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md) amends [ADR 0002](../design/adr/0002-three-phase-algorithm-lifecycle.md), 2026-09-03), applied in order wherever dynamic programming appears: pure Python (correctness) → Cython (performance) → Numba CPU-parallel, `prange` (parallel correctness, no GPU required) → Numba CUDA anti-diagonal wavefront (scale). **The decomposition is family-dependent, and only `align`'s is anti-diagonal** — settled 2026-09-10 at `hmm`'s phase 3 and recorded in [ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md)'s `Resolved` section, which withdraws ADR 0002's claim that the wavefront is "the same transformation for every DP kernel in the family". A Viterbi recurrence is 1-D over time with dense `S × S` coupling, and `delta[t, j]` reads *every* `delta[t-1, i]`, so the anti-diagonal `{t + j = c}` holds `(t, j)` and `(t-1, j+1)` with a dependency between them: it is not an independent set, and a wavefront there is a race rather than an optimisation. `hmm`'s phase 3 parallelises **states within one timestep** — `prange` over `j`, with the reduction over `i` left serial and ascending, which is what preserves the first-wins tie-break the next invariant makes contract. Parallelising `i` instead is a parallel reduction that resolves ties arbitrarily, and it fails silently: the oracles hold 0 exact ties in 3804 positions, so only the constructed uniform-model test catches it. **Phase 3 landed 2026-09-10 and is much slower than phase 2, which is the ordinary outcome here rather than a defect** — ADR 0016 scopes the phase to parallel *correctness*. Measured at `N = 400`, post-compile: `S = 5` is 853x slower, `S = 16` 100x, `S = 64` 7.5x, `S = 160` 1.48x, and the cost is **flat at ~34 ms regardless of `S`**. That flatness is the diagnosis, not the ratio: `t` is strictly sequential, so a parallel region opens and closes once per timestep while the work inside is only `O(S²)`, putting the crossover near `S = 250` against a ceiling around 50 here. **Those figures are host-specific**, measured where phase 3 landed with 10 numba threads: re-measured 2026-09-13 on a 4-vCPU Xeon, the same kernel takes 1.4 ms at `S = 5` and already beats phase 2 at `S = 64` (9.9 vs 20.2 ms), because fork/join cost varies by an order of magnitude between machines — a speed claim without its host is not falsifiable. **Phase 4 landed 2026-09-13, and the per-timestep wall is real but not the whole story.** On an NVIDIA L4 at `N = 400` a decode costs ~1.5 ms fixed plus ~54 µs per timestep, flat to `S = 160`: 205x slower than Cython at `S = 5`, level at `S = 64`, and **9x faster at `S = 160`** (24.5 vs 222 ms), where Cython's cost was dominated by its ~10⁷ scalar libm `log2` calls — it was slower than pure numpy there. Those figures predate 2026-09-14, when phases 2-3 moved their logarithms onto the host and out of the inner loop; they have not been re-measured since. The batch had the better speed story, and since 2026-09-15 `viterbi_batch` expresses it with a batched kernel in every phase. **Batching pays on the GPU and barely elsewhere**, measured that day on the same L4 with a 4-vCPU Xeon (`.scratch/hmm-lush/measurements/viterbi_batch_speed.py`, records ragged in [200, 400]): the CUDA batch beat its own per-record loop 7.5-53x at every `B > 1` and is the fastest backend from `S = 64, B = 256` (147 ms against Cython's 466) and `S = 160, B = 16`, but still loses to Cython by 4.4x at `S = 5`, so the crossover moved down rather than vanished. Cython's batch gains 1.1-2.1x, its per-call overhead. **The numpy batch is slower than its loop at `S = 160, B = 256`** (35.8 against 21.0 s), because one call holds ~131 MB each of δ and ψ; that is what `batch_size` bounds. **The batched phase 3 parallelises each timestep's `(record, state)` cells, not whole records, and that is not the fastest CPU axis** (2026-09-15, 4-vCPU Xeon, 4 threads): `prange` over records was 1.4-4x faster wherever `B` filled the threads, 1.6 against 6.9 ms at `B = 64, L = 400, S = 5`. It was declined because a device thread cannot run a record's whole `L·S²` loop, so only the per-step cell grid is phase 4's launch geometry, and ADR 0016 scopes phase 3 to rehearsing that. A race mutant sharing the `i` reduction's accumulator across threads was **not observable** there -- the compiler kept the accumulator in a register -- so no mutant has yet shown the thread-count tests catching a race.
- **One parameterized test suite per algorithm**, run automatically against every available backend, so backend equivalence is enforced rather than assumed. Absent hardware (no CUDA device) skips, but *loudly* — the session header names every backend excluded and why; a backend that is implemented but not importable (missing or stale Cython build) is a hard failure, never a skip; a lifecycle phase not yet reached contributes no parameter at all. `PFSMGRAPH_REQUIRE_BACKENDS` escalates skips to failures for CI. See ADR 0003. **The parameterization half is not in force yet** (found 2026-09-04): ADR 0003 also requires tests be written against the public API only, and `viterbi(params, record)` has nowhere to put a backend — adding one is the runtime backend-selection API that ADR's own Open section deferred, so the two requirements are jointly unsatisfiable until it exists. **It is in force for Viterbi as of 2026-09-14**, decided by `hmm` rather than the `align` that Open section expected: [ADR 0021](../design/adr/0021-runtime-backend-selection.md) gives every public DP call a keyword-only `backend=` string defaulting to `"python"` and raises `BackendUnavailableError` rather than falling back, and `packages/pfsmgraph-hmm/tests/conftest.py`'s `backend` fixture passes each backend in the table to it, skipping an unavailable one with its `backends()` reason. So `backends: python ✓ · cython ✓ · …` now means **each shared `test_viterbi.py` case ran on each available backend**: 55 per backend, including an exact equivalence test against `backend="python"` over 200 generated models. What stays in the labelled non-shared section is what the public call cannot see: the kernel contract (a kernel returns `inf` rather than raising) reached through the private `_resolve`, phase 3's thread-count tests, phase 4's launch-geometry tests, and the libm discrimination check. Forward-backward has no public call, so `test_forward_backward.py` is unparameterised, and its phases are compared in `test_forward_backward_backends.py` through a `phase` fixture over every table row but `torch`; `baum_welch`'s cross-backend suite is parameterised over its own table, `python`, `cython`, `cpu_parallel`, `cuda` and `torch`, through a module-level `backend` fixture. Its corollary is easy to miss: **a tie-breaking rule is contract**, not an implementation detail, because two correct backends would otherwise legitimately disagree.
- **~~Build backends are per-package, not family-wide.~~ Superseded 2026-09-04: the backend is family-wide, and it is meson-python for all five members.** The original invariant (ADR 0008) said meson-python for compiled members and hatchling for pure ones; that split is what the namespace shadowing makes unworkable, since a member on a plain `.pth` is shadowed by any sibling's meson-python finder. What survives from it: meson-python editable installs need `ninja` present for rebuild-on-import, and — because the loader bakes an absolute path to it rather than consulting `PATH` — every member must also be built without build isolation. See "Current state" and [ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md).
- **A released member ships four files the version bump does not imply**, all inside
  `packages/pfsmgraph-<pkg>/`: a `README.md` (its PyPI long description — the root one is
  about the workspace and every relative link in it 404s there), a `LICENSE` **file**, the
  `Typing :: Typed` classifier, and a PEP 561 marker at
  `src/pfsmgraph/<pkg>/py.typed`. Two of these fail *silently* if placed wrong, which is why
  they are an invariant rather than a checklist. A `LICENSE` symlinked to the repo-root one
  builds a valid-looking sdist and then fails on **unpack** — a symlink escaping the sdist
  root is refused — so it must be a real copy. The copy is also a
  **silent drift surface**, and the mechanism is not the one this sentence used to claim.
  It read "the wheel ships the member's copy, not the root one"; **measured 2026-09-16, the
  wheel shipped *neither*.** meson-python includes no license file unless `license-files`
  is declared, so `pfsmgraph-hmm` 0.1.0 went to PyPI with `License-Expression: MIT` in its
  METADATA and no license text at all -- valid PEP 639, accepted by `twine check`, rendered
  as MIT by PyPI, and visible only by opening the archive. `pfsmgraph-dataseq` 0.1.0 does
  carry `dist-info/licenses/`, because it was built by **hatchling**, which globs `LICENSE*`
  by default; the invariant was written in the hatchling era and inherited unexamined into a
  backend with different defaults. **So a released member must also declare
  `license-files = ["LICENSE"]`** -- one line, verified to produce
  `dist-info/licenses/LICENSE` and a `License-File:` field. None of the five declared it as
  of 2026-09-16; `hmm` gained it at 0.2.0, and `align`/`hseg`/`dl` should gain it before
  their first release, where it still costs nothing. The old conclusion survives its broken
  premise: editing the repo-root `LICENSE` alone still changes nothing a consumer sees. Keep the two byte-identical
  — both carry the copyright line `Copyright (c) 2026 Panayotis Mavromatis`, the legal name
  rather than the professional one, because a license is read by lawyers. A `py.typed` at the distribution root instead
  of inside the importable package reaches no wheel at all, with no error and no warning, and
  a type checker then discards every annotation in the package (measured on `dataseq`: a
  deliberate `bad: str = vocab.size` was *accepted*). It cannot go at the `pfsmgraph/`
  namespace level either, for the same reason no `__init__.py` may: no single distribution
  owns that level. **Since all five members moved to meson-python (ADR 0018) there is a
  third way it misses the wheel**, and it is the one the next release commit will hit:
  `meson.build` does not glob, so a marker at the correct path but absent from that
  member's `install_sources` reaches no wheel either. That one is *not* silent —
  `tests/test_meson_sources.py` covers `py.typed` by name — so adding the marker and
  listing it are one change, not two. Verify by installing the built wheel into a clean venv outside the
  workspace — a file listing shows what went into the box, not what a consumer gets out.
  All four land **in the release commit**, since they are wheel content and adding them later
  leaves a published version standing as the broken one.
- **"GPU" means two unrelated things.** `numba-cuda` for the DP packages, `torch` for `dl` and for `hmm`'s `baum_welch` torch backend, whose `device="cuda"` needs the `torch` extra and never `gpu`. Do not unify these into one `[gpu]` extra. **And not everything Numba is GPU**: `pfsmgraph-hmm` declared `cpu-parallel = ["numba>=0.61"]` for the ADR 0002 phase-3 backend from 2026-09-10, separately from `gpu`, because it needs no device and is deliberately its own extra; 0.1.0 withheld both while no backend was selectable, and both are declared again since ADR 0021 (2026-09-14): `gpu = ["numba-cuda>=0.30.4", "numpy<2.5"]`, both marked `sys_platform != 'darwin'` because numba-cuda publishes no macOS wheel or sdist, so the extra installs nothing there and `backends()` says so rather than the install failing to resolve. `gpu` happens to satisfy it transitively, because `numba-cuda` depends on `numba` — a resolution accident, not a reason to merge them. numba imposes a numpy **ceiling**: measured 2026-09-10, `numba==0.61.*` caps numpy at 2.2.6 against this member's `>=2.1` floor. **`numba-cuda` imposes a tighter one that its metadata does not declare**: 0.30.4, the latest as of 2026-09-13, calls `np.row_stack` at import and numpy 2.5 removed it, so the workspace carries a Linux-marked `numpy<2.5` beside it — a resolution that installs cleanly and then dies `AttributeError` at the first `cuda.jit` import. That is why both are optional rather than required — as a hard dependency the ceiling would bind every consumer of `pfsmgraph-hmm`, including one that only ever runs the pure-Python decode. Note the direction: a transitive *upper* bound arriving from someone else's metadata is the mirror image of the workspace footgun below, and neither is visible in `uv.lock`, which records one resolution rather than the space of them.
- **The ADRs outrank the imported implementations.** The `dataseq` merge takes the `dl`
  (MelodyHPO) version as its *base* because it is the most mature of the three, but base
  means starting point, not authority. Where any of the three disagrees with an Accepted
  ADR, the ADR wins and the implementation is changed, unless that ADR says otherwise in
  its own text. The imports are evidence about what has been tried; they are not a source
  of decisions that have already been made. This bites hardest on the reserved block:
  four sources use three different offsets for user symbols — `dl` at 3, the Lush original
  at 2, the rudimentary `segalign` at 2 (with `PAD` at **1**), the proof-of-concept
  `tokalign` at 4. The two that collide are both containers, and they mean different things
  by the same integers: `0` is `begin` in Lush and `:EOS` in `segalign`. **None of the three
  containers has a `GAP` code**; `tokalign` does, at index 3, which is why its user symbols
  start at 4 — the one source with a gap code being the one written to align sequences.
  *(Corrected 2026-08-31 by the goal-4 measurement in
  `.scratch/RESERVED-BLOCK.md` §2, which is authoritative for this table. The earlier wording
  counted the proof-of-concept as one of the three containers and denied it a `GAP` code;
  both were written from recollection before `segalign` and `tokalign` had been read.)*
  *(As-imported offsets. `tokalign` was renumbered onto the block on 2026-09-01 and now
  puts `GAP` at 4 with user symbols from 6; the other three are unchanged, `dl`'s having
  been settled inside the merge. The offsets above stay written as they are because this
  paragraph is about what the four sources disagreed on, which is the evidence ADR 0011
  was decided against.)*
  [ADR 0011](../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
  settles this: `PAD`=0 … `MSK`=5, user symbols from 6, and the renumbering lands **as part
  of** the merge rather than after it -- which it did, on 2026-09-01, closing that
  `DEFERRED.md` entry. So a merge note reading "the base must be overridden
  here" is about the points the ADRs leave open, never about reopening the ones they close.

## Workspace footgun

During development a `{ workspace = true }` path source satisfies *any* version constraint, so a missing or wrong bound in `[project.dependencies]` never fails locally — it only breaks a pip user after publish. Keep published lower bounds honest and review them on every breaking change.

The `0.0.0` placeholder releases already on PyPI are intentionally dependency-free; do not add dependency declarations to them.

A live instance of the footgun, worth recognising: the three members still in development declare `0.1.0.dev0` (`dataseq` moved to `0.1.0` at its release commit, 2026-09-02, and `hmm` at its own, 2026-09-13), and `0.1.0.dev0` does **not** satisfy `>=0.1` under PEP 440 — a `.devN` release sorts strictly before the final, and is excluded even with `prereleases=True`. So `align`'s declared `pfsmgraph-dataseq>=0.1.0` (reviewed and spelled in full on 2026-09-01; the `pfsmgraph-align>=0.1` bounds are deliberately left unreviewed, and the divergent spelling is what records that) is satisfiable by nothing that exists today: PyPI has only `0.0.0`, and local is `0.1.0.dev0`. It never fails because the workspace source satisfies any constraint, and **`uv.lock` cannot catch it either** — a workspace member's `requires-dist` entry records no version specifier at all, so changing all four declared bounds left the lockfile byte-identical (measured 2026-09-01). Review is the only mechanism there is. **Qualified 2026-09-16, while freezing `hmm` 0.2.0's metadata: review is the only mechanism for the *bound*, and not for the thing the bound is about.** Whether a declared floor is satisfiable stays unresolvable locally, for the reasons just given. But whether the names the dependent imports actually exist in the published release is checkable mechanically and cheaply: `uv run --no-project --with 'pfsmgraph-dataseq==0.1.0' python -c ...` installs from PyPI into a throwaway environment, outside the workspace and outside the lock, and its `__all__` can be compared against what the dependent imports. Done for `hmm` 0.2.0 -- all five names it takes from `dataseq` (`RESERVED_SYMBOLS`, `USER_BASE`, `Vocabulary`, `SequenceRecord`, `pad_collate`) are in the published 0.1.0's 15-name `__all__`, the first time that bound has been checked by anything. It does not make the *version* right, since an older release might carry the same names; it makes the version's content checked rather than assumed, which is the half a reader can actually be wrong about. The `pfsmgraph-dataseq` half of this resolves as soon as `0.1.0` is on PyPI; the two `pfsmgraph-align>=0.1` bounds stay unsatisfiable until `align` releases. **A third blocked a release and is gone** (found 2026-09-13 reconciling `docs/api/hmm/` against the metadata; removed on `chore/release-hmm-0.1.0` under [ADR 0019](../design/adr/0019-declared-dependencies-follow-imports.md)): `pfsmgraph-hmm` declared it while no module imported `align`, so an `hmm` 0.1.0 published before `align` would not have installed for any pip user. `hseg`'s and `dl`'s face the same check at their own release commits. Note the asymmetry the removal measured: `uv lock` rewrote exactly two lines, `hmm`'s `dependencies` and `requires-dist` entries, so the lockfile sees an edge appear or vanish even though it never sees a bound change.

## Versioning

**Versions are per-package, and there is deliberately no `VERSION` file at the repo root.** Release order is forced by the dependency graph — `dataseq` must publish before `align` can — so the five members can never share a version, and a repo-wide version number would be a claim about nothing. Each member owns the `version` field in its own `pyproject.toml`: `pfsmgraph-dataseq` reads `0.1.0` as of its release commit (2026-09-02); `pfsmgraph-hmm` read `0.1.0` from its own (2026-09-13), moved to `0.2.0.dev0` on 2026-09-16 when the 0.2.0 release branch needed artifacts to verify, to a bare `0.2.0` at that release's own commit the same day, and to `0.3.0.dev0` hours later at revision 04's opening — **the version field leads the release rather than following it**, so a bare number in the tree means the release commit has been made, not that PyPI has the version; and the other three still read `0.1.0.dev0` — the scheme working as intended rather than drift. **The window between a release commit and the next `.devN` bump is worth naming, because its hazard is not the one the `.dev0` rule is usually explained by.** For those three days `hmm` declared a bare version that was *already published*, and the risk there is not burning a number: a local build now produces a **platform** wheel, whose filename differs from the `py3-none-any` one PyPI already holds for 0.1.0, so an upload would be **accepted** onto the released version rather than rejected as a duplicate. Adding files to a published release changes what `pip install pfsmgraph-hmm==0.1.0` gives people. Two guards added on `chore/release-hmm-0.2.0` close that path — `release` refuses `pfsmgraph-hmm` outright, and the publish job asserts every artifact carries the tag's version — but the cheap structural fix is to bump to the next `.devN` at the release commit rather than days later. That fix was missed again at 0.2.0 and applied at revision 04's opening instead, so the second window was hours rather than days — narrower, and with both guards already in place, but the same shape.

Release tags are per-package too: `pfsmgraph-<pkg>-v<version>`, e.g. `pfsmgraph-dataseq-v0.1.0`. Hyphen rather than slash, because git refs are paths and a `pfsmgraph-dataseq/v0.1.0` tag cannot coexist with a plain `pfsmgraph-dataseq` one. The first is `pfsmgraph-dataseq-v0.1.0`, cut by hand at the release commit (`docs/plan/DEFERRED.md`, trigger "the first real release"); no command in use here creates a per-package tag.

The `.dev0` suffix stays until that release commit. `uv build` stamps whatever `pyproject.toml` declares onto the wheel, so a bare `0.1.0` on an incomplete package means one accidental publish burns `0.1.0` on PyPI permanently — versions are immutable, and yanking or deleting a release does not free the number. A burnt `0.1.0.dev0` costs nothing by comparison, and pip will not install a pre-release by default.

## Design docs

- `docs/ops/release.md` — the release runbook: what a member ships, how the `justfile` recipes
  compose, and the token/Trusted-Publishing posture. `tests/test_release_runbook.py` checks that
  every recipe it names exists, and nothing more — the boundary against ADR 0013 is argued in
  that test's docstring.
- `docs/design/PRD.md` — packaging, naming, and distribution architecture; the source for the initial ADR set (§9).
- `docs/plan/DEFERRED.md` — decided-but-not-yet-actionable work, indexed by the trigger that unblocks it (the `dataseq` merge, the first `.pyx`, CI existing, the `align` migration, the first real release — an illustrative list, not the full set; the file's `## Trigger:` headings are). Check it when starting any of those; several items must land *as part of* their trigger rather than after it.
- `docs/design/arc-emission-hmm-handoff.md` — a design conversation handoff from 2026-06-13, moved out of an untracked scratch directory on 2026-09-03 and kept **verbatim under a provenance preamble**. Authoritative for nothing; where it disagrees with an ADR or the master plan, they win. It is retained because it is the only written source for the arc-emission commitment's rationale (now [ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md)) and for the alignment-derived seed mechanism that `core.md`'s "alignment is a training accelerant for HMM topology search" has been standing on since the PRD. Its §2.1 recommends PyTorch and **does not** reopen the settled numpy-reference decision — the preamble says why.
- `docs/benchmarks/` — dated measurements, **authoritative for nothing**, added 2026-09-17
  with `hmm-backends.md`: which `hmm` backend to choose, what a search round costs before and
  after ADR 0024 §2, and whether a `torch` search ranks candidates differently (it did not).
  Timings live here rather than in `docs/api/`, whose blocks `tests/test_api_docs.py`
  executes against pasted output, so a host-specific number there would fail on the next
  machine. Each page names the measurement script and the SHA-256 of the version that
  produced its log, so staleness is detectable the way a `FORMALIZATION.md`'s is.
- `docs/design/references.md` — external work a decision here leans on. Each entry records **what it is relied on for**, not a bare citation, so a misremembered or stale reference is detectable rather than merely present. Added 2026-09-03.
- `docs/plan/TODO.md` — the master plan. Its `## Planned revisions` section registers revisions drafted but not yet opened; each one's subgoals are drafted as `docs/plan/planned/<label>.md`, so opening it is a splice rather than an authoring job. **The draft must not sit in `docs/plan/<label>/`**, tempting as that is since it is the shape `/close-revision` archives: `/open-revision` refuses any label whose directory already exists, correctly assuming only `/file-plans` creates one, so a draft filed there makes a free label look taken and the command refuses on every planned revision. Measured 2026-09-03, when it did. Each draft carries a `**Status**: planned` preamble that stays behind at splice time.
- `docs/design/algorithms/<algorithm>/FORMALIZATION.md` — the language-agnostic
  specification a DP kernel's lifecycle phases are written against: recurrence, base cases,
  iteration order, backtrace, parallel decomposition, differential oracles and numbered test
  cases. It is **phase 0** of the [ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md)
  chain — a specification, not a backend, which is why the four-phase count and the
  `dp-compile` plugin's five-stage count agree (see [`claude.md`](claude.md)). The first is
  `viterbi/`, landed 2026-09-09 and **recovered from the phase-1 implementation rather than
  written before it**; its `Derived from` line carries the SHA-256 of `_viterbi.py`, and the
  document is stale when that hash stops matching. Read it before writing any later phase:
  its tie-breaking rule is contract, and its `Parallel decomposition` section names the
  decomposition each later phase must use. That section read "undetermined" from the
  recovery until 2026-09-10, when phase 3 settled it as **states within one timestep** and
  explicitly not anti-diagonal; the entry it replaced is a worked example of "undetermined"
  recorded as an answer rather than a gap. **It carries a `Batched recurrence` section since
  2026-09-15**, whose two contract choices are what make padding testable: the final argmin
  reads column `L` rather than each record's length, and ψ past a record's end is unread.
  Its hash had been stale since the backend seam; it was refreshed **by a checked amendment
  rather than a re-derivation**, after confirming with docstrings stripped that `_viterbi`
  and its helpers were unchanged. The document's own rule, re-derive when stale, was left
  as written. **The second is `forward_backward/`**, recovered 2026-09-15 on
  `feat/hmm-forward-phases` from `_forward_backward.py` and specifying both the per-record
  recurrences and the batched E-step, which every later phase implements. It has no
  tie-breaking rule, and **its evaluation order takes that place as contract**: every sum is
  an ordered fold, ξ associates as `(α × w) × β`, and no backend fuses multiply-add. Its
  `Parallel decomposition` was decided *before* phase 3 rather than by it: each timestep's
  `(record, state)` cells, with every reduction serial. The associative scan ADR 0020 leaves
  open is rejected there. It specifies 0 bits for an empty record, where exact enumeration
  gives `-log2 Σ init_p`; the kernel's docstring claimed otherwise until it was corrected
  after the merge, and the document's hash was then refreshed by a checked amendment, as
  Viterbi's was, since only a docstring had moved.
- `docs/design/adr/` — twenty-four records: the twelve initial ADRs from the PRD plus 0013 through 0024, authoritative for the decisions they cover; [`adr/README.md`](../design/adr/README.md) indexes them. Add new records with the next unused number and a row in that index; numbers are never reused. **0015 is the first record about a model rather than about packaging, tooling or process**: `pfsmgraph.hmm` is arc-emission (Mealy), so a symbol is emitted on the transition `i → j` and the emission parameter is `output_p[i, j, symbol]`, never `B[state, symbol]`. Every textbook and every library is the other one, so this is the fact most likely to be lost in translation — and it is why `dataseq`'s `seq-state` carries state arrays one longer than its symbol array. **0017 is the second, and reads after it**: model parameters are a *frozen value* — the three arc-emission arrays with `writeable = False` buffers and the `Vocabulary` that fixes what their symbol axis means, derived quantities (the stationary distribution, the entropies) computed rather than stored, and algorithms taking parameters rather than owning them. Viterbi is therefore a free function over parameters and one `dataseq` record, not a method on a trainer: the Lush decode reads no forward variable, so its placement on `hmm-trainer` was an artefact of where the corpus lived (`HMMLIB-ACCOUNT.md` §7). It returns a result rather than writing the path back into the sequence object, and `pad_collate` is out of scope until revision 03 — a record never holds padding, so a single-sequence decode has no mask to consult. Lush's `hmm`/`hmm-param` mutable working-copy split is deliberately **not** inherited; its motivation was two buttons in a GUI that migrates nowhere, and its own surgery methods reallocate rather than mutate in place, so it never bought what it appeared to. **0020 is a numeric contract, and it is the one to read before writing any forward-backward code**: the recurrence runs in the scaled probability domain, as the Lush original does, not in log space. The kernel performs only `+ * /` over a host-built `(S, S, U)` table of arc probabilities; summation order (ascending `i`, then `j`) is part of the contract; the only logarithms are `bits(Q_t)` of the scale factors, on the host; and no backend may fuse multiply-add. Viterbi's contract needed none of the ordering, because `min` gives the same result in any order and a sum does not: numpy's `@` and an ascending loop disagree by 1 ulp in about half of all scale factors. The promise is bit-exactness between the numpy reference and the compiled phases on one host; `torch` is held to `N·eps·max(1, x)` per element, measured and amended in 2026-09-14. **0022 is the third record about the model, and the one to read before writing a split, a merge, `try-split` or the search loop**: a split carries `init_state_p` (the Lush `:172`/`:262` seeding from `state_p` is a defect, confirmed by the original author), perturbs each predecessor's arc into the split state independently rather than redrawing the twins' outbound emissions, seeds every live outbound fibre with a 1% `rand_p_vector` mixture, and obliges whatever re-converges a split to give it a minimum EM budget, because a split that starts at the incumbent's likelihood trips `run-converge`'s stop before its twins separate. **0023 is the search loop, and reads after 0022**: it ports the original's headless `repeat (suggest-move) (keep-model)` loop (`Training/hmm-train-new-nw`, which `HMMLIB-ACCOUNT.md` §11 had wrongly said did not exist) and names its departures. The walk keeps each round's winner unconditionally but returns the best model it visits by strict `<`. It stops on patience, a required `max_rounds` or a round with no possible move. Seeds are keyed by role, `(1, r, s, t)` for split trial `(s, t)` of round `r`. `d` comes from a bounded exact scan, a split gets a 200-cycle EM floor (on `m008` 100 separated no split and 200 all 16), and the total's forward pass takes the search's backend, where Cython is 120–330 times faster than numpy and bit-identical. Decided and written 2026-09-17, in `_search.py`. **0024 is Accepted, and reads after 0023 before anyone compiles or parallelizes the search**: the search, the trials and the surgery stay interpreted, because a profile of one round (`m001_0005_005`, `cython` throughout) spent 10.6 of 12.7 s in the numpy forward pass that `baum_welch`'s convergence check runs whatever `backend=` says, and about 1 s in everything above the kernels. The phases are alternatives chosen per call by name, not layers, and all but `torch` return bit-identical results. The record requires that check to take a named backend, by a separate `score_backend=` on `baum_welch` rather than a phase derived from `backend=`, which would substitute one backend for another as ADR 0021 forbids. It defers parallelizing the search along trials within a round, which stays bit-identical to the serial run when results are gathered in enumeration order. It was implemented on `feat/hmm-convergence-backend` the same day: the check takes `score_backend=`, which took a measured round from 9.38 s to 1.70 s, and a torch search was then measured *not* to leave the serial path, all 66 candidates scoring bit-identical totals.
