# 0020. Forward-backward runs in the scaled probability domain, in a fixed order

- **Status:** Accepted
- **Date:** 2026-09-14
- **Source:** none in the PRD — postdates it. Settles the log-base question that
  `docs/plan/TODO.md` poses in revision 03-hmm-v0.2.0's forward-backward subgoal, and the
  semiring its phases 2-4 subgoal deferred to that answer. Extends the numeric contract
  that [`viterbi/FORMALIZATION.md`](../algorithms/viterbi/FORMALIZATION.md) states for the
  decode to the sum-product recurrences.

## Context

Revision 03 builds Baum-Welch on a fixed topology. Its first kernel is forward-backward:
α, β, the pairwise posteriors ξ, the state posteriors γ, and the sequence likelihood. The
numpy version is the reference that the M-step, the `hmmlearn` oracle, the `torch`
backend and the compiled [ADR 0016](0016-numba-cpu-parallel-phase.md) phases are all
checked against. The master plan drafted it "in log space" and left one question open:
whether the base stays 2, which suits the description lengths of revision 04, or becomes
*e* with a conversion at the description-length boundary.

Three facts reshaped that question before it could be answered.

**The original never works in log space.** `update-data-p`, `update-data-dl` and
`run-add` (`hmm-trainer.lsh:126-184`, `346-402`, `483-652`) all run a scaled
probability-domain recurrence, Rabiner's scaling. Each α column is divided by its own
sum `Q-t[i]`, and `run-add` divides β by the same factors. That makes `P-t = α*·a·o·β*`
ξ already normalised by the likelihood. Logarithms are taken of the `N` scale factors
alone, in base 2 through `safe-add--log2`, and their sum is the data description length
(`HMMLIB-ACCOUNT.md` §3, §6, §9). "Log space" was the draft's intent, not a translation.

**The decode's numeric contract does not transfer by itself.** Revision 03's first
subgoal (`fix/hmm-viterbi-log2`, PR #28) made all four Viterbi backends bit-exact on one
host. It did so by taking every logarithm on the host through `bits`, leaving the kernels
only `+` and `<`. That was enough because Viterbi's reduction is `min`, which gives the
same result in any order. A sum is order-dependent, and log-space forward-backward needs
`exp` and `log` inside the recurrence, where every backend's implementation rounds
differently.

**A log-space gradient fails on a state revision 04 constructs.** The revision's
preamble names it as a falsifier: `+inf` bits become `-inf` log-probabilities inside a
differentiable log-sum-exp, and the gradient may come out `nan`. `split-state` creates
states that are unreachable at some positions, so this is a search outcome, not a
corner case.

## Decision

**Forward-backward is computed in the scaled probability domain, as the original
computes it, and every backend follows one fixed evaluation order.** The contract has
four parts.

1. **The kernel performs only `+`, `*` and `/`.** The arc weights
   `w[i, j, u] = transition_p[i, j] * output_p[i, j, present[u]]` are built on the host
   by numpy, over the symbols present in the record. This is the `(S, S, U)` table the
   Viterbi backends already use, without the logarithm. Each accumulated term is then a
   product of two factors. IEEE multiplication is commutative, so how a longer product
   associates never enters the contract.
2. **The summation order is part of the contract.** α accumulates over the source state
   `i` in ascending order, starting from `0.0`, for each destination `j`. `Q_t` sums the
   new column over `j` in ascending order. β mirrors this over `j`. The numpy reference
   implements that loop order explicitly and does **not** use `@`, `np.dot`,
   `np.sum` or `einsum` for these reductions, since each is free to reduce in whatever
   order its SIMD or BLAS implementation prefers.
3. **The logarithms are taken on the host, of the scale factors only.** The data
   description length is `Σ bits(Q_t)`. The decode's rule holds unchanged: numpy's
   `log2`, through `bits`. A zero scale factor marks an impossible sequence. Scaling uses
   `safe_divide`, so the α column goes to zero and `bits(0)` makes the total `+inf`. This
   is the behaviour of `update-data-p` and `update-data-dl`. `run-add`'s bare `/`, which
   turns the same sequence into `NaN` in every re-estimated parameter, is not reproduced.
4. **No backend fuses multiply-add.** A compiled kernel is built so that `a*b + c` is
   never contracted into a fused instruction, and numba kernels never set `fastmath`.

**What the contract promises is bit-exact agreement between the numpy reference and the
compiled phases on one host.** It is not bit-exact across machines, for the same reason
the decode's is not: numpy's `log2` dispatches by CPU. Nor is it bit-exact against
`torch`, whose reductions no caller can order. The `torch` backend is held to the
reference **within a stated tolerance of a few ulps**, and its tests say so.

The base is 2, because the only logarithms are description lengths. The semiring the
phases 2-4 subgoal asked about is **sum-product with per-step renormalisation**. It is
neither max-plus, which is Viterbi's min-sum, nor log-sum-exp.

## Consequences

### Positive

- **The decode's numeric contract extends rather than splits.** One rule, numpy's `log2`
  through `bits` on the host with only arithmetic in the kernel, now covers both kernels.
  Phases 2-4 of forward-backward can be held bit-exact with the reference, as Viterbi's
  are, instead of starting from a tolerance.
- **`torch` gradients stay finite on unreachable states.** In the scaled domain a dead
  column contributes a product of 0 and a gradient of 0. The log-space version produced
  `NaN` in every transition gradient, not only the dead ones.
- **Revision 04 needs no conversion.** The scale factors are already in the unit its
  minimum-description-length score uses.
- **The translation stays checkable against the original.** The recurrence, its scale
  factors and ξ's normalisation match `run-add` term for term, so `HMMLIB-ACCOUNT.md` can
  be read beside the code.
- **Compiled phases avoid a transcendental call per arc.** Log space would evaluate `exp`
  and `log` on every arc at every position. That is the cost that made Viterbi's Cython
  phase slower than numpy at `S = 160` before its logarithms moved to the host.

### Negative / costs

- **The reference is slower than it could be.** A matrix product is the natural numpy
  form, and it was 2-4× faster than log space in the measurements below. The explicit
  ascending loop gives that up to make the reference a specification of evaluation order
  as well as of value.
- **Evaluation order is now a correctness property, and it is easy to break silently.**
  Replacing a loop with `np.sum`, reordering indices, or letting a compiler contract
  `*` and `+` changes about 1 ulp in half the scale factors. Every value stays within
  tolerance, so only a bit-exact cross-backend test notices.
- **Build flags become part of the contract on some platforms.** The Cython build on
  x86-64 is safe only because `meson.build` passes no `-march`, so GCC targets baseline
  x86-64, which has no FMA. aarch64 GCC and Apple clang contract by default and will need
  `-ffp-contract=off`. That is a platform-conditional compiler argument this repository
  does not yet have.
- **The `torch` comparison is a tolerance, not an equivalence.** The master plan hoped
  masking could keep it exact. No domain does, because torch's reductions are not
  ordered by the caller.
- **The master plan's wording changes.** The forward-backward subgoal said "in log
  space", and the phases 2-4 subgoal kept "max-plus/logsumexp" pending this answer. Both
  are superseded here rather than edited in place.

## Alternatives considered

- **Log space in base *e*, converting to bits at the description-length boundary.**
  Rejected. Its accuracy was indistinguishable from the scaled domain, but it was 2-4×
  slower in numpy. It puts `exp` and `log` in the recurrence, so no two backends can be
  bit-exact. And its gradient is `NaN` on a state unreachable at every position, which
  revision 04 constructs.
- **Log space in base 2**, with `np.logaddexp2`. Rejected for the same reasons. It was
  also the slowest option beyond small `S`, since `logaddexp2.reduce` runs as a
  Python-level reduction: 268 ms against 49 ms at `S = 160`.
- **The scaled domain with numpy's matrix product, and a tolerance for every
  cross-backend test.** Rejected. It is the fast reference, but it gives up bit-exactness
  with the compiled phases for good, and it brings back the "tolerance nobody chose" that
  revision 03's first subgoal existed to prevent.
- **Log space with dead arcs masked out of the sum.** Rejected. Masking fixes the `NaN`
  but not the per-backend `exp`/`log` rounding, and it adds a mask whose own correctness
  would need testing, to recover something the scaled domain gives unconditionally.

## Evidence

All measured 2026-09-14 on an Intel Xeon at 2.20 GHz with AVX-512 and FMA, numpy 2.4.6,
torch 2.13.0+cu130, Python 3.12.3, seed `20260914`. The scripts are tracked under
[`.scratch/hmm-lush/measurements/`](../../../.scratch/hmm-lush/measurements/), one per
group of bullets, and each opens with its run command.

- *Observed, accuracy:* against exact rational enumeration over every state path, on
  200 random models (`S = 3`, 4 symbols, `N = 7`, 30% dead arcs), the relative error in
  total bits was median / max `1.75e-15` / `4.31e-15` for the scaled domain, `1.75e-15` /
  `4.68e-15` for base *e*, and `1.70e-15` / `4.46e-15` for base 2. At `N = 100 000` and
  `S = 16`, a total of 300 235 bits, base *e* differed from scaled by `-6.5e-9` bits and
  base 2 by `+9.0e-9` bits, relative `2e-14` and `3e-14`. All three return `+inf` for an
  impossible sequence.
- *Observed, speed:* numpy forward pass at `N = 400`, vectorised per timestep, scaled /
  base *e* / base 2 in ms. `S = 5`: 2.9 / 9.2 / 1.9. `S = 16`: 3.8 / 11.6 / 4.7.
  `S = 64`: 5.9 / 22.5 / 41.0. `S = 160`: 48.7 / 102.5 / 268.2. These time the
  matrix-product form this record rejects for the reference; they compare the domains,
  not the reference's eventual speed.
- *Observed, order:* scaled scale factors from numpy's `@` against an explicit ascending
  `i`-then-`j` loop differed in 48%, 62% and 77% of positions at `S = 4`, 16 and 64, by
  at most `1.3e-15` bits. torch float64 on the same operations differed from numpy in
  62% of positions on CPU and 40% on CUDA, by 1 ulp.
- *Observed, contraction:* a numba `@njit` accumulation of `x*y*z` matched unfused Python
  floats in 2000 of 2000 trials. With `fastmath=True`, 900 of 2000 differed.
- *Observed, autograd:* on a 6-state model with 40% dead arcs and one state that cannot
  start, `d log P / d log a_ij` matched numpy's explicit expected transition counts to
  `8.9e-16` in the scaled domain and `2.8e-14` in log space, with zero gradient on dead
  arcs in both. On a 3-state model where one state is unreachable at every position, the
  scaled domain gave 0 of 9 transition gradients `NaN`, and log space gave 9 of 9.
- *Observed:* `packages/pfsmgraph-hmm/meson.build` sets `buildtype=release` and no
  `-march`.
- *Reasoned, not measured here:* GCC on aarch64 defaults to `-ffp-contract=fast` with
  FMA in the baseline instruction set. Apple clang defaults to contracting within an
  expression.

## Extended

**2026-09-14, while the reference was written.** The Decision fixes the order of the
recurrences, α, β and the scale factors, but the reference also reduces over positions.
The same rule is extended to those reductions:

- **The description length** is `Σ bits(Q_t)` summed in ascending `t`.
- **The expected counts** are accumulated in ascending `t`, one elementwise addition of
  that position's ξ per step. `init_counts` is ξ₀ summed over the destination in
  ascending order, as `run-add` forms `C-in`.
- **ξ is never stored in full.** Each position's `(S, S)` slice is built, added and
  discarded, so memory is `O(S²·A)` rather than `run-add`'s `O(N·S²)`. This is a
  consequence, not a new rule.

The ordered reductions are `np.add.accumulate(...)[-1]`. It returns every partial sum,
so its order is fixed by definition, and it was bit-identical to an explicit ascending
loop at `S` = 4, 16, 64 and 160 while being 2-66× faster than that loop. That makes
most of the reference-speed cost under *Negative / costs* moot: at `S = 160` it took
0.14 ms per step, about the same as the `@` form measured above.

## Open

- **Whether numba-cuda contracts multiply-add in device code.** Unmeasured. Settle it at
  forward-backward's phase 4, with a constructed test in the manner of Viterbi's TC-21.
- **The platform-conditional `-ffp-contract=off`.** It is not needed on the x86-64 host
  where the Cython phase will first build. Add it when a non-x86 build is first
  exercised, or pre-emptively at forward-backward's phase 2 if the constructed test
  would not run on such a host anyway.
- **The `torch` tolerance.** "A few ulps" is the float64 expectation. A float32 `torch`
  run needs its own figure, recorded in the `torch` subgoal as the master plan already
  requires.
- **An associative scan over time.** Sum-product with renormalisation is associative up
  to the scale factors, so a parallel prefix is possible in principle. Whether it is
  worth it, or whether batch parallelism alone is the phase-3 answer, is still the phases
  2-4 subgoal's question. That subgoal now knows which semiring it is asking about.
