# Viterbi decode (arc-emission HMM)

## Metadata

| Field | Value |
|-------|-------|
| Source | **Recovered from the phase-1 implementation.** `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi.py` at commit `232a207`, which was ported from `update-viterbi-path` in `.scratch/hmm-lush/Code/HMMlib/hmm-trainer.lsh:216-227`. The written account of that original is `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, §3 (the bit domain) and §7 (the two defects). |
| Derived from | `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi.py` `sha256:48666ca8a3126ba141ec1a0ddba8ed1823e502f93d72483181906afd24135742` |
| Family | Hidden Markov model — **arc-emission (Mealy)**, [ADR 0015](../../adr/0015-arc-emission-mealy-formulation.md) |
| Variant | Decode — the single most probable state path. Not forward, not posterior. |
| Objective | **Minimise the total description length of the path, in bits: the min-plus (tropical) semiring.** Its identity is `0.0` and its absorbing element is `+inf`. |
| Parallel decomposition | **States within one timestep** — `prange` over `j`, serial reduction over `i`. This recurrence has no anti-diagonals. See the section below. |
| Time complexity | `O(N·S²)` |
| Space complexity | `O(N·S)` |
| Optimality | Optimal — exact minimisation over all `S^(N+1)` paths. |
| Algorithm-specific params | None. The signature is `(params, record)` and nothing else. |

**The direction and the semiring are the two facts a port loses first.** The accumulated
quantity is `-log2(p)`, so it *grows* as probability falls, and the decode is a **min-sum,
not a max-product**. A port reaching for `max` inverts every comparison. The original's own
naming conceals this: its `data-p` and `result-p` hold bits despite a `-p` suffix that means
"probability" everywhere else in that library. Nothing on this side carries that suffix.

**`+inf` is load-bearing, not a sentinel choice.** It absorbs under addition and sorts where
it means, which is what makes impossibility fall out of the arithmetic instead of needing a
branch. Do not normalise this objective into max-product "for consistency": that inverts the
recurrence and destroys the absorbing element in one move.

## Deviations from the legacy source

Six. Three are defects the port fixed, three are conventions it matched **on purpose** — and
the second kind is why this section exists at all. A convention matched deliberately leaves
no trace in a diff, in the code, or in the tests, so without a record here a later cleanup
"simplifies" it away.

**The recurrence below states this repository's behaviour, not the original's.** Where the
port fixed a defect, the fix is the specification; writing the original's behaviour into the
recurrence would specify the bug and make every later phase reproduce it.

### Fixed — the recurrence specifies the corrected behaviour

1. **The δ-seeding defect.** The original seeds `delta[0][j]` with the raw `init-state-p[j]`
   — a probability — into an accumulator holding bits everywhere else
   (`hmm-trainer.lsh:216-218`; `HMMLIB-ACCOUNT.md` §7, marked *provenance unknown*). Since
   smaller is better here, that inverts the preference among start states; and an exactly
   zero `init_p` seeds `0.0`, the *best* possible value, where it should be impossible.
   Here `delta[0] = bits(init_state_p)`.

   **Demonstrated causally, 2026-09-09, not merely inferred.** Substituting the defective
   seed into this implementation reproduces the original's choice exactly — start state
   **0** — while the corrected seed gives **5**. So the single divergence recorded under
   *Differential oracles* is produced by this deviation on demand, rather than being
   consistent with it. On `m008_0001_008` at the first symbol (code 6): `init_p[0] =
   0.3665`, `init_p[5] = 0.6335`, and the original prefers the *less* probable one.

   Two margins are involved and they are easy to conflate. The **arc margin** — the gap
   between the two best outgoing arcs — is **0.004218 bits** (`1.594225` against
   `1.598444`), and it is small enough that the seed alone decides the outcome. The **seed
   difference** is **0.789531 bits**. Only the first is the "0.004 bits" figure quoted in
   `docs/agents/core.md`.

   **The degenerate half is masked by these models and is not guaranteed.** Every state with
   `init_p == 0` in the tracked models also cannot emit `begin` on any outgoing arc, so
   `+inf` absorbs before the `0.0` seed can win. Revision 04's `split-state` halves initial
   probabilities without halving a topology, which decouples them — so the case the fixtures
   cannot exhibit is the one the next revision constructs.

2. **The `-1` log-zero sentinel is not reproduced.** The original represents log-zero as
   `-1`, requiring `safe->--log` at every comparison. Here `bits(0)` is `+inf`, which
   dissolves `safe->--log` into plain `>` and `int-delta` into `np.eye`. Faithfulness was
   declined as *uncheckable*: there is no Lush runtime in this repository, and the sentinel
   reaches no persisted artifact.

3. **`psi` as a float matrix is not reproduced.** `HMMLIB-ACCOUNT.md` §7's second defect is
   `psi` declared `float-matrix` and round-tripping state indices through it, exact only
   below `2**24` states. Here `psi` is `int64` (`STATE_DTYPE`). Note this defect is
   *harmless* in the original — every index below `2**24` is exactly representable — which
   is why it is recorded as not-reproduced rather than as a fix.

### Matched on purpose — no diff records these

4. **The tie-breaking rule.** The original's update is guarded by a strict "candidate is
   better than current" test, so an equal candidate never displaces an earlier one — first
   index wins. `np.argmin` returns the first minimal index, which is the same rule. **This
   is contract, not implementation detail** ([ADR 0003](../../adr/0003-one-parameterized-test-suite-per-algorithm.md)):
   two correct backends that break ties differently will disagree, and the suite meant to
   prove their equivalence will fail on a difference neither got wrong.

5. **The `N + 1` path geometry.** A path over `N` symbols visits `N + 1` states, because a
   symbol is emitted while *crossing* an arc. The original printed this too, as
   `save-viterbi-path`'s leading `-` row. Matched, not invented.

6. **The kernel neither validates nor raises.** An impossible sequence returns
   `total_bits == +inf` and a meaningless path; the *wrapper* turns that into
   `ImpossibleSequenceError`. This is an adaptation rather than a deviation — every later
   phase implements the same signature and a CUDA device function cannot raise a Python
   exception, so keeping the kernel purely numeric is what leaves phases 2-4 a
   transliteration.

## Source Implementation

`update-viterbi-path`, `.scratch/hmm-lush/Code/HMMlib/hmm-trainer.lsh:216-227`, with
`save-viterbi-path` at `:712-727` producing the oracles. The tree is present in this
repository, so this rendering was checked against a second text rather than written from
recollection. `HMMLIB-ACCOUNT.md` is that second text and is authoritative for what the
original *did*; this document is authoritative for what this repository does.

## Definitions

| Name | Shape | Meaning |
|------|-------|---------|
| `S` | scalar | number of states |
| `A` | scalar | vocabulary size — the **whole** vocabulary, reserved codes included |
| `N` | scalar | number of symbols in the record |
| `init_p` | `(S,)` | initial state distribution; sums to 1 |
| `trans_p` | `(S, S)` | `trans_p[i, j]` = probability of the arc `i -> j`; rows sum to 1 |
| `out_p` | `(S, S, A)` | `out_p[i, j, k]` = probability of emitting `k` while crossing `i -> j`. **Indexed by both endpoints**; the six reserved fibres are exactly zero. |
| `codes` | `(N,)` | integer codes, `0 <= codes[t] < A`, pre-encoded by the caller |
| `delta` | `(N+1, S)` | `delta[t, j]` = least description length of any path reaching state `j` after `t` symbols |
| `psi` | `(N+1, S)` | `psi[t, j]` = the predecessor state achieving `delta[t, j]`. Row 0 is never read. |
| `bits(p)` | — | `-log2(p)`; `bits(0) = +inf`. The only place the domain changes. |

## Recurrence

**Boundary note.** Everything below operates on **pre-encoded integer inputs**. Encoding and
decoding happen at the caller's boundary; no symbol operation occurs here, and `codes[t]`
indexes `out_p`'s third axis directly with **no offset arithmetic anywhere**. That is what
makes phases 2-4 transliterations rather than rewrites.

    arc_bits(i, j, t) = bits( trans_p[i, j] * out_p[i, j, codes[t]] )

    delta[t, j] = min over i of ( delta[t-1, i] + arc_bits(i, j, t) )
    psi[t, j]   = argmin over i of ( same ),  ties resolved to the SMALLEST i

**The emission factor cannot be hoisted.** `arc_bits` depends on both `i` and `j`, so it
cannot be lifted out of the inner loop the way `B[state, symbol]` can in the state-emission
formulation every textbook and every library uses. An implementation that hoists it has
silently changed the model.

## Base Cases

    delta[0, j] = bits(init_p[j])      for all j
    psi[0, *]   = undefined, never read

`delta[0]` is in the **bit domain** — this is deviation 1, and it is the whole of the fix.

## Iteration Order

    for t in 1 .. N:                 # strictly sequential: delta[t] needs delta[t-1]
        for j in 0 .. S-1:           # independent within a fixed t
            for i in 0 .. S-1:       # the reduction

## Backtrace

    states[N] = argmin over j of delta[N, j]        # ties to the smallest j
    for t in N-1 down to 0:
        states[t] = psi[t+1, states[t+1]]

    total_bits = delta[N, states[N]]

`states` has length `N + 1`. `states[t]` is the state occupied **before** symbol `t` is
emitted.

## Parallel decomposition

**States within one timestep.** Settled 2026-09-10 at phase 3, against the landed kernels,
as [ADR 0016](../../adr/0016-numba-cpu-parallel-phase.md) deferred it to be.

`prange` over `j`; the reduction over `i` stays serial and ascending. For fixed `t` all `S`
values of `j` are independent — each reads only `delta[t-1, :]` and writes only
`delta[t, j]` and `psi[t, j]` — so the parallel loop is race-free by construction rather
than by scheduling discipline.

**The `i` axis must not be the parallel one, and the reason is contract rather than
performance.** `argmin` over `i` returns the *first* minimal index, and a parallel reduction
combines partials in an unspecified order, so a tie between `i = 2` and `i = 7` may resolve
to either. That breaks the tie-breaking rule below **silently**: the `.vpath.xls` oracles
contain 0 exact ties in 3804 positions, because learned float parameters do not collide, so
no differential test can catch it. The constructed uniform-model case is the only guard, and
it exists because `rand_p_vector(size, noise_width=0)` ties at every position.

**This recurrence has no anti-diagonals, and here the anti-diagonal is worse than absent —
it is a race.** In an alignment matrix `{i + j = c}` is an independent set *because* a cell
reads only its three neighbours. This array is time by state, and `delta[t, j]` reads
**every** `delta[t-1, i]`. So the anti-diagonal `{t + j = c}` holds both `(t, j)` and
`(t-1, j+1)`, and `delta[t, j]` reads `delta[t-1, j+1]` as its `i = j+1` term: two cells on
one anti-diagonal with a direct dependency between them. The decomposition
[ADR 0002](../../adr/0002-three-phase-algorithm-lifecycle.md) calls "the same transformation
for every DP kernel in the family" does not merely go unchosen here — applying it would be
the defect. That claim is withdrawn for this family in ADR 0016's `Resolved` section.

**The parallelism is thin, and that is accepted rather than overlooked.** `S` is 5 and 8 in
the tracked fixtures and on the order of 50 at the ceiling, so `prange` over `j` offers at
most `S`-way parallelism over an `S`-element reduction, and per-timestep thread overhead may
exceed the work. This kernel may be slower than phase 2. ADR 0016 scopes phase 3 to parallel
*correctness* — validating a decomposition under real concurrency before CUDA — so a phase-3
kernel that loses on wall-clock is within what the phase is for.

The two decompositions not taken:

- **A batch of independent sequences.** The best speed story, and genuinely embarrassingly
  parallel, but unavailable at this signature: `viterbi(params, record)` takes one record,
  and batching arrives with revision 03.
- **An associative scan over time** in the min-plus semiring, since `(min, +)` matrix
  "multiplication" is associative. Costs `O(N·S³)` against `O(N·S²)`, does not produce
  `psi`, and re-associates the comparisons — so it forfeits bit-exactness with phases 1-2
  and would need its own equivalence argument rather than a differential test.

**The tie-breaking rule belongs here as much as in the recurrence**: smallest index wins, at
both the recurrence and the final `argmin`. Two backends disagreeing on ties will fail an
equivalence suite on a difference neither got wrong.


## Differential oracles

**Three, and they are the only evidence about this algorithm that is not downstream of code
written in this repository.**

1. **What produced them.** `save-viterbi-path` (`hmm-trainer.lsh:712-727`) wrote a
   `<model>.vpath.xls` beside each saved model during the original's own training runs:
   one `Output / States / Entropy` row per position, plus a leading `-` row for the state
   before the first symbol. Files:
   `.scratch/hmm-lush/Training/set02a/set02a_200/{m001_0001_001,m001_0005_005,m008_0001_008}.vpath.xls`.

2. **Which inputs they pair with.** The corpus at `set02a_200.sds`, whose 200 `.seq` files
   concatenated reproduce each oracle's `Output` column **exactly** — that check is what
   establishes the pairing, and it is TC-01. The models themselves are the three tracked
   `.hmm` directories. Inputs and outputs are tracked together; without that the files
   would be numbers.

   **A saved model is not loadable without the ADR 0011 renumbering.** Lush's user symbols
   start at code 2, so the fixtures' `(S, S, 6)` `output_p` becomes `(S, S, 12)` here,
   placed at `[..., USER_BASE:]`. Every derived quantity is invariant under that.

3. **It is not an equality check.** The port deliberately fixed the δ-seeding defect
   (deviation 1), so the oracle is *wrong* exactly where the fix bites, and the two are
   **expected** to disagree there.

4. **The divergence, pinned exactly.** Measured 2026-09-09 against all three models over the
   full 1269-position corpus:

   | Model | Agreement | Divergence |
   |---|---|---|
   | `m001_0001_001` | 1269 / 1269 | none |
   | `m001_0005_005` | 1269 / 1269 | none |
   | `m008_0001_008` | 1268 / 1269 | **position 0 only** |
   | **Total** | **3806 / 3807** | one position |

   The cause is deviation 1, demonstrated by substitution rather than inferred. **Pin this;
   never widen the assertion to cover it** — "agrees at 3806 of 3807, differing only at
   position 0 of `m008_0001_008`, because the seeding defect was fixed" is enforceable and
   checkable. "Mostly agrees" silently accepts the *next* divergence too.

5. **Carried by TC-02 through TC-05.**

**A green differential suite is evidence about this corpus, not about this algorithm.**
Measured 2026-09-09: **0 exact ties in 6081 live recurrence cells**, because learned float
parameters do not collide. So the differential tests alone would accept a last-wins
tie-break. That is why TC-06 is constructed rather than drawn from data.

## Test Cases

Provenance is recorded per case: `oracle` (an output the legacy implementation produced),
`extracted` (from a test the implementation already passes), or `constructed` (written from
the recurrence). A case extracted from a passing test is one the implementation already
satisfies, so a reader must be able to tell a specified behaviour from a ratified one.

### TC-01: the corpus reproduces each oracle's Output column — `oracle`, relational
    Input:  set02a_200 corpus, concatenated; each .vpath.xls
    Expect: Output column reproduced exactly. This establishes the input/output pairing;
            without it the other differential cases prove nothing.

### TC-02: the decode reproduces the original's path after the seed — `oracle`, relational
    Input:  each of the three models + corpus
    Expect: agreement at every position t >= 1

### TC-03: the divergence is exactly one position — `oracle`, relational
    Input:  m008_0001_008 + corpus
    Expect: 1268/1269 agreement; the sole difference at position 0

### TC-04: the divergence prefers the more probable start state — `oracle`, exact
    Input:  m008_0001_008, first symbol code 6
    Expect: we choose state 5 (init_p 0.6335); the oracle chose 0 (0.3665).
            Substituting the raw-probability seed reproduces the oracle's 0.

### TC-05: the original's path entropies are reproduced — `oracle`, relational
    Input:  each model + corpus
    Expect: state_entropies[path.states] matches the Entropy column, giving the
            oracle's third column an independent fourth check

### TC-06: exactly-tied predecessors resolve to the smallest index — `constructed`, exact
    Input:  a uniform model (rand_p_vector with noise_width=0) — ties at EVERY position
    Expect: the smaller state index wins, at both the recurrence and the final argmin.
            Not drawn from data: 0 ties in 6081 live cells means real data cannot test it.

### TC-07: the total is the sum of the arcs the path crossed — `extracted`, relational
### TC-08: the decoded path is the cheapest of all paths — `extracted`, exact
    Input:  a model small enough to enumerate all S^(N+1) paths
    Expect: brute-force minimum equals the decode's total_bits

### TC-09: emission depends on the destination as well as the source — `extracted`, exact
    Expect: two models differing only in out_p[i, j2, k] produce different decodes.
            Fails for any implementation that hoists the emission factor.

### TC-10: a path over N symbols visits N+1 states — `extracted`, property
### TC-11: an empty record still visits one state — `extracted`, exact
    Input:  codes = []
    Expect: states has length 1; total_bits = min(bits(init_p))

### TC-12: a record carrying UNK is impossible — `extracted`, exact
### TC-13: a record carrying PAD is impossible — `extracted`, exact
    Expect: ImpossibleSequenceError. Reserved fibres are exactly zero, so
            impossibility is arithmetic rather than convention.

### TC-14: an unseen bigram is impossible — `extracted`, exact
### TC-15: no infinite total ever reaches a caller — `extracted`, property
### TC-16: a code past the symbol axis is refused — `extracted`, exact
### TC-17: a negative code is refused — `extracted`, exact
    Note: -5 is a valid numpy index. The guard is explicit for that reason.

### TC-18: an impossible start state is never chosen — `extracted`, exact
    Expect: a state with init_p == 0 never begins the path

### TC-19: a code exactly equal to the vocabulary size — `constructed`, exact
    Input:  a record containing code A, where out_p has shape (S, S, A)
    Expect: ValueError, naming the range [0, A)
    Why:    the existing range tests use 99 and -5, both far from the edge, so a guard
            written `>` where it should be `>=` passes the entire suite while letting the
            boundary code through to the emission lookup.

    PIN THE MODEL: whether A-1 decodes is a property of the model, not of the guard.
    In the suite's `build` helper A = 8 and code(1) = 7 = A-1 is emittable, so A-1
    decodes and the contrast is decode-versus-refuse — the sharpest available. In a model
    that cannot emit A-1 it raises ImpossibleSequenceError instead, and a case written
    without pinning the model then passes or fails for an unrelated reason. What the
    `>`/`>=` slip actually changes is that code A stops being refused by the guard and
    reaches `output_p[:, :, A]`.

    Implemented 2026-09-09 as `test_a_code_exactly_at_the_end_of_the_symbol_axis_is_the_boundary`.
    Mutation-tested: with the guard weakened to `>`, the case fails with IndexError at
    `_viterbi.py:98` — numpy's, not the ValueError the contract promises.

### TC-20: impossibility at position 0 — `constructed`, exact
    Input:  a record whose FIRST symbol cannot be emitted from any reachable state
    Expect: ImpossibleSequenceError naming symbol index 0
    Why:    every other impossibility test places the dead symbol at position 1 or later,
            so `_dead_symbol`'s `enumerate` is never checked at its own first iteration.

    Implemented 2026-09-09 as `test_impossibility_is_reported_at_position_zero`.
    Mutation-tested: with `enumerate(codes, start=1)`, the case fails.

## Notes

- **This document was recovered from an implementation, not written before one.** The
  implementation is therefore *evidence* for the specification rather than a realisation of
  it, and the two can only disagree in one direction: where they differ, this document is
  wrong until someone establishes otherwise. That is the opposite of the forward entrance
  and worth remembering when reading the `extracted` test cases — they are ratified
  behaviour, not specified behaviour.
- **`Derived from` decides staleness.** When `_viterbi.py`'s SHA-256 no longer matches the
  recorded hash, this document is stale and must be re-derived rather than patched.
- **TC-19 and TC-20 were specified here before they existed**, found by drafting this
  document against the code, and implemented the same day — the suite went 280 to 282. Both
  were **mutation-tested rather than merely run**: a test that passes says nothing about
  whether it would fail on the defect it names. Weakening the range guard to `>` fails
  TC-19; starting `_dead_symbol`'s `enumerate` at 1 fails TC-20.
- **TC-19's premise took two corrections**, which is worth recording because the stable
  form is narrower than either draft. The source note said "code 7 decodes, code 8 raises";
  a constructed model showed both raising; the resolution is that *both are right about
  their own model*, and the case is only meaningful once the model is pinned. A boundary
  assertion that does not say which model it holds in is not yet a specification.
- **Phase 3 must not invent a decomposition.** See *Parallel decomposition*; "undetermined"
  is the answer until a kernel that exists settles it.
