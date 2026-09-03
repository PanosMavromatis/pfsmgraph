# feat/hmm-viterbi-python

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy) with the ADR 0003 test suite, and register it as the first backend (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Implement `HMMParams`, the ADR 0017 frozen parameter value
  > **Done:** `_params.py` and 44 tests; the package exports something for the first time.
  > Suite 160 → 204. Four things worth carrying forward beyond the subgoal notes.
  >
  > **The validation boundary PR #16 deferred here is now drawn**, and two of its rules are
  > decisions rather than mechanics. A **zero `transition_p` row is rejected**, with no
  > exemption: §5 records that `merge-states` divides with `safe-/`, so merging two
  > unreachable states yields zeros, and revision 04 can therefore construct one — it
  > should learn that at construction, where it has to decide what an unreachable state
  > means, rather than downstream in a stationary solve that has no answer. Conversely an
  > emission fibre on a **dead arc is not checked**, because `bits(0)` on the transition
  > absorbs the path and the fibre cannot reach the recurrence. That exemption turned out
  > to be load-bearing rather than theoretical: the original's own saved models are full of
  > all-zero fibres on zero-probability arcs, and a blanket rule would have rejected them.
  >
  > **`SUM_TOL` was wrong at `1e-6` and is `1e-5`.** The lower bound, not the upper, is what
  > bit: `float32` eps is `1.19e-7`, so a vector normalized in float32 drifts past `1e-6`
  > over a symbol axis of a few dozen — and that consumer is named in ADR 0017's own
  > Negative section, where revision 03's `torch` backend returns a new frozen value from a
  > float32 step. Caught by a test asserting the bound's own justification, which is why the
  > justification is worth asserting.
  >
  > **`tests/_lush_fixtures.py` extracted**, and the trigger fired a revision early. The
  > reader's docstring in `test_numeric.py` said "when revision 03's differential tests want
  > it too, that is the moment a shared fixture earns itself"; the second consumer arrived
  > in revision 02. Extracting rather than copying applies §13's own finding — near-
  > duplicate code is the migration's most likely hiding place for a translation error.
  >
  > **The saved models' symbol names are unrecoverable.** `_alphabet` holds Lush pointer
  > addresses (`#$11F50E0`), not names, so only the codes survive. Independent confirmation
  > of ADR 0001's cost clause and of `DEFERRED.md`'s serialization trigger: a persisted
  > model without its mapping stops denoting anything.
  > **Q:** How should the `A` axis of `output_p` be sized, and how much should `HMMParams`
  > enforce about it — `vocab.size` with the reserved fibres required to be zero,
  > `vocab.size` with them unconstrained, or user symbols only with a `USER_BASE` offset in
  > every kernel?
  > **A:** `vocab.size`, with the six reserved fibres required to be exactly zero at
  > construction. Direct indexing `output_p[i, j, codes[t]]` with no offset anywhere, so
  > ADR 0002's phases 2–4 have no index arithmetic to port; and emitting a reserved symbol
  > becomes impossible by arithmetic rather than by convention, since `bits(0) = +inf`
  > absorbs under addition. The decisive case is `UNK`: it is the one reserved code a
  > record acquires on a documented `dataseq` path (`encode(..., on_unknown="unk")`), and
  > under the user-symbols-only option `1 - USER_BASE = -5` is a valid negative index that
  > numpy accepts silently — a confident wrong path. Cost is `6 · S²` dead entries,
  > 120 KB at `S = 50`, scaling with `S²` rather than with the corpus.
  - [x] Settle the `A` axis — `vocab.size` (reserved block included, so the emission tensor carries fibres for `PAD`/`UNK`/… that must never be emitted) or only the user symbols. PR #15 surfaced this and deliberately did not decide it, because it changes the `(S, S, A)` shape and so belongs with the kernel rather than with the boundary.
    > **Done:** `vocab.size`, reserved fibres enforced zero at construction — see the Q&A
    > above. The decision paid off immediately in an unplanned place: the tracked `.hmm`
    > fixtures store `output_p` as `(S, S, 6)` against Lush's `_alphabet_size`, whose user
    > symbols start at code 2, so under ADR 0011 the same six symbols are codes 6–11 and
    > `A` is **12, not 6**. A saved model is therefore not loadable without the
    > renumbering, which `tests/_lush_fixtures.py::load_params` performs. Every derived
    > quantity is invariant under it: `state_p` never reads `output_p`, and zero-padding a
    > symbol axis adds only `0·log2(1) = 0` terms to an entropy.
  - [x] Freeze the three arrays with `writeable = False` and retain the `Vocabulary` as `dataseq`'s `Protocol`, held for `size` and for identity comparison — not for any symbol string
    > **Done:** Plus one thing ADR 0017 does not say and needs: **the cached arrays are
    > frozen too.** A `cached_property` hands back the same object on every access, so a
    > caller who writes into `state_p` corrupts the value every later reader sees — the
    > stored-and-stale failure the ADR claims becomes unrepresentable, reintroduced one
    > level out. Freezing the inputs alone does not deliver the claim. Also asserted, as an
    > implementation detail rather than a guarantee: `cached_property` works on a frozen
    > dataclass only because it writes through `instance.__dict__` rather than
    > `__setattr__`, and would break under `slots=True`.
  - [x] `state_p` and the state entropies as cached properties over `_numeric.py`: `stationary_distribution(transition_p)`, and `entropy` of the `(S, A)` marginal `np.einsum("ij,ijk->ik", transition_p, output_p)`. ADR 0017 makes these derived rather than stored, so a reducible chain surfaces its `ValueError` on an attribute access.
    > **Done:** Three properties, since `update-entropy` (`hmm.lsh:228-262`) computes three:
    > `state_p`, `state_entropies`, and the model `entropy` as `state_p @ state_entropies`.
    > All three are checked differentially against all three saved models' own numbers.

- [ ] Decide the δ-seeding defect: fix it, or reproduce it faithfully
  - [ ] Weigh the two consequences separately (`HMMLIB-ACCOUNT.md` §7). The typical case is bounded by one bit and decides *backwards* within that band; the degenerate case turns an impossible start state into the best possible δ, and revision 04's `split-state`/`merge-states` can produce one where `init-random` cannot.
  - [ ] Record the decision where the ADR 0003 suite encodes the choice rather than accidentally validating the bug against itself
  - [ ] Note `psi`'s float round-trip as decided-not-reproduced — the master plan already settled it as harmless below 2²⁴ states

- [ ] Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy)
  - [ ] The private kernel `_viterbi(init_p, transition_p, output_p, codes)` — a **min-sum** over bits, not a max-product (`HMMLIB-ACCOUNT.md` §3), with `output_p[i, j, symbol]` unhoistable from the inner loop because it depends on both endpoints (ADR 0015)
  - [ ] The public `viterbi(params, record) -> ViterbiPath` wrapper, taking one `SequenceRecord` and returning a result rather than writing back into it. `N` symbols visit `N+1` states; `ViterbiPath.states` is not decoded and `.label` is the only string that passes through.
  - [ ] Establish whether the three tracked `.hmm` fixtures support a differential test of the decode, as they did for the stationary solve. Their four-decimal print format is a known trap — see PR #16.

- [ ] Register `python` as the first ADR 0003 backend
  - [ ] Add the row to `_backends.py`, with `hardware=None` so a failed import escalates rather than skips
  - [ ] Update the repo-root backend tests that currently assert an empty matrix and the `EMPTY_HEADER` line
  - [ ] Confirm the parameterized suite shape works with one backend, since backend *equivalence* has nothing to compare against until phase 2
