# feat/hmm-viterbi-python

**Status**: merged — PR #17 — 2026-09-04
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

- [x] Decide the δ-seeding defect: fix it, or reproduce it faithfully
  > **Done:** Fixed, and the choice was settled by measurement rather than argument
  > because **the original's own decodes turned out to be on disk**. That is the finding
  > that reshaped this goal, and it reaches forward into the next two.
  >
  > **`save-viterbi-path` wrote `<model>.vpath.xls`** (`hmm-trainer.lsh:712-727`) — one
  > `Output / States / Entropy` row per position, plus a leading `-` row for the state
  > before the first symbol, which is ADR 0015's N+1 geometry printed to disk. All three
  > tracked models have one, and the corpus is beside them at `set02a_200.sds`. So the
  > port has a **decode oracle**, and this branch never has to validate its kernel against
  > itself. That answers goal 3's third subgoal in advance, in the affirmative.
  >
  > **The min-sum translation already reproduces the original at 1269/1269 positions** on
  > two of three models, and 1268/1269 on the third — measured before writing `_viterbi`,
  > using goal 1's `load_params` plus a sixty-line probe. The kernel's *correctness* is
  > therefore established before goal 3 opens; what goal 3 owes is its packaging.
  >
  > **The defect propagates nowhere**, which is why fixing it is cheap. `run-add` is
  > genuine Baum-Welch over α/β/ξ, and the MDL score comes from `update-data-dl`, a clone
  > of the *forward* pass. Grepping the tree, `path-states`/`path-entropy` are read only by
  > `save-viterbi-path` and `seq-state`'s own printer. The decode is annotation-only, so
  > revision 03's training and revision 04's search are untouched by the choice.
  >
  > **The corpus recovers the symbol names the model lost.** `set02a_200.sds/_alphabet`
  > reads `0→begin, 1→end, 2→c, 3→d, 4→b, 5→a`, where the same filename inside a `.hmm`
  > directory holds Lush pointer addresses. Goal 1 recorded "the symbol names are gone";
  > that was true of the model directory and false of the repository. The sharpened form is
  > better evidence for the same conclusion: persistence split the mapping from the model,
  > so the model alone stops denoting — ADR 0001's cost clause, demonstrated rather than
  > argued.
  > **Q:** The δ-seeding defect is now measured: fixing it costs exactly one position out
  > of 3807, at `m008_0001_008` position 0, where our answer prefers the 63% start state
  > over the 37% one. How should `pfsmgraph.hmm` seed δ — fix it, reproduce it faithfully,
  > or fix it while keeping the original reachable behind a kernel flag?
  > **A:** Fix it. `delta[0] = bits(init_state_p)`, so the degenerate case becomes correct
  > by arithmetic — `bits(0)` is `+inf` and absorbs — which is the same move goal 1 made to
  > render reserved-symbol emission impossible. The differential test asserts full
  > agreement on positions 1..N across all three models, plus an explicit assertion that
  > position 0 diverges on `m008` *and* that our choice is the higher-probability start.
  > The seed-override option was declined: a parameter existing only to reproduce a bug is
  > one ADR 0002 phases 2–4 would each have to carry into Cython, Numba and CUDA.
  - [x] Weigh the two consequences separately (`HMMLIB-ACCOUNT.md` §7). The typical case is bounded by one bit and decides *backwards* within that band; the degenerate case turns an impossible start state into the best possible δ, and revision 04's `split-state`/`merge-states` can produce one where `init-random` cannot.
    > **Done:** Both weighed against numbers, and they came out asymmetrically.
    >
    > **The typical case fires exactly once in 3807 positions**, and it is the account's
    > description holding precisely. At `m008_0001_008` position 0 the two live start
    > states seed as `0.3665` / `0.6335` under the original and `1.4481` / `0.6586`
    > corrected, while their best outgoing arcs on `begin` differ by 0.004 bits — so the
    > seed alone decides it, and the original picks the 37% state over the 63% one. The
    > inversion is real, is confined to the one position the seed touches, and stays
    > inside the predicted one-bit band.
    >
    > **The degenerate case never fires, for a reason that is not a guarantee.** Every
    > state with `init_p == 0` in these models is *also* unable to emit `begin` on any
    > outgoing arc — 0 of 4 reachable in `m001_0005_005`, 0 of 6 in `m008_0001_008` — so
    > `+inf` absorbs and the δ = 0.0 seed can never win. **The learned topology masks the
    > worse defect.** That correlation is a property of trained models, not an invariant,
    > and revision 04's state surgery is exactly what decouples the two: `split-state`
    > halves initial probabilities without halving a topology. So the case the fixtures
    > cannot exhibit is the one the next revision can construct, which is an argument for
    > fixing it that the fixtures themselves cannot supply.
  - [x] Record the decision where the ADR 0003 suite encodes the choice rather than accidentally validating the bug against itself
    > **Done:** Recorded in `docs/agents/core.md` — following the precedent set for the
    > `-1` sentinel divergence, which was written there rather than given an ADR. The two
    > are not the same *kind* of divergence and the distinction is worth keeping: the
    > sentinel was a representation change with identical semantics and was declined as
    > *uncheckable*, whereas this one changes output and is now checkable. Had the vpath
    > files been found a revision earlier, the sentinel's justification would have needed
    > re-examining too.
    >
    > The structural half of this subgoal is the `.gitignore` widening: with the three
    > `.vpath.xls` and the corpus tracked, goal 3's test compares against **the original's
    > output**, so self-validation is not something the test has to be careful about — it
    > is unavailable. 210 files added to the tracked set, verified to exclude `.DS_Store`.
  - [x] Note `psi`'s float round-trip as decided-not-reproduced — the master plan already settled it as harmless below 2²⁴ states
    > **Done:** Not reproduced; `psi` is `np.int64`. Nothing was reopened — the master plan
    > settled it and the fixtures agree, at 5 and 8 states against a 2²⁴ bound. Recorded
    > alongside the seeding decision so the two §7 defects are answered in one place rather
    > than one being answered and the other looking overlooked. The author's own adjacent
    > note (`hmm-trainer.lsh:219`, "`(psi 0 state-j)` is never used, and is in fact
    > undefined") carries over intact: our `psi[0]` row is likewise never read, since the
    > backtrace stops at `psi[1]`.

- [x] Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy)
  > **Done:** `_viterbi.py` — the project's first DP kernel — with 55 tests; suite 204 →
  > 259. `pfsmgraph.hmm` exports four names where it exported one: `HMMParams`, `viterbi`,
  > `ViterbiPath`, `ImpossibleSequenceError`.
  >
  > **Goal 2 established the kernel's correctness before this goal opened, so what remained
  > was packaging — and both of this goal's real findings came out of the packaging rather
  > than the algorithm.** That is worth carrying into goal 4 and into revision 03: the
  > algorithm was the part already validated, and the seams around it were not.
  >
  > **Mutation testing falsified a comment I had written in the same session.** Six
  > mutations of the kernel; four caught. The tie-break mutation passed, which contradicted
  > the comment claiming that matching Lush's first-wins tie-break "is what makes a
  > position-for-position differential test possible at all" — there are **0 exact ties in
  > 3804 positions**, because learned float parameters do not collide. The property is real
  > but the fixtures cannot exercise it, and the case that will is revision 03's own
  > initialisation: `rand_p_vector(size, noise_width=0)` returns an exactly uniform vector,
  > which ties at every position. This is the same shape as goal 2's finding about the
  > degenerate seed — **the fixtures systematically cannot exhibit the cases the next
  > revision constructs**, which is now twice, and is an argument for constructing those
  > cases in tests rather than waiting for them.
  >
  > **The surviving `psi`-as-float mutation is the opposite finding and was left alone.**
  > Every state index below 2²⁴ is exactly representable in float64, so the mutation changes
  > nothing observable at any testable scale — which is precisely the master plan's reason
  > for judging that defect harmless. Writing a test that failed there would have asserted
  > something untrue.
  >
  > **A dead branch was found by smoke-testing and deleted rather than documented**: the
  > "no live start state" diagnostic cannot fire, because `HMMParams` requires
  > `init_state_p` to sum to 1. The case is real in the mathematics and unrepresentable in a
  > constructed model.
  >
  > One process note. `cp` is aliased to `cp -i` in this shell, so the first mutation run's
  > restores silently did nothing and four mutations accumulated into one file; the run
  > reported plausible-looking failures that meant nothing. `command cp` bypasses it, and a
  > `diff -q` against the backup after each restore is what turns a silent no-op into a
  > visible one.
  > **Q:** A record can be impossible under a model — every path costs `+inf` bits, which is
  > what `output_p[i, j, UNK] = 0` produces for an `on_unknown="unk"` record and what a dead
  > arc produces for an unseen bigram. The Lush original returns a garbage path silently: δ
  > stays at its `-1` sentinel, ψ stays at its `0` initialisation, so the backtrace walks
  > zeros. What should the public `viterbi` do — raise a dedicated error type, raise a plain
  > `ValueError`, or return a total function with `total_bits = inf`?
  > **A:** A dedicated `ImpossibleSequenceError(ValueError)`, naming the position at which
  > the last live state died and the symbol code that killed it. Subclassing `ValueError`
  > keeps `except ValueError` working, and the distinct type is what revision 04 needs: its
  > topology search decodes many sequences against many candidate topologies, where an
  > impossible sequence is an ordinary *search outcome* to be caught by type, not a
  > malformed input. Returning `inf` was declined for the reason goal 1 declined the
  > user-symbols-only axis — a confident-looking answer distinguishable from a real one only
  > by a check the caller has to remember.
  - [x] The private kernel `_viterbi(init_p, transition_p, output_p, codes)` — a **min-sum** over bits, not a max-product (`HMMLIB-ACCOUNT.md` §3), with `output_p[i, j, symbol]` unhoistable from the inner loop because it depends on both endpoints (ADR 0015)
    > **Done:** `_viterbi.py`, 1269/1269 · 1269/1269 · 1268/1269 against the three oracles,
    > the one mismatch being `m008` position 0 — the seeding divergence, exactly as goal 2
    > measured with its probe.
    >
    > **The kernel does not validate and does not raise**, and that is the backend contract
    > rather than an omission. Every later ADR 0002 phase implements this signature, and a
    > CUDA device function cannot raise a Python exception; an impossible sequence comes back
    > as `total_bits == inf` and the wrapper turns it into an error. Keeping the kernel purely
    > numeric is what leaves phases 2–4 a transliteration rather than a redesign.
    >
    > **The `(S, S, A)` precompute was refused.** Goal 2's probe hoisted
    > `bits(transition_p[:, :, None] * output_p)` out of the loop, which is fine for a
    > six-symbol probe and `O(S²·A)` for a real vocabulary. Worse, it *reads* like the
    > hoisted emission factor ADR 0015 forbids while not being one, so the reference forms a
    > fresh `(S, S)` per position instead. Phase 2 fuses it away regardless.
    >
    > **The tie-break matches by convention, not by design.** Lush guards its update with
    > `safe->--log` — strictly better — so an equal candidate never displaces an earlier one,
    > and `np.argmin` returns the first minimal index. Had the original used `>=`, a
    > position-for-position agreement would have been unreachable and the cause invisible.
  - [x] The public `viterbi(params, record) -> ViterbiPath` wrapper, taking one `SequenceRecord` and returning a result rather than writing back into it. `N` symbols visit `N+1` states; `ViterbiPath.states` is not decoded and `.label` is the only string that passes through.
    > **Done:** `viterbi`, `ViterbiPath` and `ImpossibleSequenceError`, all three exported —
    > `pfsmgraph.hmm` now has four public names where it had one. `ViterbiPath` carries
    > `states` (frozen, `int64`, `N+1` long), `total_bits` and `label`.
    >
    > **Per-position entropies are not carried**, though the original has a `path-entropy`
    > slot and the `.vpath.xls` oracle prints the column. It is
    > `params.state_entropies[path.states]` — one fancy-index, and derived, so ADR 0017's
    > "computed, never stored" applies. The differential test derives it, which turns the
    > oracle's third column into a **fourth independent check** rather than a stored copy.
    >
    > **A dead branch was found by smoke-testing and removed rather than documented.** The
    > diagnostic first distinguished "no live start state" from "symbol *t* killed it"; the
    > first cannot fire, because `HMMParams` requires `init_state_p` to sum to 1 and so
    > forbids an all-zero seed. The case is real in the mathematics and unrepresentable in a
    > constructed model — `_dead_symbol` now says so in its docstring instead of branching
    > on it.
    >
    > **The diagnostic is boolean, not numeric, and only runs on failure.** A path has finite
    > cost exactly when every arc it crosses has positive probability, since `bits` is finite
    > precisely on the positive reals — so a reachability sweep over `transition_p > 0` and
    > `output_p[..., code] > 0` has the same support as the min-sum and cannot disagree with
    > it.
  - [x] Establish whether the three tracked `.hmm` fixtures support a differential test of the decode, as they did for the stationary solve. Their four-decimal print format is a known trap — see PR #16.
    > **Answered in advance by goal 2: yes.** Each model has a `<name>.vpath.xls` written by
    > `save-viterbi-path`, and the corpus is at `set02a_200.sds`; both were tracked on
    > 2026-09-03. The min-sum recurrence already reproduces the `States` column at 1269/1269
    > on `m001_*` and 1268/1269 on `m008_0001_008` (position 0, the seeding divergence). The
    > four-decimal trap does *not* bite here — the `States` column holds integers. What
    > remains for this subgoal is building it into `tests/`, not establishing feasibility.
    >
    > **Done:** `test_viterbi.py`, 52 tests; suite 204 → 256. `_lush_fixtures.py` grew the
    > corpus and vpath readers (`load_corpus_codes`, `load_corpus_record`, `load_vpath`,
    > `corpus_alphabet`), so the oracle is shared rather than re-derived.
    >
    > **The suite was mutation-tested rather than trusted for passing**, and that changed
    > two things. Six mutations of the kernel: max-product, the seeding defect reinstated,
    > the emission losing its source index, an off-by-one backtrace — all four caught. Two
    > survived, and only one was a gap.
    >
    > **A claim I had written into the kernel was false.** The comment said matching Lush's
    > first-wins tie-break "is what makes a position-for-position differential test possible
    > at all". Measured: **0 exact ties in 3804 positions** across all three models, because
    > learned float parameters do not collide — so a last-wins port passes the differential
    > test unchanged. The tie-break is now pinned by a constructed uniform model, and the
    > comment says what is true: it matters for `rand_p_vector(size, noise_width=0)`, which
    > returns exactly `[0.2, …]` and is how revision 03 initialises. The fixtures cannot
    > exercise the case revision 03 will hit on its first call.
    >
    > **The surviving `psi`-as-float mutation is not a gap**, and the test now says so. Every
    > index below 2²⁴ is exactly representable in float64, so the mutation changes nothing
    > observable at any testable scale — which *is* the master plan's reason for judging it
    > harmless. A test that failed there would be asserting something untrue; what is pinned
    > instead is that a caller receives integers.
    >
    > Two further checks the oracle turned out to support, neither planned: the `Entropy`
    > column reproduces at `5e-4` from `params.state_entropies[path.states]` — the third
    > independent arrival of that same bound — and `total_bits` is asserted no worse than the
    > cost of the original's own path, which says the one disagreement is a strictly better
    > path rather than a coin toss.
  - [x] Decide how the fixture loader maps Lush's `begin`/`end` onto the ADR 0011 block, and record why. **The obvious answer is wrong in a way that fails loudly, which is the good case.** Lush codes 0 and 1 are `begin`/`end` — semantically BOS/EOS, which ADR 0011 puts at 2 and 3 — but `load_params` maps every Lush code through `USER_BASE + c`, making them user symbols 6 and 7. Mapping them onto BOS/EOS instead would put nonzero emission mass on reserved fibres, and goal 1's rule rejects that at construction, so the fixtures would stop loading. Whether the *library* should treat a corpus's own boundary markers as BOS/EOS is a separate question and belongs to `dataseq`, not here.
    > **Done:** `USER_BASE + c`, recorded in `load_params`'s docstring and **pinned by three
    > tests rather than left as prose**, because "it would fail loudly" is a claim about
    > behaviour and is cheap to check: the corpus names its first two codes `begin`/`end`;
    > building the fixture's `output_p` under the BOS/EOS mapping is refused by `HMMParams`
    > naming the reserved symbol; and the mapping actually used decodes the whole corpus to
    > a finite cost.
    >
    > The consequence is sharper than "the fixtures would stop loading". Every sequence in
    > the corpus *opens* with `begin`, so under the BOS/EOS mapping the reserved fibre it
    > reached would be zero, `bits(0)` would be `+inf`, and the entire corpus would be
    > reported impossible — the failure lands at symbol 0 of every sequence rather than
    > somewhere subtle. `begin` and `end` are ordinary user symbols *to this model*: it emits
    > them and scores them, which is what makes them user symbols regardless of what they are
    > called.
    >
    > Two stale claims in `_lush_fixtures.py` were corrected while here: its docstring cited
    > `SUM_TOL` as `1e-6` (goal 1 moved it to `1e-5`), and it still said the models' symbol
    > names were unrecoverable, which goal 2 disproved from the corpus.

- [x] Register `python` as the first ADR 0003 backend
  > **Done:** The matrix is no longer empty. `backends: python ✓` is what a run now opens
  > with, ending the `EMPTY_HEADER` era that began 2026-09-01. Suite 259 → 264.
  >
  > **The goal's real finding is that ADR 0003 cannot be fully satisfied here, and the
  > obstruction is in ADR 0003.** It asks for one suite per algorithm, parameterized over
  > backends, with tests written against the public API only; it also defers the
  > backend-selection API to `align`. Without that API the public surface has nowhere to put
  > a backend, so the two requirements are jointly unsatisfiable in revision 02. Recorded in
  > `_backends.py`'s docstring rather than worked around, and the cost is named there: until
  > the seam exists, the table says which phases *exist*, not which ones the suite exercises.
  >
  > That distinction is worth keeping, because the header now reads as a stronger claim than
  > it is. `backends: python ✓` means the kernel imports, not that anything ran twice.
  >
  > **Two smaller things fell out.** The row names the *kernel module* rather than the
  > package, because `import pfsmgraph.hmm` succeeds with or without a decode in it — a probe
  > that cannot fail is not a probe. And ADR 0003's Negative section turns out to have
  > anticipated goal 3's tie-break finding and to justify it more strongly: a tie-break is
  > **contract**, since two correct backends would otherwise legitimately disagree.
  > **Q:** ADR 0003 wants one suite per algorithm parameterized over backends, but its own
  > Open section defers the backend-selection API to `align` ("it warrants its own record"),
  > and `viterbi(params, record)` has nowhere to put a backend today. How far should this
  > goal go — register the row and record the tension, build the parameterized fixture over
  > a private kernel registry now, or add a public `backend=` parameter?
  > **A:** Register only, and record the tension. Add the `python` row, fix the root tests,
  > verify the one-backend mechanism end to end, and label the kernel-level tests in
  > `test_viterbi.py` as the "separate, explicitly non-shared home" ADR 0003 already
  > requires for a test that reaches one backend's internals. Nothing is guessed, and the
  > constraint is written where phase 2 will hit it. The public `backend=` parameter was
  > declined because it would settle ADR 0003's open question in revision 02, when that ADR
  > routes it to `align`; the private kernel registry was declined because it would make the
  > algorithm suite test `_viterbi` rather than `viterbi`, which is exactly what that ADR's
  > Negative section warns against.
  - [x] Add the row to `_backends.py`, with `hardware=None` so a failed import escalates rather than skips
    > **Done:** `Backend("python", "pfsmgraph.hmm._viterbi")`. The module named is the
    > **kernel, not the package**, and that is the substantive choice: `import pfsmgraph.hmm`
    > succeeds whether or not a decode exists in it, so the package would be a row that
    > cannot fail — and a row that cannot fail is not a probe. `hardware=None` because
    > nothing external is needed to run pure Python, so a failed import means a broken
    > working copy, which is exactly what ADR 0003 forbids concealing behind a skip.
  - [x] Update the repo-root backend tests that currently assert an empty matrix and the `EMPTY_HEADER` line
    > **Done:** Three failed as designed — `test_registry_is_empty_until_a_dp_kernel_lands`
    > carried a comment saying it *would* fail when `align` or `hmm` added the first row,
    > "which is exactly when the surrounding docs need revisiting". It did, and they were.
    > Its replacement asserts the exact one-row tuple and carries the same forward warning
    > for the second row.
    >
    > `EMPTY_HEADER` is **kept under test** rather than deleted with its last caller: the
    > branch stays live, and ADR 0003 requires that an empty matrix say so in as many words,
    > because a missing line is indistinguishable from a hook that was never registered.
    > Five tests added rather than three changed — the extra two assert against the *real*
    > matrix rather than a synthetic one: that the registered module actually resolves, and
    > that `PFSMGRAPH_REQUIRE_BACKENDS=python` passes, which is what a CI runner does.
    > Suite 259 → 264, and the header now reads `backends: python ✓`.
  - [x] Confirm the parameterized suite shape works with one backend, since backend *equivalence* has nothing to compare against until phase 2
    > **Done, and the answer is that it does not work yet — for a reason in ADR 0003's own
    > text.** The mechanism half is confirmed end to end: the header prints
    > `backends: python ✓`, `detect()` resolves the row, and the escalation now names a real
    > registry (`registered: ['python']` where it said `none`).
    >
    > The suite half cannot be built. ADR 0003 wants the backend as a fixture parameter with
    > tests "written against the public API only" — but `viterbi(params, record)` has nowhere
    > to put a backend, and giving it one is the runtime backend-selection API that the same
    > ADR's Open section routes elsewhere: "settle this when `align` acquires a
    > backend-selection API; it warrants its own record." **So the two halves of ADR 0003
    > cannot both hold until `align`**, and revision 02 is not the place to settle it.
    >
    > What was done instead is the part ADR 0003 asks for unconditionally: the two tests that
    > call `_viterbi` directly now sit in a labelled section, the "separate, explicitly
    > non-shared home" that ADR's Negative section requires for a test reaching one backend's
    > internals. The constraint is written where phase 2 will hit it, in `_backends.py`'s
    > docstring and in that section's comment.
    >
    > **ADR 0003 also retro-justifies goal 3's tie-break test, more strongly than goal 3
    > did.** Its Negative section: "Ties and other under-specified outcomes must be pinned
    > down. Where a DP traceback has multiple optimal paths, the algorithm's tie-breaking
    > rule becomes part of the contract, because otherwise two correct backends legitimately
    > disagree." Goal 3 justified that test as mutation coverage; it is **contract**, and a
    > Cython wavefront breaking ties the other way would be a correct backend giving a
    > different answer.
