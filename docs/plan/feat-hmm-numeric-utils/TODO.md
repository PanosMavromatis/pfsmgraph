# feat/hmm-numeric-utils

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Migrate the Utility code this release needs, private to the package (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Read `Code/Utility/util.lsh` and settle where the migrated code lands
  > **Done:** The shape of the port is settled, and it is smaller than the subgoal
  > assumed: **four functions, not six**, since `int-delta` dissolves into `np.eye`
  > and `safe->--log` into `>`. One `_numeric.py` holds them. The `+inf` decision was
  > verified rather than reasoned — `np.log2(0.0)` is `-inf`, so `total - log2(p)`
  > yields `+inf` and absorbs, while `inf > 3` is `True` and `3 > inf` is `False`.
  >
  > The goal grew a fifth subgoal, `Execute the tracking decision`, because the
  > tracking question turned out to have an execution half worth showing: the
  > `.gitignore` widening is where a mistake would be silent.
  >
  > **The unlooked-for result is that goal 3's test got much stronger.** A saved
  > `.hmm` directory persists `transition_p` next to `state_p` and `output_p` next
  > to `state_entropies` — the inputs *and* outputs of both computations this branch
  > ports — in a plain-ASCII format needing no model code to read. Measured on all
  > three fixtures before tracking them: `A = Pᵀ - I; A[0,:] = 1; b = e₀` reproduces
  > every saved `state_p` to 5e-5, the print format's own rounding. So goal 3's test
  > is a **differential test against the original's own numbers**, not the
  > closed-form two-state chain the plan proposed, and the stationary solve is
  > effectively validated before it is written.
  >
  > That also answers a risk worth naming: [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md)
  > exists partly because Lush refreshes derived quantities by a manual
  > `update-entropy` call, so a saved `state_p` *could* have been stale against its
  > own `transition_p`. On these three it is not.
  > **Q:** What becomes of the `-1` log-zero sentinel in the port?
  > **A:** Replaced by `+inf`. IEEE-754 reproduces every property the sentinel was
  > hand-built for: absorption (`x + inf = inf`), the comparator (`inf > y` true,
  > `y > inf` false, `inf > inf` false), and the `< 0` test at `hmm-trainer.lsh:430`.
  > So `safe->--log` dissolves into plain `>` and `safe-add--log2` into
  > `total - np.log2(p)`. The decisive point is that faithfulness here is
  > **uncheckable**: there is no Lush runtime in this repository, and the sentinel
  > reaches no persisted artifact — it lives only in the runtime accumulators
  > `data-p`, `result-p` and δ, and even `_total_dl` is written after
  > `update-total-dl` maps the sentinel to `1e100`. Cost accepted: a deliberate
  > `np.errstate` around `log2(0)`, and `inf - inf = nan` is unreachable here only
  > because probabilities never exceed 1 (so `log2(p) ≤ 0`) — worth a comment at the
  > site rather than a guard.
  > **Q:** Where does the migrated code land?
  > **A:** One `_numeric.py`. Four to six small functions and one test module;
  > splitting ahead of a second consumer would invent structure, and
  > [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md)'s `Open`
  > section — where revision 04 may change the parameter representation — is where a
  > `_linalg.py` would earn itself. The marginalization
  > `p_i(k) = Σⱼ transition_p[i][j]·output_p[i][j][k]` (§4) is **not** in scope for
  > this module: it is model-shaped and belongs with `HMMParams`, which calls
  > `entropy` on its result.
  > **Q:** The tracked set only ever widens, and this is the cheap moment. What joins
  > it under `.scratch/hmm-lush/`?
  > **A:** `Code/Utility/C/util.c` and a few saved `.hmm` model directories; **not**
  > `util.lsh copy`. The generated C is the only machine-checked statement of what
  > these primitives mean in a repository with no Lush runtime, and it had already
  > earned its place before the question was put. The saved models persist the inputs
  > *and* outputs of both computations this branch ports. The abandoned draft is
  > declined on §13's own grounds — a near-identical 574-line file beside `util.lsh`
  > is the "similar enough to skim and different enough to matter" hazard, and its one
  > finding is recorded in prose below.
  - [x] Read the six migrating functions and the LU trio in their own terms, against `HMMLIB-ACCOUNT.md` §3 and §4
    > **Done:** Read whole (574 lines). Seven findings that change what gets written:
    >
    > 1. **Two of the six named functions dissolve rather than migrate.** `int-delta`
    >    (`417-422`) is a Kronecker delta with exactly 2 call sites, both inside the
    >    stationary solve, where it builds `(Pᵀ - I)`; in numpy that is
    >    `transition_p.T - np.eye(S)` and the function has no remaining purpose.
    >    `safe->--log` dissolves into `>` under the `+inf` decision above. What
    >    actually gets written is four functions, not six.
    > 2. **Viterbi is min-sum over bits, not max-product over probabilities** (§3).
    >    `safe->--log(x, y)` reads as "x greater than y" and *means* "`y` is strictly
    >    better than `x`", because smaller bits are higher probability. Recorded here
    >    rather than in the Viterbi subgoal because it is a property of the comparator
    >    being ported now: a port that reaches for `max` inverts every comparison.
    > 3. **The `-p` suffix lies in this one place.** `data-p` and `result-p` are
    >    accumulated with `safe-add--log2` and hold description lengths, not
    >    probabilities (§3). The Python names must not carry `_p`.
    > 4. **`rand-p-vector` overwrites; it does not perturb.** `util.lsh:529-530`
    >    assigns `1 + noise_width·rand(-1,1)` to each element, discarding the input
    >    array's contents; the perturbing variant at `531-533` is commented out as "it
    >    doesn't work very well". Confirmed in the generated C — a plain
    >    `IDX_PTR(...)[...] = (1+(noise_width*rand))`. So the Python signature takes a
    >    size and returns a new array rather than taking an out-parameter.
    > 5. **`(rand 1.0 -1.0)` is uniform on `(-1, 1]`.** The reversed bounds are
    >    harmless: the generated C is `((-1) - (1)) * Frand() + (1)`, i.e.
    >    `rand(a, b) = (b - a)·U[0,1) + a`. Settled mechanically rather than guessed.
    > 6. **numpy fails loudly where NR degraded silently.** `LU-decomposition` raises
    >    only on an all-zero row (`281`) and substitutes `*TINY* = 1e-20` for a zero
    >    pivot (`321-322`); `numpy.linalg.solve` raises `LinAlgError` instead. The row
    >    replacement should make this unreachable, which is the point — if it ever
    >    fires, that is a signal rather than a silently perturbed answer.
    > 7. **`INDX` is a `float-matrix` holding integer row indices** (`397`). That is
    >    the second instance of the float-round-tripping-integers idiom, the first
    >    being `psi` in the next subgoal. Two instances make it an author habit, which
    >    is a firmer basis for not reproducing it than one would be.
    >
    > **Scope boundary found, not assumed:** `int-code-length` and `comb-code-length`
    > also live in `util.lsh` and are in its `dhc-make` list, but their only call sites
    > are `hmm-trainer.lsh:402-430` (`update-model-dl`, `update-total-dl`) — the MDL
    > apparatus of §8. They migrate at **revision 04**, not here.
    >
    > **Call sites confirmed by count** rather than carried over: `safe-/` 15 (matches
    > the master plan), `rand-p-vector` 7, `safe-add--log2` 5, `safe->--log` 2,
    > `int-delta` 2, `LU-solve` 2, `calculate-entropy` 2. And `dhc-make` (`548-574`)
    > **comments out** `minimize` and `idx-copy-svector`, which corroborates the
    > zero-call-site finding independently of the grep.
  - [x] Decide the module layout — whether `_numeric.py` carries all of it, or the solve and the initialisation split out
    > **Done:** One `_numeric.py`; see the Q&A above. Contents settled as
    > `safe_divide`, the log-domain accumulator, `stationary_distribution`, `entropy`
    > and `rand_p_vector`. The state-entropy *marginalization* is explicitly not here.
  - [x] Check whether `Code/Utility/C/util.c` is a compiled counterpart with semantics of its own; it is untracked, ignored by `.scratch/hmm-lush/.gitignore:91`, and the tracked set only ever widens
    > **Done:** It has no semantics of its own — it opens "WARNING: Automatically
    > generated code. This code has been generated by the DH compiler" and is the
    > output of `util.lsh`'s own `dhc-make` list. Generated *from the current file*,
    > not the older draft: `C_int_delta` appears 5 times in it and `int-delta` is
    > absent from `util.lsh copy`; mtimes agree, the draft at 2009-07-08 and the C at
    > 2009-07-14. That makes it evidence of a different kind than was expected —
    > not a second implementation to reconcile, but the **only machine-checked
    > statement of Lush semantics available in a repository with no Lush runtime**,
    > which is what settled findings 4 and 5 above. Tracked on that basis.
    >
    > **Second finding, unlooked-for:** `util.lsh copy` is not a backup. It is the
    > 2009-07-08 predecessor in which the NR routines were **0-indexed** throughout,
    > with `LU-solve (A B)` solving in place; the current file reverts to NR's
    > 1-indexed convention behind a copy-in/copy-out wrapper. The author attempted
    > exactly the index conversion a numpy port would need and backed it out — in the
    > one routine this branch deletes in favour of `numpy.linalg.solve`. Declined for
    > tracking; the finding is here instead.
  - [-] Create `packages/pfsmgraph-hmm/tests/`, the package's first
    > **Descoped:** Not independently completable, and completing it would mean
    > inventing a convention. Git tracks no empty directory, and
    > `packages/pfsmgraph-dataseq/tests/` carries **no** `__init__.py` and **no**
    > `conftest.py` — six bare `test_*.py` files and nothing else — so there is no
    > scaffolding file to seed it with. The directory therefore comes into existence
    > with the first real test in goal 2.
    >
    > What the subgoal was actually guarding is verified instead, and is the part
    > worth having recorded: `import pfsmgraph.hmm` resolves to
    > `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/__init__.py` under `uv run`, numpy
    > 2.5.2 satisfies the declared `>=2.1`, and the root `[tool.pytest.ini_options]`
    > sets only `addopts = "-ra"` with **no `testpaths`** — so pytest discovers from
    > the rootdir recursively and a new `packages/pfsmgraph-hmm/tests/` is collected
    > with no configuration change at all.
  - [x] Execute the tracking decision — widen `.scratch/hmm-lush/.gitignore`
    > **Done:** Three hunks. `util.c` negated (the directory un-ignored first, since
    > git never descends into an ignored one), the three `.hmm` fixture directories
    > negated, and the `Training/<all others>` declining note amended to point at its
    > own exceptions. Verified by `git status --untracked-files=all`: exactly
    > `util.c` plus 36 fixture files appear, while the two arch subdirectories,
    > `util.lsh copy`, and the other 15 models under `set02a_200/` all remain
    > ignored — each confirmed by `git check-ignore -v` naming the rule that matched.

- [x] Migrate the log-domain arithmetic Viterbi's inner loop calls
  > **Done:** `_numeric.py` and `tests/test_numeric.py` exist — the first code and
  > the first tests in `packages/pfsmgraph-hmm/`. Two functions written, two
  > dissolved, 30 tests, suite 94 → 124.
  >
  > **`bits` is unary where the original was binary, and that is a design call
  > rather than a transcription.** `safe-add--log2(sum, x)` took the accumulator as
  > an argument only so it could check `(= sum -1)`; with `+inf` absorbing on its
  > own, that argument does no work, so the primitive is `bits(p) = -log2(p)` and
  > accumulation is plain `+`. It also *exposes* the seeding defect the next
  > master-plan subgoal has to decide on: writing `delta[0] = init_state_p` and then
  > `delta[i-1, k] + bits(...)` puts a probability and a bit-count in the same
  > expression visibly, where `safe-add--log2` absorbed the mismatch silently.
  >
  > **The function survives for one reason that is not arithmetic.** `np.log2(0)`
  > raises a `divide by zero` RuntimeWarning; `bits` suppresses it in exactly one
  > place. Without that the function would genuinely dissolve into its one-line
  > body, and every call site would either emit spurious warnings or repeat the
  > `errstate` — until one of them suppressed a warning that mattered. `invalid` is
  > deliberately left unsuppressed, so a negative probability still yields `nan`
  > loudly.
  >
  > **Correction to this subgoal's own text:** it offered "replaced by `-inf`". The
  > sentinel is `+inf`. `np.log2(0)` is `-inf`, but a description length is
  > `-log2(p)`, so an impossible event costs `+inf` bits. The sign slip is the
  > §3 orientation trap in miniature, and is recorded rather than silently fixed.
  - [x] `safe-add--log2` and `safe->--log`, and decide the fate of the `-1` log-zero sentinel (§3) — faithfully reproduced, or replaced by `-inf`
    > **Done:** `safe-add--log2` → `bits`. `safe->--log` → **nothing**, and that is
    > the deliverable: the original spelled
    > `(and (<> y -1) (or (= x -1) (> x y)))` only because `-1` sorted numerically
    > *below* every real description length while meaning "worse than all of them".
    > `+inf` sorts where it means, so `>` is the entire function and a wrapper would
    > hide that. The original's truth table is kept as a parametrized test over
    > plain `>`, so reinstating a sentinel fails a test rather than passing quietly.
  - [x] `safe-/` (15 call sites) and `int-delta`
    > **Done:** `safe_divide` written **array-aware**, where the original is scalar
    > inside explicit loops — every one of the 15 sites normalizes an array by a
    > scalar or divides two arrays elementwise, so a scalar port would put loops
    > back into code numpy writes in one line. Zero is returned for `0/0` *and*
    > `x/0`, matching the original; numpy alone gives `nan` and `inf`, either of
    > which propagates into a parameter array as silent corruption.
    >
    > **Flagged, not resolved: `safe_divide` has no consumer in 0.1.0.** All 15
    > sites are in `update-data-p`, `update-approx-*` and `update-data-dl` (revision
    > 03) or `hmm-param`'s surgery (revision 04); none is in `update-viterbi-path`,
    > `update-entropy` or `init-random`. The master-plan subgoal's framing is
    > "the Utility code this release needs", and by that test this one does not
    > qualify — the call-site count was taken across the whole library. Migrated
    > anyway because both the master plan and this subgoal name it explicitly, it is
    > private so it reaches no public surface, and revision 03 then finds it done
    > and tested. Dropping it is a one-function revert if that is preferred.
    >
    > `int-delta` → **`np.eye`**. Both its call sites build the identity term of
    > `(Pᵀ - I)` one element at a time; the dissolution is recorded here and applied
    > in goal 3, which is where the solve is written.
  - [x] Tests, including behaviour at the sentinel boundary
    > **Done:** 30 tests. Several assert the *absence* of the original's machinery
    > rather than the presence of ours, which is what keeps the `+inf` claim honest:
    > absorption in both operand orders and across a whole accumulation, the
    > dissolved comparator's six-row truth table, non-negativity of a real
    > description length (the property that made `-1` available as a sentinel in the
    > first place), and silence at `log2(0)` versus a preserved `nan` warning for a
    > negative input.
    >
    > **One test was wrong and is corrected in place with its reason**, because the
    > next person writes the same one: `assert not hasattr(pfsmgraph.hmm, "_numeric")`
    > can never pass. Importing a submodule binds it as an attribute of its parent
    > package — the test file's own import does it — so the privacy that *is*
    > checkable is that neither helper is reachable as a top-level name.

- [x] Port the stationary-distribution solve
  > **Done:** `stationary_distribution` lands in `_numeric.py` with sixteen test cases;
  > suite 124 → 140. The solve itself is five lines, which goal 1 had already validated
  > — so the work here was the two things around it.
  >
  > **The reducible chain is the finding.** The row replacement trades one redundant
  > equation for the normalization, which suffices only when the null space is
  > one-dimensional. A chain with two closed communicating classes has nullity 2 and
  > stays singular; measured on a four-state example, `A` is still rank 3 of 4 after the
  > replacement. The original never noticed, because `LU-decomposition` substitutes
  > `TINY = 1e-20` for the zero pivot and returns a perturbed answer. This is not an
  > exotic input: **revision 04 searches topology by state merge and split**, so a
  > disconnected component is a plausible *outcome of the search*, and `state_p` is a
  > cached property under ADR 0017, so the failure surfaces on an attribute access. Hence
  > a `ValueError` naming the cause, with `raise ... from err` keeping the `LinAlgError`.
  >
  > **The fixtures are four-decimal prints, and that cost two test failures in unrelated
  > places** — an `abs=5e-5` tolerance that a value landed exactly on, and a singularity
  > assertion defeated by rows summing to `1 ± 1e-4`. Both are written up under the
  > second subgoal. Revision 03's differential tests inherit the hazard, which is the
  > reason it is recorded at this length rather than fixed in silence.
  > **Q:** The three tracked `.hmm` fixtures hold `transition_p` beside `state_p`. How
  > should the test reach them?
  > **A:** Read `.scratch/` in place, locating the repo root from `__file__`. They are
  > *tracked*, so unlike `.notebooks/`/`.data/` they exist in every clone, and goal 1
  > tracked them for exactly this purpose. The invariant they brush against says nothing
  > outside `.scratch/` may **import** from it; a test reading a tracked data file is not
  > an import, and the "exists on one machine only" reasoning behind the sibling
  > prohibition is precisely what tracking removes. `tests/` never ships either —
  > `[tool.hatch.build.targets.wheel] packages = ["src/pfsmgraph"]` — so "the fixture is
  > absent in an installed wheel" is not a scenario this repo has. Revisions 03 and 04
  > inherit the pattern, which is the point: `output_p` differential tests are far too
  > large to inline.
  > **Q:** A reducible chain leaves `A` singular even after the row replacement. What
  > should the port do?
  > **A:** Catch `LinAlgError` and re-raise as `ValueError` naming the domain cause, with
  > `raise ... from err` preserving the numerical detail. The row replacement trades *one*
  > redundant equation for the normalization, which suffices only when the null space is
  > one-dimensional — i.e. when the chain is irreducible. Measured on a four-state chain
  > with two closed classes: nullity 2, and `A` still rank 3 of 4 after the replacement.
  > This is not a hypothetical input, because revision 04 searches topology by state merge
  > and split, so a disconnected component is a plausible *search outcome*; and
  > `state_p` is a cached property under [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md),
  > so a bare "Singular matrix" would surface on attribute access. Row-stochasticity is
  > deliberately **not** validated here — that check belongs at `HMMParams` construction,
  > and duplicating it in a private helper creates two places that can disagree.
  - [x] Reproduce the row-replacement trick, not just the result: row 0 of `(Pᵀ - I)` overwritten with `Σπ = 1` before `numpy.linalg.solve` (§4)
    > **Done:** `stationary_distribution(transition_p)` in `_numeric.py`, five lines of
    > arithmetic under a docstring that carries the reason. The trick's *necessity* is
    > what got tested, since its result is indistinguishable from replacing any other
    > row: `test_the_homogeneous_system_needs_the_replacement` asserts
    > `matrix_rank(Pᵀ - I) < S` on all three fixtures, so a port that "simplifies" the
    > replacement away fails rather than quietly returning zeros.
    >
    > The squareness check is included and row-stochasticity deliberately is not; the
    > line between them is a structural precondition of the solve versus a property of
    > the model, and the second belongs to `HMMParams` under ADR 0017.
  - [x] Test against a chain whose stationary distribution is known in closed form
    > **Done:** `[b, a]/(a+b)` for the two-state chain, kept because it depends on
    > nothing under `.scratch/`. But the differential tests against the original's own
    > saved `state_p` are the stronger check, and `pi @ P == pi` — the defining property,
    > asserted with no expected value at all — is stronger still. Sixteen cases in total,
    > suite 124 → 140.
    >
    > **Two of them failed first, and both were the tests being wrong about the fixture
    > format rather than the solve.** Recorded because the same fact bit twice in
    > unrelated places, and will bite revision 03 too.
    >
    > 1. **A tolerance a value can land exactly on is a coin flip.** The 5-state model's
    >    true `π₀` is `0.10135`, printed as `"0.1014"` — precisely on the .5 tie — so the
    >    residual is `5.000000000000837e-05` and `approx(abs=5e-5)` failed by under an
    >    ulp. The bound was also incomplete: `transition_p` is printed to four decimals
    >    too, so *its* rounding propagates through the solve (renormalising the rows moves
    >    the 8-state residual 4.975e-5 → 4.784e-5, which is that term made visible). Now
    >    `1e-4`, with both sources named.
    > 2. **Rounding destroys exact singularity.** The 8-state fixture's rows sum to
    >    `1 ± 1e-4` after printing, lifting the smallest singular value of `(Pᵀ - I)` from
    >    `6.6e-17` to `1.2e-5` — nine orders above `matrix_rank`'s `3.6e-15` tolerance, so
    >    it reported *full* rank. Row-stochasticity is the hypothesis of "π is an
    >    eigenvector"; the fix is to renormalise and restore it, not to loosen a
    >    tolerance.
  - [x] Record that `LU-solve`, `LU-decomposition` and `LU-back-substitution` were replaced by a library call rather than translated
    > **Done:** In the function's docstring, with the behaviour change stated rather than
    > implied: the trio (`util.lsh:246-415`) transcribes the Numerical Recipes routines
    > down to NR's 1-indexed convention behind a copy-in/copy-out wrapper, and
    > `LU-decomposition` substitutes `TINY = 1e-20` for a zero pivot (`321-322`),
    > returning a silently perturbed answer where `numpy.linalg.solve` raises. That is
    > goal 1's finding 6 reaching the code it was about. `int-delta` → `np.eye` is
    > applied here, which was the other half of the dissolution recorded in goal 2.

- [ ] Port `rand-p-vector` and `calculate-entropy`, and record what migrates nowhere
  - [ ] `rand-p-vector` for parameter initialisation — settle the RNG seam: a passed-in `numpy.random.Generator` versus module-level state
  - [ ] `calculate-entropy`
  - [ ] Record that `minimize` / `minimize-from` / `minimize-int` and `mc.lsh` had **zero** call sites from `HMMlib`, and so migrate nowhere
  - [ ] Review the member's declared `numpy>=2.1` against what this code actually uses — the first numpy in `hmm`, so the first moment the bound is checkable
