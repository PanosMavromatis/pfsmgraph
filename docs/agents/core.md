# pfsmgraph

Shared project knowledge for any coding agent working in this repository.

## Current state

**`dataseq` is implemented and released; `hmm` has its first code as of 2026-09-03; the other three members are still empty scaffolding.** In place: the uv workspace root `pyproject.toml` (virtual — no `[project]` table), `uv.lock`, all five `packages/*` members with their own `pyproject.toml`, the (currently dormant) `meson.build` files for `align` and `hmm`, and an empty `pfsmgraph/<pkg>/__init__.py` for the three members that have no code yet (plus `dl/rnn/` and `dl/transformer/`). The ADRs in `docs/design/adr/` are authoritative for the decisions they cover — the twelve initial records from the PRD, plus 0013 (how this family documents its public surfaces) and 0014 (how imported migration source is retained), both added 2026-09-01; the PRD remains the narrative design document.

**What `dataseq` now contains.** Six modules under `packages/pfsmgraph-dataseq/src/pfsmgraph/dataseq/` (the container landed 2026-08-31, the encoder API 2026-09-01) and 74 tests — the first tests in this repository, and 74 of the suite's 271 today; 165 are `hmm`'s and the remaining 32 are the repo-root backend-matrix, API-docs, release-runbook and meson-source tests. `_reserved.py` hard-codes the ADR 0011 block as module constants, with no class or parameter that could relocate it; `_vocabulary.py` holds the `Vocabulary` protocol and `SymbolTable`, a frozen first-appearance-ordered implementation that encodes strictly and decodes *totally*, reserved codes included; `_record.py` and `_dataset.py` are the ragged container, whose records carry true lengths and never padding; `_collate.py` is `pad_collate`, where padding is introduced and always returned with its mask. The container imports neither torch nor pandas — verified in a subprocess — and its one runtime dependency is `numpy`.

**What `hmm` now contains.** Three modules and 165 tests. `_numeric.py` is the numeric
Utility code migrated from the Lush original, landed 2026-09-03 and complete for 0.1.0 at
five functions; `_params.py` is `HMMParams`, the ADR 0017 frozen parameter value, landed
the same day; `_viterbi.py` is the decode, landed 2026-09-04 and **the project's first
dynamic-programming kernel**. Four names are exported from
`pfsmgraph/hmm/__init__.py` — `HMMParams`, `viterbi`, `ViterbiPath` and
`ImpossibleSequenceError`.
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
`stationary_distribution` reports as a `ValueError` naming the cause rather than numpy's bare
`LinAlgError`. That matters because revision 04 searches topology by state merge and split, so
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

**The ADR 0003 reporting mechanism is complete as of 2026-09-01, and it lives at the repo
root.** `conftest.py` carries `pytest_report_header` and nothing else; `_backends.py`
beside it holds the registry, the import probe and the `PFSMGRAPH_REQUIRE_BACKENDS`
escalation; the repo-root `tests/` covers both. **The conftest must stay at the rootdir** —
`pytest_report_header` is a *startup* hook while conftest files under `packages/*/tests/`
are loaded during collection, so a hook sited there is registered too late and discarded
with no warning at all (measured on pytest 9.1.1: of two conftests each defining it, only
the root one printed). `tests/test_backends.py` asserts the placement precisely because
that failure is silent. **The matrix stopped being empty on 2026-09-04**, when
`hmm/_viterbi.py` reached ADR 0002 phase 1: `BACKENDS` holds one row,
`Backend("python", "pfsmgraph.hmm._viterbi")`, and a run now opens with
`backends: python ✓`. The row names the *kernel module*, not the package, because
`import pfsmgraph.hmm` succeeds with or without a decode in it and a probe that cannot fail
is not a probe; `hardware=None`, so a failed import escalates rather than skipping, since
nothing external is needed to run pure Python. `EMPTY_HEADER` stays under test — the branch
is still live and ADR 0003 requires that an empty matrix say so in as many words. Backend
enumeration is deliberately test-only and reaches no shipped artifact, because enumerating
backends is most of what a runtime backend-selection API needs and ADR 0003 leaves that
question explicitly open.

**The API documentation lives in a repo-level `docs/api/`, and that layout is now binding on all five members.** [ADR 0013](../design/adr/0013-api-documentation-layout-and-tooling.md) settles it: one subdirectory per distribution (`docs/api/dataseq/` is the only one so far — a member gets its subdirectory when it gets code), hand-written Markdown rather than a generator, no build step and no addition to the `dev` group. Two rules divide the labour and are the reason the choice is sustainable: **docstrings are normative for signatures**, since that is where an editor and `help()` look, while **`docs/api/` is normative for contracts** — the invariants, the reasons behind them, and the seams between distributions, which are contracts even when stated nowhere else. And **every code block is executed and its output pasted from the run**, error messages and tracebacks included; it is the only guard against drift that a hand-written layout has. Sphinx is deferred rather than refused (the docstrings already speak reST, so the migration is mostly configuration), and mkdocstrings is refused outright, because its Google/NumPy style expectation would force rewriting all six modules' docstrings to satisfy a tool.

**The encoder API is settled (2026-09-01), and three parts of it are contracts rather than choices.** `SymbolTable(symbols)` builds a frozen, first-appearance-ordered table, with `from_sequences(sequences)` for a corpus; the name is deliberate, since `Alphabet` implies single characters and this family's symbols are words. Encoding is strict by default and the opt-in is spelled per call — `encode(symbols, on_unknown="raise" | "unk")` — so one mapping serves curated training data and uncurated inference without needing two tables; the value is validated *before* the loop, so a misspelled policy cannot behave as `"raise"` until the first unseen symbol shows up in production. Decoding is total over `range(size)`, reserved codes included, because a padded batch is the array most likely to be decoded. And the symbol→code mapping is **public**: `code(symbol)` plus a `sym_to_code` property returning a `MappingProxyType` — a live read-only view, not a copy — because `pfsmgraph-align` builds an `(size, size)` scoring matrix from the whole mapping at construction and must not reach into a private attribute across a distribution boundary. Persistence and a frequency reordering were deliberately left out; see `docs/plan/DEFERRED.md` for both triggers.

**All five members are on meson-python as of 2026-09-04, and three of them contain no compiled code and never will.** The history is kept because the conclusion is counter-intuitive and the reasoning that produced it was wrong twice. meson-python's editable-install import hook injects a `sys.meta_path` finder that claims the entire `pfsmgraph` PEP 420 namespace and shadows any distribution left on a plain `.pth`, so while `align`/`hmm` alone were on meson-python, `import pfsmgraph.dataseq` (and `hseg`, `dl`) failed after `uv sync`. Neither package has compiled code yet — the `meson.build` extension blocks are dormant `if fs.exists()` guards — so the switch to meson-python was deferred to when the first `.pyx` lands, at which point the namespace/editable interaction must be solved (non-editable install of the compiled members, a single combined compiled distribution, or an upstream fix). **The second of those is refuted**, measured on `exp/meson-python-namespace` 2026-09-04: two meson-python finders *chain* rather than conflict — a finder that does not recognise a submodule returns `None` and the import falls through — so one finder is not the problem and any finder is, and a combined compiled distribution would still shadow the three pure-Python members. The boundary is meson-python versus plain `.pth`, which makes a fourth option visible that ADR 0012 does not list: put all five members on meson-python, so every one has a finder. Those revert recipes are gone now, spent; every member carries a `meson.build` instead. This qualifies the PRD §6.1 note, whose "namespace is fine" evidence was gathered with *setuptools* editable, which composes; meson-python's finder does not. **That deferral's own premise is falsified, and the resolution is scheduled *before* the first `.pyx` rather than with it** (decided 2026-09-04). ADR 0012 holds that choosing between the candidates without a compiled kernel "would be guessing"; but the finder is injected by the *editable install*, not by compilation, so it appears identically with the extension blocks dormant — which is how candidate 2 was refuted and the fourth option found. The ADR names the right mechanism in its Context and then reasons from the wrong one in its Alternatives, treating the deferred cost as compilation. What a real `.pyx` still adds — whether rebuild-on-import works, whether a stale `.so` can appear, how much dev-loop friction each option costs — bears on the *quality* of the surviving candidates, not on which are viable. **The choice is made: all five members on meson-python** (2026-09-04). Both surviving candidates were measured and both work — the compiled members installed non-editable (no finder exists at all, suite green at 271) and all five on meson-python (five finders chaining, suite green at 280). It turned on *how each fails*, not on elegance. Non-editable is the cleaner model and retires the baked-`ninja` footgun entirely, but `editable = false` must be repeated at every `[tool.uv.sources]` declaration site — a member-level declaration beats the workspace root's — and omitting it at any one site resurrects a finder that breaks **all five** members; separately, without `[tool.uv] cache-keys` a source edit is served stale with no error, since uv's default cache key for a local path is its `pyproject.toml` rather than its sources. Both failures are silent. Under all-five-on-meson-python the characteristic failure is a module missing from `install_sources`, which is a `ModuleNotFoundError` at import and is already caught by `tests/test_meson_sources.py` — that test went 7 → 16 by itself when the three new `meson.build` files appeared, which is where 271 → 280 comes from. The dev loop agrees and the gap widens with compiled code: editing a `.py` is visible with **no sync at all** under an editable finder, against a rebuild-and-reinstall for a non-editable copy that becomes a full recompile at the first `.pyx`. Note this inverts ADR 0012's implicit cost model, which treated non-editable installation as the cheap fallback and rebuild-on-import as the luxury being surrendered. Two consequences for the superseding ADR: it overrides ADR 0008's per-package build backends, and `pfsmgraph-dataseq` 0.1.0 was published from hatchling, so its next release ships a meson-built wheel and the four-file release invariant must be re-verified against an actual built wheel. **Landed 2026-09-04.** Verified from a deleted venv and a plain `uv sync` — not `--reinstall`, which reuses an environment that already has `ninja` on disk and so cannot test the question: uv reported "Prepared 5 packages without build isolation", all seven import paths resolve, five finders sit on `sys.meta_path`, and the suite is green at 280. `pfsmgraph.__path__` is still a single synthetic loader entry, which is the point stated positively — the fix is not to stop the finder replacing `__path__`, it is to leave no member relying on `__path__`. Two things the landing turned up that the evaluation had not: the scope is **five members, not two**, so `dataseq`/`hseg`/`dl` needed comments written from scratch where `align`/`hmm` had a recipe to follow; and the `dev` group needs **`numpy`** as well as `meson-python`/`cython`/`ninja`, because it is a *build* requirement of `align`/`hmm` that `build-system.requires` cannot supply once isolation is off. **Recorded as [ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md)** on the same day, which supersedes 0012 and overrides 0008; that record, not this paragraph, is authoritative, and it is where the two rejected alternatives are argued rather than merely named. 0008 and 0012 are the only `Superseded` records in the directory — the sequence 0008 → 0012 → 0018 is the one worked example here of a decision surviving two revisions of its own evidence.

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
merge.

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

Still to do, in PRD order (§11): `hmm` (Lush translation), then `align`, then `hseg`. **The `hmm` migration is planned as three revisions rather than one**, because the Lush trainer spans three problems that fail differently: `02-hmm-v0.1.0` (Viterbi and the `dataseq` interface, carrying the project's first DP kernel, first `.pyx`, first non-empty ADR 0003 backend matrix; the meson-python resolution ADR 0012 deferred to it was in the event settled *ahead* of it, on `exp/meson-python-namespace`, since the finder turned out to be injected by the editable install rather than by compilation — [ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md)), `03-hmm-v0.2.0` (Baum-Welch on a fixed topology, with an optional `torch` autograd backend held against the numpy reference), and `04-hmm-v0.3.0` (topology search by state merge and split, scored by minimum description length). Their plans are drafted under `docs/plan/planned/` and registered in the master plan. `dataseq` is finished and released: 0.1.0 is on PyPI and tagged `pfsmgraph-dataseq-v0.1.0`, as of 2026-09-02. The container half of the merge (three existing implementations, `dl` version as base — §3.5) has landed.

## Commands

Toolchain: **uv** (workspace) + **pytest**. Requires `uv` and Python ≥ 3.10.

- `uv sync` — create/refresh the venv; installs all five members editable (plain `.pth`) plus the `dev` group (`pytest`).
- `uv run pytest` — run the suite (280 tests: 74 in `packages/pfsmgraph-dataseq/tests/`, 165 in `packages/pfsmgraph-hmm/tests/`, and 41 in the repo-root `tests/` — 18 covering the ADR 0003 backend matrix, 5 executing documented code blocks against their pasted output per ADR 0013, 2 checking that `docs/ops/release.md` names only recipes the root `justfile` defines, and 16 asserting each `meson.build`'s `install_sources` matches the package on disk). That last figure was 7 until 2026-09-04: it is parameterised over `packages/*/meson.build`, so it grew by itself when the three pure members got theirs, and that growth **is** the 271 → 280 — no test was written for the change that occasioned it. That verifier reads `docs/api/*/*.md` **and `packages/*/README.md`**: a member README becomes a PyPI long description under an immutable version, so it is the one documentation surface where drift cannot be corrected in place. Every run opens with the backend header. One narrow skip is by design: `test_torch_interop.py` verifies the `DataLoader` integration and skips when torch is absent, since torch is a dependency of no member.
- `uv build --package pfsmgraph-<pkg>` — build one member's sdist + wheel.
- `uv lock` — refresh `uv.lock` (committed; one lockfile for the whole family).

The repo-root `justfile` wraps the release path — `just release <version> [package]` runs
test → build → `twine check` → preflight → upload → tag, defaulting to `pfsmgraph-dataseq`
and taking any member as its second argument. It requires `just` (`brew install just`) and
is the only place per-package release tags are formed. `just` alone lists every recipe;
[`docs/ops/release.md`](../ops/release.md) is the runbook. Two properties of it are
deliberate and easy to break: a guard must be a *prerequisite* of `release`, never a body
line, because every body line runs after `publish` — the irreversible step; and `clean` is
a prerequisite of `build` because `dist/` is shared by all five members while the publish
glob carries no version, so removing it lets a stale version of the same package be
uploaded.

All five members build through meson-python (landed 2026-09-04), so the root `dev` group carries `meson-python`, `cython`, `ninja` **and `numpy`** — the last because it is a build requirement of `align`/`hmm` that `build-system.requires` cannot supply with build isolation off. A C compiler is additionally needed once those two get their first `.pyx`; nothing is compiled today. **The dev group alone is not sufficient**, measured 2026-09-04: the generated editable loader bakes an *absolute* `ninja` path at build time and never consults `PATH`, so under uv it points into the build-isolation directory uv deletes once the build finishes, and every import then dies `FileNotFoundError` before the namespace shadowing is even reachable. Those members must also be built without build isolation — `[tool.uv] no-build-isolation-package`.

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

**Implementation order** (`dataseq` → `hmm` → `align` → `hseg`) deliberately differs from **release order**, which must follow the dependency graph since a package cannot publish before its dependencies exist on PyPI.

## Invariants

These constrain any code written here. They are inherited from the proof-of-concept and are non-negotiable without amending the PRD.

- **No `pfsmgraph/__init__.py` anywhere.** The `pfsmgraph` level is a PEP 420 implicit namespace; each distribution contributes exactly one regular subpackage beneath it. An `__init__.py` at that level breaks every other package's imports.
- **Encode at the boundary.** Multi-character string symbols are mapped to integers at the entry point of every public call; all inner computation is integer-only; results decode back to strings at exit. This is what makes Cython and CUDA backends mechanical to write — they never touch string types.
- **Fixed reserved symbol block** in `dataseq`, not configurable: `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6. `PAD` must be 0 because PyTorch's zero-fill idioms (`pad_sequence`, `torch.zeros()` buffers) would otherwise silently mean something other than "absent". Encoding is **strict by default** — unseen symbols raise; `UNK` fallback is explicit opt-in.
- **The HMM is arc-emission (Mealy), not state-emission.** A symbol is emitted while *crossing* a transition, so the emission parameter is `output_p[i, j, symbol]` — indexed by source state, destination state and symbol — and never `B[state, symbol]`. A path over *N* symbols visits *N+1* states, which is why `dataseq`'s `seq-state` carries state arrays one longer than its symbol array. Every textbook, and every library (`hmmlearn`, `pomegranate`), is the other formulation, so this is the single fact most likely to be lost in translation; the emission factor also **cannot** be hoisted out of a Viterbi or forward inner loop, because it depends on both endpoints. See [ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md).
- **Four-phase algorithm lifecycle** ([ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md) amends [ADR 0002](../design/adr/0002-three-phase-algorithm-lifecycle.md), 2026-09-03), applied in order wherever dynamic programming appears: pure Python (correctness) → Cython (performance) → Numba CPU-parallel, `prange` anti-diagonal (parallel correctness, no GPU required) → Numba CUDA anti-diagonal wavefront (scale).
- **One parameterized test suite per algorithm**, run automatically against every available backend, so backend equivalence is enforced rather than assumed. Absent hardware (no CUDA device) skips, but *loudly* — the session header names every backend excluded and why; a backend that is implemented but not importable (missing or stale Cython build) is a hard failure, never a skip; a lifecycle phase not yet reached contributes no parameter at all. `PFSMGRAPH_REQUIRE_BACKENDS` escalates skips to failures for CI. See ADR 0003. **The parameterization half is not in force yet, and cannot be until `align`** (found 2026-09-04): ADR 0003 also requires tests be written against the public API only, and `viterbi(params, record)` has nowhere to put a backend — adding one is the runtime backend-selection API that ADR's own Open section routes to `align`, so the two requirements are jointly unsatisfiable before then. Consequence to keep in mind when reading a green run: `backends: python ✓` means the kernel imports, **not** that any suite ran twice. ADR 0003 asks for one thing unconditionally in the meantime, and `test_viterbi.py` does it — a test that reaches a backend's internals lives in a labelled, explicitly non-shared section rather than among the public-API tests. Its corollary is easy to miss: **a tie-breaking rule is contract**, not an implementation detail, because two correct backends would otherwise legitimately disagree.
- **~~Build backends are per-package, not family-wide.~~ Superseded 2026-09-04: the backend is family-wide, and it is meson-python for all five members.** The original invariant (ADR 0008) said meson-python for compiled members and hatchling for pure ones; that split is what the namespace shadowing makes unworkable, since a member on a plain `.pth` is shadowed by any sibling's meson-python finder. What survives from it: meson-python editable installs need `ninja` present for rebuild-on-import, and — because the loader bakes an absolute path to it rather than consulting `PATH` — every member must also be built without build isolation. See "Current state" and [ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md).
- **A released member ships four files the version bump does not imply**, all inside
  `packages/pfsmgraph-<pkg>/`: a `README.md` (its PyPI long description — the root one is
  about the workspace and every relative link in it 404s there), a `LICENSE` **file**, the
  `Typing :: Typed` classifier, and a PEP 561 marker at
  `src/pfsmgraph/<pkg>/py.typed`. Two of these fail *silently* if placed wrong, which is why
  they are an invariant rather than a checklist. A `LICENSE` symlinked to the repo-root one
  builds a valid-looking sdist and then fails on **unpack** — a symlink escaping the sdist
  root is refused — so it must be a real copy. The copy is also a
  **silent drift surface**: the wheel ships the member's copy, not the root one, so editing
  the repo-root `LICENSE` alone changes nothing a consumer sees. Keep the two byte-identical
  — both carry the copyright line `Copyright (c) 2026 Panayotis Mavromatis`, the legal name
  rather than the professional one, because a license is read by lawyers. A `py.typed` at the distribution root instead
  of inside the importable package reaches no wheel at all, with no error and no warning, and
  a type checker then discards every annotation in the package (measured on `dataseq`: a
  deliberate `bad: str = vocab.size` was *accepted*). It cannot go at the `pfsmgraph/`
  namespace level either, for the same reason no `__init__.py` may: no single distribution
  owns that level. Verify by installing the built wheel into a clean venv outside the
  workspace — a file listing shows what went into the box, not what a consumer gets out.
  All four land **in the release commit**, since they are wheel content and adding them later
  leaves a published version standing as the broken one.
- **"GPU" means two unrelated things.** `numba-cuda` for the DP packages, `torch` for `dl`. Do not unify these into one `[gpu]` extra.
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

A live instance of the footgun, worth recognising: the four members still in development declare `0.1.0.dev0` (`dataseq` moved to `0.1.0` at its release commit, 2026-09-02), and `0.1.0.dev0` does **not** satisfy `>=0.1` under PEP 440 — a `.devN` release sorts strictly before the final, and is excluded even with `prereleases=True`. So `align`'s declared `pfsmgraph-dataseq>=0.1.0` (reviewed and spelled in full on 2026-09-01; the `pfsmgraph-align>=0.1` bounds are deliberately left unreviewed, and the divergent spelling is what records that) is satisfiable by nothing that exists today: PyPI has only `0.0.0`, and local is `0.1.0.dev0`. It never fails because the workspace source satisfies any constraint, and **`uv.lock` cannot catch it either** — a workspace member's `requires-dist` entry records no version specifier at all, so changing all four declared bounds left the lockfile byte-identical (measured 2026-09-01). Review is the only mechanism there is. The `pfsmgraph-dataseq` half of this resolves as soon as `0.1.0` is on PyPI; the three `pfsmgraph-align>=0.1` bounds stay unsatisfiable until `align` releases.

## Versioning

**Versions are per-package, and there is deliberately no `VERSION` file at the repo root.** Release order is forced by the dependency graph — `dataseq` must publish before `align` can — so the five members can never share a version, and a repo-wide version number would be a claim about nothing. Each member owns the `version` field in its own `pyproject.toml`: `pfsmgraph-dataseq` reads `0.1.0` as of its release commit (2026-09-02) and the other four still read `0.1.0.dev0` — the scheme working as intended rather than drift.

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
- `docs/design/references.md` — external work a decision here leans on. Each entry records **what it is relied on for**, not a bare citation, so a misremembered or stale reference is detectable rather than merely present. Added 2026-09-03.
- `docs/plan/TODO.md` — the master plan. Its `## Planned revisions` section registers revisions drafted but not yet opened; each one's subgoals are drafted as `docs/plan/planned/<label>.md`, so opening it is a splice rather than an authoring job. **The draft must not sit in `docs/plan/<label>/`**, tempting as that is since it is the shape `/close-revision` archives: `/open-revision` refuses any label whose directory already exists, correctly assuming only `/file-plans` creates one, so a draft filed there makes a free label look taken and the command refuses on every planned revision. Measured 2026-09-03, when it did. Each draft carries a `**Status**: planned` preamble that stays behind at splice time.
- `docs/design/adr/` — seventeen records: the twelve initial ADRs from the PRD plus 0013 through 0017, authoritative for the decisions they cover; [`adr/README.md`](../design/adr/README.md) indexes them. Add new records with the next unused number and a row in that index; numbers are never reused. **0015 is the first record about a model rather than about packaging, tooling or process**: `pfsmgraph.hmm` is arc-emission (Mealy), so a symbol is emitted on the transition `i → j` and the emission parameter is `output_p[i, j, symbol]`, never `B[state, symbol]`. Every textbook and every library is the other one, so this is the fact most likely to be lost in translation — and it is why `dataseq`'s `seq-state` carries state arrays one longer than its symbol array. **0017 is the second, and reads after it**: model parameters are a *frozen value* — the three arc-emission arrays with `writeable = False` buffers and the `Vocabulary` that fixes what their symbol axis means, derived quantities (the stationary distribution, the entropies) computed rather than stored, and algorithms taking parameters rather than owning them. Viterbi is therefore a free function over parameters and one `dataseq` record, not a method on a trainer: the Lush decode reads no forward variable, so its placement on `hmm-trainer` was an artefact of where the corpus lived (`HMMLIB-ACCOUNT.md` §7). It returns a result rather than writing the path back into the sequence object, and `pad_collate` is out of scope until revision 03 — a record never holds padding, so a single-sequence decode has no mask to consult. Lush's `hmm`/`hmm-param` mutable working-copy split is deliberately **not** inherited; its motivation was two buttons in a GUI that migrates nowhere, and its own surgery methods reallocate rather than mutate in place, so it never bought what it appeared to.
