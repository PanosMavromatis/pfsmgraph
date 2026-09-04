# Master plan — pfsmgraph

**Status**: active

The two-tier plan convention for this repository:

- **Master plan** — this file, `docs/plan/TODO.md`, lives on `main`. It defines
  **revisions** (milestones) and their **subgoals**; each subgoal spawns a branch.
- **Branch plan** — `docs/plan/<type>-<slug>/TODO.md`, created by `/new-branch`, worked
  by `/hitl-step`, stamped `merged` by `/smart-merge`. It **survives on `main`** as the
  durable record of how a subgoal was executed, and is filed into its revision's
  directory by `/file-plans`.
- **PR body** — the distilled description plus a pointer to the branch plan directory.
  Not a verbatim archive.

`TODO.md` rather than `DO.md` is deliberate: branches inherit the master plan's model,
so every subgoal here runs under `/hitl-step` with its Q&A logged inline. The `dataseq`
merge reconciles three existing implementations and settles a public API that four other
packages depend on — the questions are genuinely open, and the answers are worth keeping.

**Related documents.** Decided-but-not-yet-actionable work is _not_ listed here; it lives
in [`DEFERRED.md`](DEFERRED.md), indexed by the trigger that unblocks it. Open design
questions live in [PRD §8](../design/PRD.md) and the `Open` sections of the
[ADRs](../design/adr/README.md). This file tracks work that is active now, plus the
revisions already drafted and waiting to be opened.

## Subgoals — revision 02-hmm-v0.1.0

`dataseq` is released and the family's base layer is fixed, so PRD §11 puts `hmm` next.
This revision is the first of three, and it is deliberately the one that carries the
project's _firsts_ rather than the most HMM content: the first dynamic-programming kernel
in the repository, the first `.pyx`, the first non-empty ADR 0003 backend matrix, and the
resolution of the meson-python namespace problem that [ADR 0012](../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md)
is standing down. Viterbi is the right kernel to carry them because it is the simplest
recurrence in the library — a single max-plus pass with a backtrace — so when the compiled
phases misbehave, the algorithm is not also in question.

Settled on the planning branch and not to be relitigated here: numpy is the reference
implementation and the only required runtime dependency; `torch` enters at revision 03 as
an optional backend, never as a hard dependency; and the migrated Utility code lives
private to `pfsmgraph.hmm` rather than in a new distribution.

**This revision fires `DEFERRED.md`'s `## Trigger: the first .pyx`.** That trigger gates
the ADR 0012 revert and the meson-python editable-install shadowing, and it is why
subgoal 5 sits between the pure-Python kernel and the Cython one rather than after both.

**The public-surface subgoal below also decides the class architecture, and that is worth
separating from the encode-at-the-boundary question it's paired with.** The source splits
the model into three classes that do not obviously survive translation: `hmm` (the
persisted parameters — load/save, `update-entropy`), `hmm-param` (a mutable _working copy_,
synchronized only by `copy-from-model`/`copy-to-model`), and `hmm-trainer` (Viterbi, the EM
machinery, and topology search in revision 04). `hmm-param` exists to back an interactive
undo — `hmm-trainer-view.lsh`'s "Keep model" / "Reset model" buttons (`HMMLIB-ACCOUNT.md`
§5) — and this migration is not porting that GUI (see revision 04's subgoal recording that
`hmm-trainer-view.lsh` migrates nowhere). Reproducing the split without its reason has a
measured cost and no offsetting benefit: `update-entropy` is duplicated verbatim between
`hmm.lsh:228-262` and `hmm-param.lsh:66-100`, 35 lines (`HMMLIB-ACCOUNT.md` §13) — exactly
the drift hazard `core.md`'s "ADRs outrank the imported implementations" invariant exists
to catch, and duplication a single class would not have. Nor is the decode/train coupling
worth keeping by default: "No separation between decode and training. Viterbi is a method
on `hmm-trainer`, so decoding a sequence requires constructing a trainer, which requires a
corpus" (`HMMLIB-ACCOUNT.md` §14) is recorded there as an absence, not a feature. None of
this settles what to build instead — a single mutable model, an immutable parameter object
replaced each step, or something the optional `torch` backend pulls toward once parameters
can be `nn.Parameter`s (revision 03) — only that inheriting the Lush shape by not deciding
is not a neutral choice. Revision 04's topology search inherits whatever this revision
settles, since `split-state`/`merge-states` lived in `hmm-param` and its rollback story
(the parameter-representation subgoal) is where a wrong choice here gets expensive to
unwind. Record the decision as an ADR once subgoal 2 below settles it — this is
architecture on the order of
[ADR 0010](../design/adr/0010-dataseq-composition-merging-three-implementations.md) and
[ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md), not an implementation
detail.

**Drafted before the source was read.** The release boundaries below come from a
structural survey of `.scratch/hmm-lush/Code/HMMlib/` — definition maps, call-site counts,
comment headers — not from reading the 2,044 lines. Subgoal 1 is the reading, and its
first duty is to check these boundaries. What would falsify them:

- **Viterbi turning out to depend on the forward variables.** The split assumes
  `update-viterbi-path` (`hmm-trainer.lsh:188-257`) computes δ independently of the α that
  `update-data-p` (`126-188`) builds. If it reads α, the 02/03 boundary moves and Viterbi
  drags the forward pass into this release with it. **Checked, does not hold**:
  `update-viterbi-path` reads no forward variable — `alpha*` is a local of `update-data-p`
  alone, and the two methods are scheduled together by `update-data` only for readability,
  not a data dependency (`HMMLIB-ACCOUNT.md` §7). The boundary stands.
- **The stationary-distribution solve being something else.** `hmm-param.lsh:82` and
  `hmm.lsh:244` build a matrix from `int-delta` and call `LU-solve`; that reads as
  `(I - Pᵀ)π = 0`, but it was inferred from two lines of context. **Checked, does not
  hold**: the solve is `(Pᵀ - I)π = 0` with the first row replaced by `Σπ = 1`
  (`HMMLIB-ACCOUNT.md` §4); the guessed sign is flipped, but `(Pᵀ - I)` and `(I - Pᵀ)`
  share the same null space, so the port is unaffected.
- **`hmm-trainer.lsh:21-126` not being separable.** The scaffolding is assumed shareable
  across 02 and 03. If the constructor demands the training apparatus, 02 gets no trainer
  at all and Viterbi becomes a free function over a model — which may be the better design
  regardless. **Checked, holds**: neither constructor branch is free of the training
  apparatus, and both require a corpus unconditionally; "a decode-only use of this library
  is not expressible in its own terms" (`HMMLIB-ACCOUNT.md` §15). `21-126` is not
  shareable scaffolding. Subgoal 2's premise below — Viterbi as a method with no trainer
  object in this release — is now evidence-backed rather than assumed.

- [x] Read `Code/HMMlib/` in its own terms and write `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, following `ACCOUNT.md`'s conventions — measurements against the two tracked specimen corpora, and **provenance unknown** for behaviours the code admits but may never have exercised. Check the three falsifiers above and revise this plan if any holds.
  > **Branch:** docs/hmmlib-account
  > **Done:** `HMMLIB-ACCOUNT.md` written (600 lines, 15 sections, an appendix); the three
  > falsifiers checked and recorded above (two do not hold, one holds); subgoals 2 and 3
  > below amended with the account's consequences; and, independently, a design handoff
  > surfaced from a local scratch directory promoted to
  > [ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md) — PR #13.
- [x] Settle the public surface of `pfsmgraph.hmm` 0.1.0 and where it meets `dataseq`: what a caller constructs, what Viterbi is a method _on_ given there is no trainer object in this release, and which of `SymbolTable`, the record container and `pad_collate` it consumes. Apply _encode at the boundary_ ([ADR 0001](../design/adr/0001-encode-at-the-boundary.md)) by naming the exact entry and exit points where strings are still permitted. Decide the class architecture explicitly as part of this — see the paragraph above — rather than defaulting to Lush's `hmm`/`hmm-param`/`hmm-trainer` split by not deciding. The account gives this two more concrete constraints. First, Lush's model does not take an alphabet as an argument — its constructor reads `_alphabet_size`/`_alphabet` directly out of the corpus's `.sds` directory (`HMMLIB-ACCOUNT.md` §4), which is exactly the file-coupled seam `ACCOUNT.md` §1 already found on the container side; do not reproduce it — take a `SymbolTable` explicitly. Second, Lush's Viterbi always decodes one sequence — batching belongs to the trainer, and the trainer does not exist in this release — so `pad_collate`'s masked-batch path is plausibly not this subgoal's concern at all: settle whether Viterbi 0.1.0 consumes a single `dataseq` record directly, leaving `pad_collate` for revision 03's batched training.
  > **Branch:** feat/hmm-public-surface
  > **Done:** The class architecture is a **frozen parameter value**
  > ([ADR 0017](../design/adr/0017-frozen-parameter-object-for-hmm.md)), not Lush's mutable
  > `hmm`/`hmm-param` split — whose own surgery methods never exercised the mutability,
  > since `split-state` reallocates and rebinds every slot (`hmm-param.lsh:153-158`,
  > `210-215`) because a shape change cannot be done in place. Viterbi is a **free function
  > over parameters and one `SequenceRecord`**, returning a result rather than writing back
  > into the sequence object; §7's "reads no forward variable" is what licensed removing it
  > from the trainer. The `Vocabulary` is taken as `dataseq`'s Protocol and **retained**, so
  > a mismatch between identically-shaped tensors is detectable; `pad_collate` is deferred
  > to revision 03 structurally, a record never holding padding. The package sits entirely
  > **below** the [ADR 0001](../design/adr/0001-encode-at-the-boundary.md) boundary — two
  > string entry points, two exits, no symbol among them — and the public/kernel split
  > enforces what that ADR calls unenforceable. Revision 04's accept/reject ratio was
  > checked and **does not exist**: §11 records that the original's search was driven by
  > hand, so it is a forward assumption rather than a translated fact. Left to the kernel
  > subgoal: whether `A` is `vocab.size` or only the user symbols — PR #15.
- [x] Migrate the Utility code this release needs, private to the package: `_numeric.py` for `safe-/` (15 call sites), **`safe-add--log2` and `safe->--log`** — the log₂-domain accumulator and comparator Viterbi's inner loop actually calls, home of the `-1` log-zero sentinel (`HMMLIB-ACCOUNT.md` §3) — and `int-delta`; plus the stationary-distribution solve (`LU-solve` → `numpy.linalg.solve`), `rand-p-vector` for parameter initialisation, and `calculate-entropy`. The solve needs its row-replacement trick reproduced, not just its result: `(Pᵀ - I)π = 0` is singular by construction, so a port that hands the homogeneous system as stated to a dense solver fails outright — row 0 must be overwritten with the normalization `Σπ = 1` before calling `numpy.linalg.solve` (§4). Record which Numerical-Recipes transcriptions were replaced by a library call rather than translated, and that `minimize`/`mc.lsh` had **zero** call sites from `HMMlib` and so migrate nowhere.
  > **Branch:** feat/hmm-numeric-utils
  > **Done:** `_numeric.py` and 66 tests — the first code in `packages/pfsmgraph-hmm/`,
  > suite 94 → 160. **Six functions named, five written**: `int-delta` dissolves into
  > `np.eye` and `safe->--log` into plain `>`, both consequences of replacing the `-1`
  > log-zero sentinel with `+inf` — declined as *uncheckable*, since there is no Lush
  > runtime here and the sentinel reaches no persisted artifact. `bits(p)` is unary where
  > `safe-add--log2` was binary, the accumulator argument having existed only to test that
  > sentinel. The stationary solve reproduces the row replacement and adds what the
  > original could not detect: a **reducible** chain has nullity 2 and stays singular after
  > one replaced row, so it raises `ValueError` naming the cause where
  > `LU-decomposition` substituted `TINY = 1e-20` and returned a perturbed answer. That is
  > not exotic — revision 04's merge/split search can produce a disconnected component.
  >
  > **Two of this subgoal's own claims were wrong and are corrected rather than carried.**
  > `minimize-int` does **not** have zero call sites: `hmm-trainer.lsh:441` (`suggest-d`)
  > minimizes `total-dl` over `d`, so it migrates at **revision 04**, the same MDL boundary
  > as `int-code-length`. And `mc.lsh` *is* libloaded (`load-hmm.lsh:6`); what supports
  > "migrates nowhere" is that none of its four names is ever called. The text above records
  > what was believed at planning time and is left as written.
  >
  > Also settled: `safe_divide` has **no consumer in 0.1.0** (all fifteen sites are revisions
  > 03–04), `entropy` deliberately does not reuse `bits` (`bits(0) = +inf` is right for a
  > description length and wrong for an entropy term), `rand_p_vector` takes a **required**
  > `Generator`, and `numpy>=2.1` was reviewed and kept — justified by the compiled future,
  > not today's API. Three `.hmm` model directories are now tracked as differential
  > fixtures; their four-decimal print format is a documented trap for revision 03 — PR #16.
- [x] Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy) with the ADR 0003 test suite, and register it as the first backend. The session header stops reading `backends: none registered` for the first time since the hook landed. Two defects the account marks **provenance unknown** must be a decision, not a silent reproduction: `update-viterbi-path` seeds δ with raw `init-state-p` into the bit-domain accumulator, inverting the start-state preference and turning an exactly-zero initial probability into the _best_ possible δ rather than the impossible sentinel (`HMMLIB-ACCOUNT.md` §7); and `psi` round-trips state indices through a float matrix, harmless below 2²⁴ states and not worth reproducing. Decide and record whether the seeding bug is fixed or faithfully reproduced — the ADR 0003 test suite should encode whichever is chosen, not accidentally validate a bug against itself.
  > **Branch:** feat/hmm-viterbi-python
  > **Done:** `HMMParams` and the decode landed; the header reads `backends: python ✓`.
  > Suite 160 → 264. **Both §7 defects were decided rather than reproduced**, and the
  > seeding one was settled by measurement rather than argument: `save-viterbi-path` had
  > written a `.vpath.xls` beside each saved model, so the port has a decode oracle and the
  > correction is worth exactly one position in 3807. `psi` is `np.int64`, confirmed
  > harmless below 2²⁴ rather than assumed — PR #17.
  >
  > **Two things the next subgoals need.** *For phase 2:* the ADR 0003 suite is **not**
  > parameterized and cannot be until `align`. That ADR requires the backend be a fixture
  > parameter *and* that tests be written against the public API only; `viterbi(params,
  > record)` has nowhere to put a backend, and adding one is the selection API its own Open
  > section defers. So the line below — "backend equivalence against phase 1 is enforced by
  > the parameterized suite, not asserted" — has a prerequisite that is not scheduled
  > anywhere, and phase 2 is where that bites. *For phase 2 and 3 both:* **a tie-breaking
  > rule is contract**, per ADR 0003's Negative section, because two correct backends would
  > otherwise legitimately disagree. Ours is first-wins, matching the original, and it is
  > exercised by no fixture — 0 exact ties in 3804 positions — so it is pinned by a
  > constructed uniform model that a wavefront kernel must also satisfy.
- [ ] Resolve the meson-python namespace shadowing and move `hmm` off hatchling, reverting [ADR 0012](../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md) by whichever of its three recorded candidates survives contact: non-editable install of the compiled members, one combined compiled distribution, or an upstream fix. Re-add `meson-python`, `cython` and `ninja` to the root `dev` group.
  > **Branch:** exp/meson-python-namespace
- [ ] Implement Viterbi at ADR 0002 phase 2 (Cython), the first `.pyx` in a distribution. Backend equivalence against phase 1 is enforced by the parameterized suite, not asserted.
- [ ] Implement Viterbi at ADR 0002 phase 3 (Numba CPU-parallel, `prange`) — [ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md) inserted this phase 2026-09-03, one step before what used to sit here; it is now the earliest point real concurrent execution is attempted.
  - [ ] **Settle the anti-diagonal question first — this is the point at which it is strictly needed.** [ADR 0002](../design/adr/0002-three-phase-algorithm-lifecycle.md):53 states the wavefront transformation is "the same transformation for every DP kernel in the family". A structural survey on the planning branch suggested it is not — an HMM recurrence is 1-D over time with dense N×N state coupling, so it has no anti-diagonals, and its parallel decompositions are batch, states-within-a-timestep, and possibly an associative scan over time in the (max, +) semiring. **The finding was deliberately left undecided on the planning branch** because phases 1 and 2 do not depend on it: a Cython kernel is single-threaded, so nothing before this subgoal can falsify or need it. Decide it here, against a kernel that exists, and settle whether it is a wording fix scoped to alignment-family kernels or a reversal warranting its own ADR number.
  - [ ] Implement whichever decomposition that decision names, under `@njit(parallel=True)`/`prange`.
- [ ] Implement Viterbi at ADR 0002 phase 4 (Numba CUDA), renumbered from phase 3 by [ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md). Reuses the decomposition phase 3 already validated; what remains here is hardware-kernel-specific — memory coalescing, warp occupancy, `cuda.jit` semantics — not algorithmic. Backend equivalence against phases 1-3 is enforced by the parameterized suite.
- [ ] Write the `/docs/api/` documents that pertain to this release.
- [ ] Release `pfsmgraph-hmm` 0.1.0 via `just release 0.1.0 pfsmgraph-hmm`, shipping the four files the version bump does not imply, and set honest lower bounds on any intra-family dependency naming it.

## Planned revisions

Drafted ahead of being opened, one file each under [`planned/`](planned/), carrying the
same `## Subgoals` section `/close-revision` archives — so opening one is a splice rather
than an authoring job. They sit in `planned/` rather than in `docs/plan/<label>/` because
`/open-revision` refuses a label whose directory already exists, on the sound assumption
that only `/file-plans` creates one; a draft filed there would make every planned revision
look already-opened. The detail lives in those files and not here, which is the point: this file stays
short enough to read on every session, and a revision's subgoals enter it only while that
revision is in progress.

The `hmm` migration is **three** releases rather than one. The Lush trainer is 1,102 lines
spanning three problems that fail differently — a decode, a fixed-topology estimator, and
a search over model shapes — and each raises its own questions about parallelism and data
structures. Conflating them would put the project's first `.pyx`, its first EM loop and its
first resizing search in one revision, where a failure in any of them would be diagnosed
against all three.

The first of the three is **open** and its subgoals are above; the two below are still
planned. Each entry moves up as its revision is opened.

- **Revision 03-hmm-v0.2.0** — Baum-Welch on a fixed topology, with an optional `torch`
  backend whose autograd E-step is held against the numpy reference's explicit
  forward-backward. See [`planned/03-hmm-v0.2.0.md`](planned/03-hmm-v0.2.0.md).
- **Revision 04-hmm-v0.3.0** — topology search by state merge and split, scored by
  minimum description length. See [`planned/04-hmm-v0.3.0.md`](planned/04-hmm-v0.3.0.md).

All three were drafted from a structural survey of `.scratch/hmm-lush/Code/HMMlib/` —
definition maps and call-site counts — **before the source was read**. Each names the
findings that would falsify its boundaries, and revision 02's first subgoal is the reading
that checks them.

## Closed revisions

Extracted by `/close-revision` once finished. The subgoals, their `> **Done:**` records,
and the branch plans that executed them all live in the revision's directory.

- **Revision 01-dataseq-v0.1.0** — closed. See `docs/plan/01-dataseq-v0.1.0/_TODO.md`.
