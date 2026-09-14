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

The first of the three is **closed** (see below), and `pfsmgraph-hmm` 0.1.0 is released; the
second, `03-hmm-v0.2.0`, is **open** (its subgoals follow this section), and the one below is
still planned. Each entry moves up as its revision is opened.

- **Revision 04-hmm-v0.3.0** — topology search by state merge and split, scored by
  minimum description length. See [`planned/04-hmm-v0.3.0.md`](planned/04-hmm-v0.3.0.md).

All three were drafted from a structural survey of `.scratch/hmm-lush/Code/HMMlib/` —
definition maps and call-site counts — **before the source was read**. Each names the
findings that would falsify its boundaries, and revision 02's first subgoal is the reading
that checks them.

## Subgoals — revision 03-hmm-v0.2.0

Baum-Welch on a fixed topology: the model learns its parameters, but not its shape. This
is the revision where the HMM becomes useful and where the parallelism questions are real,
because a training run touches every sequence in the corpus on every iteration rather than
decoding one at a time. Opened 2026-09-14, the day after `pfsmgraph-hmm` 0.1.0 was released,
from the draft written on `docs/hmm-migration-plan` (2026-09-03). The draft predated
revision 02's work, so it was reconciled against that work at opening. The corrections
are listed below so a reader of the draft can tell what changed and why.

Its distinctive decision is a second implementation held against the first.
**Reverse-mode automatic differentiation computes the backward algorithm.** For
log-parameters, `∂ log P(O|θ) / ∂ log a_ij` is exactly `Σ_t ξ_t(i,j)`, the expected
transition count — so a `torch` forward pass plus `.backward()` yields the E-step's sufficient
statistics, and the M-step is a row normalisation. The numpy reference still writes α and
β explicitly. That the two agree is an unusually strong correctness result, because they
share no code: one is a hand-written recursion, the other is a gradient. Verify the
citation (Eisner 2016, ACL Anthology `W16-5901`; recorded in
[`docs/design/references.md`](../design/references.md) with what it is relied on for)
rather than repeating it on this plan's authority. The bibliographic details are verified;
the *use* made of the paper is not.

**Settled, and not to be relitigated:**

- `torch` is an **optional** backend behind a `pfsmgraph-hmm[torch]` extra, never a
  required dependency. That keeps `core.md`'s invariant that "'GPU' means two unrelated
  things" intact: `numba-cuda` and `torch` remain separate extras with separate meanings.
- **This revision builds the runtime backend seam** (decided at opening). A trainer with a
  public `torch` path is a public call that selects a backend, and 0.1.0 established that
  an extra no public call reaches must not ship. So the seam cannot wait for `align`, and
  `DEFERRED.md`'s `## Trigger: align acquiring a backend-selection API` fires here
  instead. All three of its items become 03's: the selection ADR (raise or fall back on an
  unavailable backend), parameterising the suites over backends as ADR 0003 requires, and
  restoring the `gpu`/`cpu-parallel` extras along with the platform wheels that end
  0.1.0's pure wheel. That trigger's heading names `align`, so update it when the items
  move rather than leaving it to fire twice.
- **The class architecture is inherited, not reopened.**
  [ADR 0017](../design/adr/0017-frozen-parameter-object-for-hmm.md) makes parameters a
  frozen `HMMParams` value and algorithms free functions over it, so training *returns* a
  new `HMMParams` instead of mutating one. The one place this revision can still put
  pressure on that is the `torch` backend: wrapping the same arrays as `nn.Parameter`s is
  itself a class shape. Any friction there belongs to the `torch`-backend subgoal, which
  must absorb it without redesigning `HMMParams`.
- **No log-zero sentinel.** Revision 02 declined the Lush `-1`; `bits(0)` is `+inf`. The
  draft's first subgoal said to use "the log-zero sentinel `_numeric.py` fixed in
  revision 02", which described a plan that was not carried out.

**Falsifiers, reconciled at opening.** The draft was written before the source was read.
Two of its three falsifiers are now resolved, and the third has changed shape:

- **The M-step not being separable from the search** — does not hold. `run-add`
  (`hmm-trainer.lsh:483-652`) is genuine Baum-Welch over α/β/ξ and anticipates no state
  merge/split; `update-approx-*` (`257-346`) is parameter quantization for the description
  length (`HMMLIB-ACCOUNT.md` §8-9, whose "Corrected region map" appendix flags the
  draft's original mislabeling). The line range still matters to revision 04, whose
  `try-split`/`try-merge` call back into whatever this revision builds around it.
- **`run-add` not being the EM loop** — refuted by the same reading; it stays in 03. The
  draft feared its name meant adding states, which would have moved it (173 lines, cited
  there as `477-656` from the structural survey) to 04 and left `run-converge` (`656-677`)
  as this revision's only driver. The subgoals below use the corrected ranges.
- **The autograd identity not surviving the arithmetic.** The Lush code accumulates in
  base 2 throughout because its description lengths are in bits; nothing about the
  gradient identity depends on the base. The draft's concern was whether the sentinel
  arithmetic in `safe-add--log2` could be expressed as a differentiable op. That concern is
  spent, since no sentinel was ported. What replaces it is `+inf` bits, i.e. `-inf`
  log-probabilities, inside a differentiable log-sum-exp. A dead arc produces `-inf`. The forward value survives that, but a gradient
  through it can come out `nan`. If it does, masking dead arcs out of the sum is the likely
  fix, and it keeps the test an exact equivalence. If masking cannot keep the result exact,
  the `torch` subgoal weakens from an equivalence test to a tolerance comparison, and must
  say so.

- [x] Unify the logarithm across the Viterbi backends before the forward pass consumes `bits` (`DEFERRED.md`, `## Trigger: revision 03 or 04 opening`). Phase 1 uses numpy's vectorised `log2` and phases 2-3 libm's scalar one, one ulp apart in ~0.1% of inputs; a sum reduction would inherit that as a tolerance nobody chose. Build every backend's arc costs from one numpy table (phase 4's approach) or route phase 1 through libm; either reopens phases 2 and 3 and regenerates their provenance headers, and `FORMALIZATION.md` must name *which* `log2` is contract.
  > **Branch:** fix/hmm-viterbi-log2
  > **Done:** All four Viterbi backends take their logarithms from `bits` on the host (numpy's `log2`) and perform only `+` and `<`, so they are bit-exact with one another on a given host; phases 2-3 adopted phase 4's `(S, S, U)` table without reopening phase 1. `FORMALIZATION.md` names the contract and adds TC-21, a constructed exact tie that the pre-fix kernel fails with a different path and an identical `total_bits` — PR #28
- [x] Implement forward-backward in numpy as the reference: α, β, ξ, γ, in log space, with `+inf` bits for impossible events. State whether the base stays 2 — natural for the description lengths of revision 04, unusual everywhere else — or whether base *e* is used with a conversion at the DL boundary.
  > **Branch:** feat/hmm-forward-backward
  > **Note:** The base question is settled by [ADR 0020](../design/adr/0020-scaled-probability-domain-forward-backward.md), and not in the form asked. The original never works in log space: it runs Rabiner-scaled forward-backward on probabilities and takes `log2` only of the scale factors. The reference does the same, in a fixed evaluation order, so the base is 2 and "in log space" above is superseded.
  > **Done:** `_forward_backward.py` holds the scaled α/β kernel, the description length, γ, and the expected counts `C-in`/`C-t`/`C-t-y`, built without storing ξ. Every reduction is `np.add.accumulate` in ADR 0020's fixed order. 108 tests check them against exact rational enumeration over every state path, constructed impossible and uniform cases, and a bit-for-bit scalar-loop reference. The kernel is registered with `dp-compile` at phase 1 and is not yet exported — PR #30
- [x] Implement the M-step and the EM loop: parameter re-estimation (`run-add`, `hmm-trainer.lsh:483-652`), the convergence criterion (`run-converge`, `661-674`), and the data description length (`update-data-dl`, `hmm-trainer.lsh:346-402`), which shares its accumulation shape with the likelihood and should share its code.
  > **Branch:** feat/hmm-em-loop
  > **Note:** `run-converge` can stop *on* a symmetric saddle, since it watches only the size of each batch's change, and an exactly uniform start (`rand_p_vector(noise_width=0)`) is one. Whatever initialises a model, here or in revision 04's `split-state`, must break symmetry. Also: `_total_dl` is `data-dl + model-dl`, so it becomes an oracle only once revision 04 lands the model half; the training logs' `data-dl` column already is one.
  > **Done:** `_baum_welch.py` holds `_m_step` (`run-add`'s stage 5 with `safe_divide` and ascending reductions), `_em` (counts summed over records, `run-converge`'s stopping rule, a zero-count state's previous row and fibres restored, with the likelihood bit-identical to the original's zero row) and `_data_description_length` at a given `d`, which matches the three tracked models' logged `data-dl` to 0.009 bits. All private; 78 tests, each piece mutation-checked. Choosing `d` and the model half stay with revision 04 — PR #31
- [ ] Validate the numpy reference against an **external oracle** before anything is built on it. An arc-emission model whose emission depends only on the *destination* state is a state-emission model ([ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md)), so construct that reduced case and check the forward-backward — and the fitted parameters after EM — against `hmmlearn`'s `CategoricalHMM` to machine precision. This catches a class of error the torch backend below **cannot**: autograd and a hand-written β are two implementations of the same derivation, so they can agree on a shared misreading of the model, whereas `hmmlearn` shares no lineage with either. Write it as early as the numpy forward-backward allows rather than leaving it until the EM loop is finished too. `hmmlearn` is a **test-only** dependency in the root `dev` group and reaches no shipped artifact; it is explicitly *not* an [ADR 0003](../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) backend, since a backend is another implementation of the same kernel and this is a different model that coincides with ours on one special case. Source: [`arc-emission-hmm-handoff.md`](../design/arc-emission-hmm-handoff.md) §3.
  > **Branch:** feat/hmm-hmmlearn-oracle
  > **Note:** "the fitted parameters after EM … to machine precision" cannot be met as worded. One arc-emission M-step unties the destination-only emission and counts the `s_0 → s_1` crossing, so `_em`'s output leaves the family `CategoricalHMM` can express. The forward-backward is compared directly (3e-15, with `startprob_ = init_state_p @ transition_p`); EM is compared through a start-state and tied-emission harness over `_corpus_step` and `_m_step` (1.2e-14 over 200 cycles). ADR 0015's Resolved section has the detail.
- [ ] Write the runtime backend-selection ADR and build the seam it decides: a public way to choose the training (and decode) backend, what happens when a requested backend is unavailable, and the ADR 0003 suites parameterised over it — folding `test_viterbi.py`'s labelled non-shared section into the shared tests. Restore the accelerator extras withheld from 0.1.0 as part of it (`DEFERRED.md`, `## Trigger: align acquiring a backend-selection API`).
- [ ] Add the `torch` backend behind the `[torch]` extra: the forward pass and the autograd E-step. **Assert the count identity** — numpy's explicit ξ and γ against torch's `.grad` on the log-parameters — as an ADR 0003 cross-backend test through the seam above rather than a tolerance check, and record what tolerance was actually needed.
- [ ] Batch the trainer over sequences. This is the first place `pad_collate`'s mask does real work: padded timesteps must contribute nothing to the expected counts, and a mask bug here is silent, since it shifts the estimates rather than raising.
- [ ] Migrate the training log as **standard-output reporting only**: `update-training-log` and `training-log-line` (`hmm-trainer.lsh:457-477`), plus `save-training-log` (`hmm.lsh:166`). No GUI, no widget, no plotting dependency — see `DEFERRED.md`, `## Trigger: a second module needing training progress reporting`. The assumption is a Jupyter notebook whose output cell is browsed while the run continues, so the line format has to stay readable when hundreds of them accumulate in one cell.
- [ ] Carry the forward recurrence through ADR 0002 phases 2 through 4 ([ADR 0016](../design/adr/0016-numba-cpu-parallel-phase.md) inserted a CPU-parallel phase 3, renumbering CUDA to phase 4), starting with a phase-0 `FORMALIZATION.md`. Settle whether the max-plus/logsumexp associative scan over time is worth implementing — wording kept from the draft on purpose: which semiring it should name depends on the base decision in the forward-backward subgoal, and `02-hmm-v0.1.0/_TODO.md` records why it was not silently corrected — or whether batch parallelism alone is the phase-3 answer; Viterbi's phase 3 found a per-timestep parallel region costs a flat fork/join per step, so the batch axis is the obvious candidate.
  > **Note:** The semiring is named by [ADR 0020](../design/adr/0020-scaled-probability-domain-forward-backward.md): sum-product with per-step renormalisation, neither max-plus nor log-sum-exp. The ADR also makes evaluation order and the absence of fused multiply-add part of the contract these phases inherit, and leaves numba-cuda's contraction behaviour open for phase 4.
- [ ] Release `pfsmgraph-hmm` 0.2.0 — the first release with platform wheels, if the seam subgoal ships a public call that reaches a compiled kernel.

## Closed revisions

Extracted by `/close-revision` once finished. The subgoals, their `> **Done:**` records,
and the branch plans that executed them all live in the revision's directory.

- **Revision 01-dataseq-v0.1.0** — closed. See `docs/plan/01-dataseq-v0.1.0/_TODO.md`.
- **Revision 02-hmm-v0.1.0** — closed. See `docs/plan/02-hmm-v0.1.0/_TODO.md`.
