**Status**: planned — drafted 2026-09-03 on `docs/hmm-migration-plan`, not yet opened
**Splice**: everything from `## Subgoals` down is what `/open-revision` places into
[`docs/plan/TODO.md`](../TODO.md); this preamble stays behind and this file is removed.
It lives in `planned/`, not in `docs/plan/03-hmm-v0.2.0/`, because `/open-revision` refuses a
label whose directory already exists.

**Only metadata is above the heading, and deliberately so.** The falsifiers sit *below*
`## Subgoals` because the splice boundary would otherwise leave them here to be deleted
with this file — which is what happened when revision 02 was opened, and had to be
repaired by hand. Do not tidy them back up here: everything that must survive the
splice belongs under the heading.

## Subgoals — revision 03-hmm-v0.2.0

Baum-Welch on a fixed topology: the model learns its parameters, but not its shape. This
is the revision where the HMM becomes useful and where the parallelism questions are real,
because a training run touches every sequence in the corpus on every iteration rather than
decoding one at a time.

Its distinctive decision is a second implementation held against the first.
**Reverse-mode automatic differentiation computes the backward algorithm.** For
log-parameters, `∂ log P(O|θ) / ∂ log a_ij` is exactly `Σ_t ξ_t(i,j)`, the expected
transition count — so a `torch` forward pass plus `.backward()` yields the E-step's sufficient
statistics, and the M-step is a row normalisation. The numpy reference still writes α and
β explicitly. That the two agree is an unusually strong correctness result, because they
share no code: one is a hand-written recursion, the other is a gradient. Verify the
citation (Eisner 2016, ACL Anthology `W16-5901`; recorded in
[`docs/design/references.md`](../../design/references.md) with what it is relied on for)
rather than repeating it on this plan's authority. The bibliographic details are verified;
the *use* made of the paper is not.

Settled on the planning branch: `torch` is an **optional** backend behind a
`pfsmgraph-hmm[torch]` extra, never a required dependency, so `core.md`'s invariant that
"'GPU' means two unrelated things" survives intact — `numba-cuda` and `torch` remain
separate extras with separate meanings.

**Drafted before the source was read.** What would falsify the shape below:

- **The M-step not being separable from the search.** `update-approx-*`
  (`hmm-trainer.lsh:257-346`) is assumed to be plain expected-count accumulation. If it
  already anticipates state merge/split, part of it belongs to revision 04.
- **`run-add` (173 lines, `477-656`) not being the EM loop.** Its name suggests adding
  states, which would make it topology search and move it to 04, leaving `run-converge`
  (`656-677`) as this revision's only driver.
- **The autograd identity not surviving the log₂ base.** The Lush code accumulates in
  base 2 throughout because its description lengths are in bits. Nothing about the
  gradient identity depends on the base, but the sentinel arithmetic in `safe-add--log2`
  might not be expressible as a differentiable op, which would weaken subgoal 3 from an
  equivalence test to a numerical-tolerance comparison.

- [ ] Implement forward-backward in numpy as the reference: α, β, ξ, γ, in log space, with the log-zero sentinel `_numeric.py` fixed in revision 02. State whether the base stays 2 — natural for the description lengths of revision 04, unusual everywhere else — or whether base *e* is used with a conversion at the DL boundary.
- [ ] Implement the M-step and the EM loop: parameter re-estimation, the convergence criterion, and the data description length (`update-data-dl`, `hmm-trainer.lsh:346-402`), which shares its accumulation shape with the likelihood and should share its code.
- [ ] Add the `torch` backend behind the `[torch]` extra: the forward pass and the autograd E-step. **Assert the count identity** — numpy's explicit ξ and γ against torch's `.grad` on the log-parameters — as an ADR 0003 cross-backend test rather than a tolerance check, and record what tolerance was actually needed.
- [ ] Batch the trainer over sequences. This is the first place `pad_collate`'s mask does real work: padded timesteps must contribute nothing to the expected counts, and a mask bug here is silent, since it shifts the estimates rather than raising.
- [ ] Migrate the training log as **standard-output reporting only**: `update-training-log` and `training-log-line` (`hmm-trainer.lsh:457-477`), plus `save-training-log` (`hmm.lsh:166`). No GUI, no widget, no plotting dependency — see `DEFERRED.md`, `## Trigger: a second module needing training progress reporting`. The assumption is a Jupyter notebook whose output cell is browsed while the run continues, so the line format has to stay readable when hundreds of them accumulate in one cell.
- [ ] Carry the forward recurrence through ADR 0002 phases 2 and 3, and settle whether the max-plus/logsumexp associative scan over time is worth implementing or whether batch parallelism alone is the phase-3 answer for this family of kernels.
- [ ] Release `pfsmgraph-hmm` 0.2.0.
