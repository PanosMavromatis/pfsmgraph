# docs/hmm-migration-plan

**Status**: active
**Created**: 2026-09-03
**Subgoal**: standalone — revision 01 is closed and no revision is open; this branch drafts revisions 02, 03 and 04 and opens the first of them

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

This branch plans the `hmm` migration and writes no code under
`packages/pfsmgraph-hmm/src/`. Its output is three revision plans, the ADRs their
decisions warrant, and revision 02 opened for a later branch to execute.

**The source reading is not here.** `.scratch/hmm-lush/ACCOUNT.md` records itself as
"Subgoal 2 of goal 3, `feat-dataseq-merge`" — the `dataseq` source was read *inside* the
branch that migrated it, not in a planning branch ahead of it. `HMMLIB-ACCOUNT.md` follows
that precedent and is revision 02's first subgoal. The cost is that the release plans are
drafted from a structural survey rather than a reading, which is why each one names the
findings that would falsify its boundaries.

## Goals

- [x] Draft the three revision plans and register them in the master plan
  - [x] Survey `Code/HMMlib/` structurally — definition maps, call-site counts from `Utility`, comment headers — enough to place boundaries, not enough to claim the code has been read
  - [x] Split the migration three ways and check the split against the source's own seams
  - [x] Write `docs/plan/0{2,3,4}-*/_TODO.md` in the shape `/close-revision` leaves behind, so opening one is a splice
  - [x] Add a `## Planned revisions` section to `docs/plan/TODO.md`, and amend its "tracks only work that is active now" claim, which the section falsifies
  > **Done:** three revision directories, each with a `_TODO.md` carrying a
  > `**Status**: planned` preamble that is dropped at splice time. The status line is new
  > — revision 01's `_TODO.md` has none, because until now every revision directory on
  > disk was closed by definition and the master plan was the only registry. With planned
  > revisions on disk that inference fails, so the registry entry is load-bearing.
  > **Found:** the three-way split falls on method boundaries in the Lush trainer, not
  > across them. `update-viterbi-path` is `hmm-trainer.lsh:188-257`; the M-step is
  > `257-346`; topology search is `738-1073`, a clean 335-line tail. The one place the
  > split cuts awkwardly is that **Viterbi is a trainer method**, so revision 02 cannot
  > take `hmm.lsh` alone and must decide where a decode belongs when there is no trainer.

- [x] Settle the decisions the split forces, and log them
  > **Q:** numpy or PyTorch — does autograd facilitate Baum-Welch, and do torch ops
  > facilitate parallelization?
  > **A:** numpy is the reference and the only required dependency; `torch` enters at
  > revision 03 behind a `[torch]` extra as an ADR 0003 backend. Autograd does facilitate
  > Baum-Welch, decisively and by identity rather than by heuristic: for log-parameters,
  > ∂ log P(O|θ) / ∂ log a_ij = Σ_t ξ_t(i,j), so reverse-mode AD through the forward pass
  > *is* the backward algorithm and the E-step falls out of `.grad`. Keeping it optional
  > buys the equivalence as a **test** — numpy's hand-written β against torch's gradient,
  > sharing no code — without making a ~2GB dependency mandatory for anyone who wants an
  > HMM, and so without disturbing `core.md`'s invariant that "'GPU' means two unrelated
  > things". On parallelism: torch wins decisively on the batch dimension, ties on states
  > (both are BLAS), and gives nothing for free over time.
  > **Q:** where does the migrated `Utility` code live?
  > **A:** private to `pfsmgraph.hmm` as `_mdl.py` and `_numeric.py`, with a
  > `DEFERRED.md` trigger to promote it if `hseg` scores segmentations by description
  > length too. A sixth distribution before a second consumer exists is not warranted.
  > **Q:** revision 02 fires the first-`.pyx` trigger. Isolate the packaging work?
  > **A:** carry it in 02 as its own subgoal, between the pure-Python kernel and the
  > Cython one.
  > **Found:** most of `Utility` does not migrate at all, and this was measured rather
  > than estimated. `LU-solve` (2 uses, both solving for the stationary distribution) is
  > Numerical Recipes in C and becomes `numpy.linalg.solve`; the same LU code appears
  > **three times** in the tree. `minimize` (Brent, also NR in C) and all of `mc.lsh` have
  > **zero** call sites from `HMMlib`. What must migrate is narrower and more interesting:
  > the MDL machinery — `int-code-length`, `comb-code-length`, `calculate-entropy` — which
  > is the topology-search criterion and has no library equivalent, and the numerical
  > guards, where `safe-/` alone has 15 call sites and the representation of log-zero is a
  > decision that propagates through every recurrence.
  > **Q:** `hmm-trainer-view.lsh` (237 lines) — in the distribution, out, or `dl`'s?
  > **A:** out, and no replacement. Training reports to standard output and nothing else,
  > on the assumption of a Jupyter notebook whose output cell can be browsed
  > independently while a run continues. Dashboards are deferred family-wide rather than
  > for `hmm` alone, because `dl` will want the same thing and a framework adopted for
  > one member would either be duplicated or forced on a family that has no GUI
  > dependency today. Filed under `DEFERRED.md`,
  > `## Trigger: a second module needing training progress reporting` — a **second**
  > consumer, because that is the first point at which two sets of requirements can be
  > compared rather than one guessed at.
  > **Found:** the source already contains the non-GUI half. `update-training-log` and
  > `training-log-line` (`hmm-trainer.lsh:457-477`) and `save-training-log`
  > (`hmm.lsh:166`) are text reporting that predates the view layer, so "out" removes a
  > file rather than a capability. Migrating them is revision 03's subgoal.

- [ ] Write the ADRs the decisions warrant
  - [-] **ADR 0002's anti-diagonal claim.** Deferred by decision, not by omission: moved to revision `02-hmm-v0.1.0`'s phase-3 subgoal, which is the first point at which anything depends on the answer. The finding stands — line 53 claims the wavefront is "the same transformation for every DP kernel in the family", and an HMM recurrence is 1-D over time with dense N×N state coupling — but it was reached by structural survey, phases 1 and 2 cannot falsify or need it, and deciding it against a kernel that exists beats deciding it against one that does not. Whether it is a wording fix scoped to alignment-family kernels or a reversal warranting a new number is part of what is deferred.
  - [ ] **`torch` as an optional backend for a DP package.** Its relationship to [ADR 0004](../../design/adr/0004-gpu-backends-and-optional-dependency-strategy.md) and to the "two GPUs" invariant, which it clarifies rather than reverses: `torch` may appear in a DP member as an optional backend, never as a required dependency, and the two GPU extras stay separate.
  - [ ] Take the next unused numbers, and add a row to `docs/design/adr/README.md` for each
  - [ ] Consider whether the autograd/forward-backward equivalence is itself ADR material, or belongs in `docs/api/hmm/` when that directory exists

- [ ] Record what must land with its trigger rather than after it
  - [x] Add `## Trigger: a second module needing training progress reporting` to `DEFERRED.md`, scoped family-wide rather than to `hmm`
  - [x] Record the Eisner citation in `docs/design/references.md`, with what it is relied on for and an explicit note that its bibliographic details are verified while the use made of it is not
  - [ ] Add `## Trigger: hseg needing description lengths` to `DEFERRED.md` for promoting `_mdl.py` out of `hmm`
  - [ ] Note under `## Trigger: the first .pyx` that revision 02 is what fires it, so the ADR 0012 revert and the meson-python namespace resolution are that revision's subgoal 5 and not a later cleanup
  - [ ] Sweep `docs/agents/core.md` for claims this branch falsifies — the "Still to do, in PRD order" line, the ADR 0002 summary under **Invariants**, and the backend-matrix paragraph's "until then every run opens with `backends: none registered`"

- [ ] Open revision 02 and hand off
  - [ ] Splice `docs/plan/02-hmm-v0.1.0/_TODO.md` from `## Subgoals` down into `docs/plan/TODO.md`, leaving its `**Status**: planned` preamble behind
  - [ ] Move revision 02's entry out of `## Planned revisions`
  - [ ] Confirm `/new-branch` can take the first subgoal cleanly — a revision is open, so the next branch backlinks to it rather than running standalone as this one did
