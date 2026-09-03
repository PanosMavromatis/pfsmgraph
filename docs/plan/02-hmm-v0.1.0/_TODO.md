**Status**: planned — drafted 2026-09-03 on `docs/hmm-migration-plan`, not yet opened
**Splice**: everything from `## Subgoals` down is what `/open-revision` places into
[`docs/plan/TODO.md`](../TODO.md); this preamble stays behind.

**Drafted before the source was read.** The release boundaries below come from a
structural survey of `.scratch/hmm-lush/Code/HMMlib/` — definition maps, call-site counts,
comment headers — not from reading the 2,044 lines. Subgoal 1 is the reading, and its
first duty is to check these boundaries. What would falsify them:

- **Viterbi turning out to depend on the forward variables.** The split assumes
  `update-viterbi-path` (`hmm-trainer.lsh:188-257`) computes δ independently of the α that
  `update-data-p` (`126-188`) builds. If it reads α, the 02/03 boundary moves and Viterbi
  drags the forward pass into this release with it.
- **The stationary-distribution solve being something else.** `hmm-param.lsh:82` and
  `hmm.lsh:244` build a matrix from `int-delta` and call `LU-solve`; that reads as
  `(I - Pᵀ)π = 0`, but it was inferred from two lines of context.
- **`hmm-trainer.lsh:21-126` not being separable.** The scaffolding is assumed shareable
  across 02 and 03. If the constructor demands the training apparatus, 02 gets no trainer
  at all and Viterbi becomes a free function over a model — which may be the better design
  regardless.

## Subgoals — revision 02-hmm-v0.1.0

`dataseq` is released and the family's base layer is fixed, so PRD §11 puts `hmm` next.
This revision is the first of three, and it is deliberately the one that carries the
project's *firsts* rather than the most HMM content: the first dynamic-programming kernel
in the repository, the first `.pyx`, the first non-empty ADR 0003 backend matrix, and the
resolution of the meson-python namespace problem that [ADR 0012](../../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md)
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

- [ ] Read `Code/HMMlib/` in its own terms and write `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, following `ACCOUNT.md`'s conventions — measurements against the two tracked specimen corpora, and **provenance unknown** for behaviours the code admits but may never have exercised. Check the three falsifiers above and revise this plan if any holds.
- [ ] Settle the public surface of `pfsmgraph.hmm` 0.1.0 and where it meets `dataseq`: what a caller constructs, what Viterbi is a method *on* given there is no trainer object in this release, and which of `SymbolTable`, the record container and `pad_collate` it consumes. Apply *encode at the boundary* ([ADR 0001](../../design/adr/0001-encode-at-the-boundary.md)) by naming the exact entry and exit points where strings are still permitted.
- [ ] Migrate the Utility code this release needs, private to the package: `_numeric.py` for `safe-/` (15 call sites), the log-zero sentinel that `safe->--log` implies, and `int-delta`; plus the stationary-distribution solve (`LU-solve` → `numpy.linalg.solve`), `rand-p-vector` for parameter initialisation, and `calculate-entropy`. Record which Numerical-Recipes transcriptions were replaced by a library call rather than translated, and that `minimize`/`mc.lsh` had **zero** call sites from `HMMlib` and so migrate nowhere.
- [ ] Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy) with the ADR 0003 test suite, and register it as the first backend. The session header stops reading `backends: none registered` for the first time since the hook landed.
- [ ] Resolve the meson-python namespace shadowing and move `hmm` off hatchling, reverting [ADR 0012](../../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md) by whichever of its three recorded candidates survives contact: non-editable install of the compiled members, one combined compiled distribution, or an upstream fix. Re-add `meson-python`, `cython` and `ninja` to the root `dev` group.
- [ ] Implement Viterbi at ADR 0002 phase 2 (Cython), the first `.pyx` in a distribution. Backend equivalence against phase 1 is enforced by the parameterized suite, not asserted.
- [ ] Implement Viterbi at ADR 0002 phase 3.
  - [ ] **Settle the anti-diagonal question first — this is the point at which it is strictly needed.** [ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md):53 states the wavefront transformation is "the same transformation for every DP kernel in the family". A structural survey on the planning branch suggested it is not — an HMM recurrence is 1-D over time with dense N×N state coupling, so it has no anti-diagonals, and its parallel decompositions are batch, states-within-a-timestep, and possibly an associative scan over time in the (max, +) semiring. **The finding was deliberately left undecided on the planning branch** because phases 1 and 2 do not depend on it: a Cython kernel is single-threaded, so nothing before this subgoal can falsify or need it. Decide it here, against a kernel that exists, and settle whether it is a wording fix scoped to alignment-family kernels or a reversal warranting its own ADR number.
  - [ ] Implement whichever decomposition that decision names.
- [ ] Release `pfsmgraph-hmm` 0.1.0 via `just release 0.1.0 pfsmgraph-hmm`, shipping the four files the version bump does not imply, and set honest lower bounds on any intra-family dependency naming it.
