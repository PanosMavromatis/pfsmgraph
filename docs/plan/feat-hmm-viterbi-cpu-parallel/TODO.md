# feat/hmm-viterbi-cpu-parallel

**Status**: active
**Created**: 2026-09-09
**Subgoal**: Implement Viterbi at ADR 0002 phase 3 (Numba CPU-parallel, `prange`) —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## How this branch is driven

Same three invocations as phase 2, one row changed: `/dp-compile:phase-check viterbi`
reports, `/dp-compile:next-phase viterbi` resolves the target and routes, and the skill it
routes to here is `dp-compile:cpu-parallelization`. **Enter through `/next-phase`**, for the
same reason as last time — it includes `@references/phase-detection.md`, which carries the
recovered-graph rules a phase skill's own prerequisites do not.

**The expected report before any work.** `_viterbi_cython.pyx`'s provenance header names
`_viterbi.py` at `sha256:48666ca8…`, which matches; `FORMALIZATION.md`'s `Derived from`
names the same file and the same hash. So the graph still branches at the root, nominal
phase is **2**, nothing is stale, and the target is *write `cpu_parallel`*. Confirm rather
than assume — the premise is what the first goal exists to check.

**Editing the formalization invalidates nothing.** It is a leaf: `_viterbi_cython.pyx`
derives from the phase-1 kernel, not from the document, so recording the anti-diagonal
decision in it makes no artifact stale. That is worth knowing before goal 2, because the
instinct on a recovered graph is the opposite.

## Goals

- [x] Confirm the plugin's view of the algorithm matches this branch's premise
  > **Q:** Has `docs/design/algorithms/viterbi/FORMALIZATION.md` been reviewed and
  > approved? Phase 3 is the first phase whose work *edits* it — goal 2 rewrites the
  > `Parallel decomposition` section — so the question is whether it stands as approved
  > contract going in, not merely whether it was approved once.
  > **A:** Yes — approved, and it stands. Goal 2 edits it to record a decision the
  > document itself defers, not to correct anything in it.
  > **Done:** `/dp-compile:phase-check viterbi` reproduces this branch's premise exactly.
  > Nominal phase **2**; `formalization`, `python` and `cython` all fresh; `cpu_parallel`
  > and `cuda` absent; target **write `cpu_parallel`**. Both dependents record
  > `_viterbi.py` at `sha256:48666ca8…` and the recomputed hash matches, so the graph
  > branches at the root rather than running as a chain. Suite green at 291 with
  > `backends: python ✓ · cython ✓` — no backend was withheld, since none is stale or
  > blocked.
  > **Note:** `_viterbi.py` carries no provenance header and the mtime fallback was
  > deliberately **not** applied to it. Two artifacts' headers name it, so that edge is
  > already on record running the other way, and a fallback guess would close a cycle —
  > reporting the kernel stale against a document written *from* it. It is the root: no
  > incoming edge, nothing here can make it stale. Do not "fix" this by adding a header.
  > **Note:** all three `.vpath.xls` oracles the manifest declares are exercised by
  > `packages/pfsmgraph-hmm/tests/` (2, 4 and 6 references, via `_lush_fixtures.load_vpath`).
  > That matters at the *next* goal boundary rather than this one: `/next-phase` gates
  > advancing past phase 1 on the declared oracles actually being used, because phases 2–4
  > are checked against phase 1 and an unexercised oracle leaves the whole chain agreeing
  > only with itself.
- [ ] Settle the anti-diagonal question — the decision this phase owns
  - [ ] Decide the decomposition, against the kernels that now exist
  - [ ] Decide ADR 0002:53's disposition: wording fix scoped to alignment kernels, or a
        reversal warranting its own ADR number
  - [ ] Record it in `FORMALIZATION.md` — Metadata row and `Parallel decomposition`
        section, both of which read **Undetermined** today
- [ ] Decide `numba`'s place in `pfsmgraph-hmm`'s declared dependencies
  - [ ] Runtime dependency or optional extra, given `hardware=None` makes a failed import
        a startup error rather than a skip
  - [ ] Review the `numpy>=2.1` floor alongside it, per `dp-compile.toml`'s pairing note
- [ ] Implement the decomposition under `@njit(parallel=True)`/`prange`
  - [ ] `_viterbi_cpu_parallel.py`, entered through `/dp-compile:next-phase viterbi`
  - [ ] Third `_backends.py` row
  - [ ] Differential tests in the labelled non-shared section, both tie-breaks covered
