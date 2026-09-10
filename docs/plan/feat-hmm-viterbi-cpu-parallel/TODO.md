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
- [x] Settle the anti-diagonal question — the decision this phase owns
  > **Q:** Which parallel decomposition does phase 3 implement for Viterbi — states within
  > one timestep (`prange` over `j`), an associative scan over time in min-plus, or a batch
  > of independent sequences?
  > **A:** States within one timestep. It is the only one available at this signature and
  > the only one that preserves the tie-break; see the notes below.
  > **Q:** How should ADR 0002's falsified generality claim be recorded — closed in ADR
  > 0016's `Resolved` section, edited in place in ADR 0002, or given a new ADR number?
  > **A:** Closed in ADR 0016's `Resolved` section, with a pointer added to ADR 0002's
  > Status line and that record's text left untouched.
  > **Done:** all three artifacts written and the suite green at 291.
  > `FORMALIZATION.md`'s Metadata row and `Parallel decomposition` section now record the
  > decision instead of **Undetermined**; ADR 0016's sole `Open` item moved to `Resolved`
  > as the withdrawal of ADR 0002's generality claim; ADR 0002 gained a Status pointer and
  > nothing else. What is withdrawn is a *generality*, not a decision — the four-phase
  > lifecycle stands, and the anti-diagonal wavefront still stands for `align`, whose
  > two-dimensional matrix does have independent anti-diagonals.
  > **Note:** the finding is sharper than "no anti-diagonals", and the sharper form is
  > checkable. In an alignment matrix `{i + j = c}` is independent *because* a cell reads
  > only its three neighbours. Here the array is time by state and `delta[t, j]` reads
  > **every** `delta[t-1, i]`, so `{t + j = c}` contains `(t, j)` and `(t-1, j+1)` and the
  > first reads the second as its `i = j+1` term. Anti-diagonal structure in an `hmm`
  > kernel is therefore the defect itself, not an optimisation to review for correctness.
  > **Note:** the tie-break decides which axis may be parallelised, which was not obvious
  > going in. `prange` over `j` keeps each reduction over `i` serial and ascending, so
  > first-wins survives; `prange` over `i` is a parallel reduction that combines partials
  > in unspecified order and resolves ties arbitrarily. It would fail **silently** — the
  > oracles hold 0 exact ties in 3804 positions, so no differential test can see it, and
  > the constructed uniform-model case from phase 2 is the only guard. That test was
  > written for revision 03's initialisation, not for concurrency; it now guards a design
  > choice it was never aimed at.
  > **Note:** the parallelism is thin and this kernel may be *slower* than phase 2. `S` is
  > 5 and 8 in the fixtures and around 50 at the ceiling, so `prange` over `j` offers at
  > most `S`-way parallelism over an `S`-element reduction, and per-timestep thread
  > overhead may exceed the work. Recorded in advance rather than discovered at goal 4:
  > ADR 0016 scopes phase 3 to parallel *correctness*, so losing on wall-clock is within
  > what the phase is for and is not a reason to abandon the decomposition.
  > **Note:** the line citations rotted twice, and both records now cite by section
  > instead. This plan and the branch doc said `ADR 0002:53` for a claim that was at `:55`;
  > adding the withdrawal pointer to ADR 0002's Status block then shifted it to `:59`,
  > invalidating the citation in the same change that wrote it. A citation into a record
  > the same change is editing cannot be a line number.
  > **Note:** `docs/agents/codex.md` still tells a reviewer the `Parallel decomposition`
  > section is "`undetermined` pending phase 3", which these edits make false. Left for
  > `/agents-docs-update` at commit time rather than fixed here — it is an agent-docs
  > source, and editing it out of band skips the `/agents-docs-build` rebuild that
  > sequences with it.
  - [x] Decide the decomposition, against the kernels that now exist
  - [x] Decide ADR 0002:53's disposition: wording fix scoped to alignment kernels, or a
        reversal warranting its own ADR number
  - [x] Record it in `FORMALIZATION.md` — Metadata row and `Parallel decomposition`
        section, both of which read **Undetermined** today
- [x] Decide `numba`'s place in `pfsmgraph-hmm`'s declared dependencies
  > **Q:** How should numba be declared — a new optional extra, a hard runtime
  > dependency, or folded into the existing `gpu` extra?
  > **A:** A new optional extra, `cpu-parallel = ["numba>=0.61"]`.
  > **Q:** The `cpu_parallel` row's absence is legitimate but is not hardware. How should
  > `_backends.py` express that — rename the field, keep `hardware` with a strained value,
  > or keep `hardware=None` and hard-fail?
  > **A:** Rename the field to `optional_on`.
  > **Done:** `packages/pfsmgraph-hmm/pyproject.toml` gains
  > `cpu-parallel = ["numba>=0.61"]` beside the untouched `gpu` extra; the root `dev` group
  > gains `numba>=0.61`; `_backends.py`'s `hardware` field is now `optional_on`, with the
  > three registry comments, `detect()` and four references in `tests/test_backends.py`
  > following. `uv lock` resolved 56 packages and `uv sync` installed numba 0.67.0 and
  > llvmlite 0.49.0. Suite green at 291 — the rename broke nothing, which is what being
  > test-only infrastructure buys.
  > **Note:** numba imposes a numpy **upper** bound, and that is what settled the extra
  > question. Measured 2026-09-10: `numba>=0.61` with `numpy>=2.1` resolves to numba 0.67 /
  > numpy 2.5.3, but pinning `numba==0.61.*` caps numpy at **2.2.6**. As a hard dependency
  > that ceiling would bind every consumer of `pfsmgraph-hmm`, including one that only ever
  > runs the pure-Python decode. Note the direction: this is the mirror image of the
  > workspace footgun `core.md` documents — that one is about *our* lower bounds never
  > failing locally, this is someone else's ceiling arriving transitively. Neither is
  > visible in `uv.lock`, which records one resolution rather than the space of them.
  > **Note:** the `numpy>=2.1` floor stands unchanged, and numba's ceiling was deliberately
  > **not** mirrored into our own declaration. The floor's justification is still the
  > compiled one — a C extension built against numpy 2.x headers will not run on 1.x — and
  > adding an upper bound we did not derive would be drift, not caution. numba declares its
  > own ceiling; a resolver already reads it.
  > **Note:** no `requires-python` conflict. numba 0.67 resolves on 3.10, 3.11, 3.13 and
  > 3.14, so `hmm`'s `>=3.10` is safe. Checked rather than assumed, because numba has
  > historically lagged new Python releases and the failure would land at a consumer's
  > install rather than here.
  > **Note:** the `gpu` extra already pulls numba transitively — `numba-cuda` depends on
  > `numba` and `llvmlite`. So phase 4's extra satisfies phase 3's requirement for free,
  > which is coherent given phase 4 reuses the decomposition phase 3 validates. It is not
  > a reason to fold the two together: ADR 0004 refuses to unify unrelated accelerators,
  > and a CPU backend behind a `[gpu]` install is exactly that.
  > **Note:** `numba` is in the root `dev` group as well as in the extra, and that half was
  > forced rather than chosen. Without it a plain `uv sync` leaves the phase-3 backend
  > unexercised, and validating the decomposition under real concurrency is the entire
  > stated purpose of the phase — a parallel backend nobody runs is worse than none.
  > **Note:** the row's `optional_on` value will read `"numba"`, not `"the cpu-parallel
  > extra"` as the decision preview showed. `detect()` builds its reason as
  > `f"no {optional_on} detected"`, which the longer phrase renders ungrammatical. The
  > extra is named in the registry docstring instead, where a reader adding a row is
  > already looking. The row itself lands in the next goal.
  > **Note:** `docs/agents/core.md` still says both backend rows "carry `hardware=None`",
  > which the rename makes false. Left for `/agents-docs-update` at commit time, same
  > routing as the `codex.md` correction in the previous goal.
  - [x] Runtime dependency or optional extra, given `hardware=None` makes a failed import
        a startup error rather than a skip
  - [x] Review the `numpy>=2.1` floor alongside it, per `dp-compile.toml`'s pairing note
- [ ] Implement the decomposition under `@njit(parallel=True)`/`prange`
  - [ ] `_viterbi_cpu_parallel.py`, entered through `/dp-compile:next-phase viterbi`
  - [ ] Third `_backends.py` row
  - [ ] Differential tests in the labelled non-shared section, both tie-breaks covered
