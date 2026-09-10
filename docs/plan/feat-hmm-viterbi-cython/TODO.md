# feat/hmm-viterbi-cython

**Status**: active
**Created**: 2026-09-09
**Subgoal**: Implement Viterbi at ADR 0002 phase 2 (Cython), the first `.pyx` in
a distribution — `docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## How this branch is driven

Phases 2, 3 and 4 run under the `dp-compile` plugin, so this is the plugin's first
real use on the kernel it was revised for. Two commands and one skill are in scope:

| Invocation | Role |
|---|---|
| `/dp-compile:phase-check viterbi` | Reporting only. Never advances, never suggests advancing. |
| `/dp-compile:next-phase viterbi` | Resolves the target and routes to the skill for **the target's own phase**. |
| `dp-compile:cython-translation` | The phase-2 skill `/next-phase` routes to. Not invoked directly — see below. |

**Enter through `/next-phase`, not by invoking `dp-compile:cython-translation`
directly.** The two are not equivalent here. `/next-phase` includes
`@references/phase-detection.md`, which carries the recovered-graph rules; the skill's
own prerequisite 2 does not, and defines phase-1 staleness as *"the hash recorded in
its `derived-from` header no longer matches the formalization's current hash"* — a
header this kernel correctly does not have. Routing through `/next-phase` is what
supplies the missing carve-out.

**Phase 0 already exists, and `algorithm-recover` has already run.** This Viterbi did
not begin in a natural-language specification: it was translated from Lush
(`update-viterbi-path`, `hmm-trainer.lsh:216-227`), so `algorithm-formalize` never ran
and never could have. The recovery entrance was taken instead, on 2026-09-09 in PR #20.
Verified on this branch before any of the goals below were written:

- `docs/design/algorithms/viterbi/FORMALIZATION.md` exists, and its Metadata table
  reads `Source: **Recovered from the phase-1 implementation.**`
- Its `Derived from` row names `_viterbi.py` at
  `sha256:48666ca8a3126ba141ec1a0ddba8ed1823e502f93d72483181906afd24135742`, which
  **matches** the current file. The document is `fresh`, so phase 0 is not a target and
  `algorithm-recover` does not need re-running.
- `_viterbi.py` carries **no** `# dp-compile: derived-from` line, which is correct
  rather than missing. In a recovered graph the kernel is the root: it derives from
  nothing the plugin tracks, and `phase-detection.md` forbids the mtime fallback giving
  it an incoming edge from the document that was written from it — that would report
  the kernel stale the moment a recovery lands and regenerate it from a description of
  itself.

So the graph branches at the root rather than running as a chain, and the expected
`/phase-check` report is the last row of `phase-detection.md`'s worked-examples table:
nominal phase **1**, `formalization` fresh, `cython` / `cpu_parallel` / `cuda` absent,
target **write `cython`**.

## Goals

- [x] Confirm the plugin's view of the algorithm matches this branch's premise
  > **Q:** Has `docs/design/algorithms/viterbi/FORMALIZATION.md` been reviewed and
  > approved? (`/phase-check`'s phase-0 validation requires asking; it is the one thing
  > in the report that is not derivable from the files.)
  > **A:** Yes — reviewed and approved. Phase 0 stands as the contract phase 2 is written
  > against, and the `.pyx` is written to the document as it is.
  > **Done:** the plugin's report reproduces the premise exactly. Nominal 1, nothing
  > stale, target *write `cython`* — the last row of `phase-detection.md`'s
  > worked-examples table. The branch proceeds.
  - [x] Run `/dp-compile:phase-check viterbi` and record the report verbatim. Expect
        nominal 1 and target *write cython*; anything else means the premise above is
        wrong and the branch stops here rather than proceeding on it
    > **Ran:** 2026-09-09. Nominal phase **1**. `formalization` **fresh** — recomputed
    > `sha256:48666ca8…` for `_viterbi.py` with its own provenance line stripped (none
    > to strip) and it matches the recorded hash byte for byte. `python` **fresh**, as a
    > root: no ancestors, so freshness's "no ancestor is stale or absent" clause is
    > vacuous and its hash clause has nothing to check. `cython`, `cpu_parallel`, `cuda`
    > all **absent**; all five phases are declared, so none is a not-applicable row.
    > Target **write `cython`**. Tests for fresh backends: `uv run pytest
    > packages/pfsmgraph-hmm/tests/test_viterbi.py` → **57 passed**; nothing was skipped
    > for staleness because nothing is stale.
    > **Note:** all three declared oracles are exercised — `m001_0001_001` and
    > `m001_0005_005` at `test_viterbi.py:222`, `m008_0001_008` at `:229` and `:243`,
    > all through `load_vpath`. No declared oracle is unused, which is the check that
    > matters most here and is re-run by `/next-phase`'s step-3 gate.
  - [x] Note whether the report names `_viterbi.py` under "any provenance header that
        was missing". It should — the header genuinely is absent — but the mtime
        fallback must **not** be reported as in use for it, since the recorded edge
        from the formalization already runs the other way
    > **Note:** it does, and the fallback is correctly suppressed. The report names the
    > file as headerless *and* states that mtime is not in use for it, because the
    > formalization's header already names it. That is the exception `phase-detection.md`
    > spells out at length: the fallback must never reverse an edge already on record.
    > Consequence worth keeping — the absent header is **not** a defect to tidy up. Adding
    > one would close a cycle, and a recovered document is always newer than its source,
    > so mtime would report the kernel stale the instant the recovery landed and
    > regenerate the implementation from a description of itself.
  - [x] Confirm the report says which entrance produced phase 0. `/phase-check`
        requires this, and "recovered" is the answer a reader needs in order to weigh
        how far the document can be trusted: it was read off the code it now governs
    > **Note:** it does — `Source` reads **"Recovered from the phase-1 implementation."**
    > All mandatory sections are filled (350 lines), and `Parallel decomposition` records
    > *"Undetermined — and that is the finding, not a deferral"*, which is an answer
    > rather than a gap: the recurrence is one-dimensional over time with dense `S×S`
    > coupling, so it has no anti-diagonals for phases 3 and 4 to exploit.
- [x] Settle what this branch may claim about equivalence
  > **Q:** How far should the master-plan correction go — line 200 only, both lines, or
  > neither, given the phase-1 note already records the tension?
  > **A:** Both lines, and requalify the note. The phase-1 note quotes line 200 verbatim as
  > "the line below", so correcting the line would silently falsify a true record.
  > **Q:** The phase-1 note says the parameterization prerequisite "is not scheduled
  > anywhere", which is still literally true. Schedule it?
  > **A:** Yes — file it under `DEFERRED.md`'s existing `## Trigger: align acquiring a
  > backend-selection API`.
  > **Q:** Does the `cython-translation` prerequisite gap earn its own `DEFERRED.md`
  > trigger section, or does it belong under an existing one?
  > **A:** Its own — a new `## Trigger: the next dp-compile revision`, mirroring the
  > `workflow-claude` section.
  > **Done:** all three settled and landed on `main` as `8747778`, pushed. **What this
  > branch may claim:** equivalence against phase 1 is enforced by explicit differential
  > tests in the labelled non-shared section at the foot of `test_viterbi.py` — which is
  > what ADR 0003 asks for unconditionally in the meantime, not a workaround. It may
  > **not** claim the parameterized suite, at any phase in this revision.
  - [x] Reconcile the subgoal's "enforced by the parameterized suite" with `core.md`'s
        finding that ADR 0003's parameterization cannot be in force until `align`;
        whichever is wrong is corrected on `main`, not worked around here
    > **Note:** neither document was wrong about the *mechanism* — the tension was already
    > recorded, in the phase-1 subgoal's `> **Done:**` block at `docs/plan/TODO.md:163-172`,
    > which gives the same two-requirements argument and names line 200 as the problem.
    > `core.md` was checked against ADR 0003's Decision section rather than trusted as a
    > paraphrase: "a test is written once, against the algorithm's public API, and the
    > backend is a fixture parameter" is verbatim. What was undone was the acting on it.
    > **Note:** the correction is **two** lines, not one. Line 209 (phase 4) repeats the
    > claim and is outside the phase-1 note's scope, which says "*For phase 2:*". It is
    > unkeepable for an extra reason worth stating once: `align` follows `hmm` in
    > implementation order, so **no phase in revision 02 can use the parameterized suite**.
    > That is structural, so phase 4 would have rediscovered this independently.
    > **Note:** the phase-1 note's verbatim quotation of line 200 now reads "which then
    > read", with a parenthetical recording what was acted on. Acting on a finding tends to
    > invalidate the finding's own wording — past tense is what makes such a record
    > permanent, and this is the third instance of that pattern this session.
  - [x] Triage the three `first .pyx` DEFERRED items: close the meson-python revert
        (spent by ADR 0018), schedule the tokalign comma-form indexing fix here, leave
        the `numba-cuda` pin to phase 4
    > **Note:** triaged exactly as predicted. The meson-python revert is closed **in place**
    > — this file annotates rather than deletes, a convention confirmed against eight prior
    > closures all reading "**Settled (date), and the entry is closed.**" The finding kept
    > is that its trigger *never fired*: ADR 0012 deferred the choice to the first `.pyx`
    > believing compilation injects the import finder, but the editable install does, so it
    > was settled early on `exp/meson-python-namespace` and superseded by ADR 0018 with the
    > extension blocks still dormant. The tokalign comma-form fix stays scheduled under
    > goal 3 here and the `numba-cuda` pin stays put for phase 4; neither needed an edit.
  - [x] Decide whether the `cython-translation` prerequisite gap earns a `DEFERRED.md`
        trigger of its own. There is no `dp-compile` trigger section yet, only a
        `workflow-claude` one
    > **Note:** it does — `DEFERRED.md` now has 14 trigger headings, the last being
    > `## Trigger: the next dp-compile revision`. Two measurements sharpened the entry
    > beyond what this plan's header claims. First, the defect is worse than a missing
    > carve-out: prerequisite 2 *asserts* that phase 1 derives from phase 0, so followed
    > literally on a recovered algorithm it hands off to `algorithm-prototype` to regenerate
    > the kernel from the document written out of it. Second, checked against all three
    > phase skills, the gap is **unique to `cython-translation`** and structurally so —
    > `cpu-parallelization` derives from phase 2 and `gpu-parallelization` from phase 3, and
    > those are forward edges in either graph shape. It is the only skill whose immediate
    > source is phase 1, the one node whose incoming edge differs between the forward chain
    > and the recovered graph. So the fix is narrow and will stay narrow.
- [x] Advance to phase 2 via `/dp-compile:next-phase viterbi`
  > **Done:** phase 2 `cython` — `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi_cython.pyx`,
  > derived-from `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi.py`
  > `sha256:48666ca8a3126ba141ec1a0ddba8ed1823e502f93d72483181906afd24135742`,
  > suite green at 282. Entered through `/next-phase`, which is what supplied the
  > recovered-graph rule the skill's own prerequisite 2 lacks.
  > **Commit:** committed here with `/smart-commit` rather than a plain commit, because
  > `core.md`'s "nothing is compiled today" became false the moment this `.pyx` built.
  > That pulls part of goal 6 forward deliberately: a doc sentence contradicted by the
  > code committed beside it is the failure `/agents-docs-update` exists to prevent, and
  > correcting it in the commit that falsifies it costs nothing.
  > **Note:** the skill's scope overlaps the next two goals, and the overlap was **not**
  > silently absorbed. Its "Build configuration" step needed no `meson.build` edit —
  > the extension block is already an `if fs.exists(viterbi_pyx)` guard that activates
  > by itself, and `tests/test_meson_sources.py` filters to `.py` plus
  > `PACKAGE_DATA_NAMES`, so a `.pyx` is deliberately not required in
  > `install_sources`. The `_backends.py` row is **not** added (goal 4 owns it) and the
  > property test is **not** written into the suite (goal 5 owns it). Suite is green at
  > 282 rather than higher precisely because no test was added here.
  - [x] Expect its step-3 oracle gate to pass: all three `.vpath.xls` files the
        manifest declares are exercised by `test_viterbi.py` through `load_vpath`.
        The gate matters most for exactly this algorithm — a recovered formalization
        and its extracted test cases both came from the kernel, so the oracles are
        the only evidence left that could disagree with it
    > **Ran:** it passed. Both gates did: the phase the target depends on is `python`,
    > whose suite was green at 282 immediately before routing, and no declared oracle
    > is unexercised. Target resolved as **write `cython`** — an advance, not a
    > regeneration, with no second minimal stale artifact to report.
  - [x] Let it route to `dp-compile:cython-translation`; the `.pyx` is written against
        the phase-0 specification, whose min-plus objective and smallest-index
        tie-break are contract rather than description
    > **Note:** **the frozen parameter arrays force `const` memoryviews, and this is the
    > one thing that would have failed at runtime rather than at compile time.** ADR 0017
    > gives `init_state_p`, `transition_p` and `output_p` `writeable = False`; a plain
    > `double[:, ::1]` on a read-only buffer compiles cleanly and then raises
    > "buffer source array is read-only" on *every* call. Measured, not assumed — all
    > three are float64, C-contiguous and read-only. `codes` is `int32`, not `int64`:
    > it is `dataseq`'s `CODE_DTYPE`, and the two state arrays are `int64` to match
    > `_viterbi.py`'s `STATE_DTYPE`.
    > **Note:** equivalence is bit-exact rather than approximate, by construction: the
    > same two float64 operations in the same order (multiply, `-log2`, add), and a
    > strict `cand < best` that keeps the earliest `i` exactly as `np.argmin` does.
    > Verified against phase 1 on **405 comparisons** — the three tracked Lush models,
    > 400 random models of varying `S` and vocabulary, plus the cases the fixtures
    > cannot exhibit: an exactly uniform model (ties at *every* position), an
    > all-`+inf` row, an empty record, `S = 1`, and `init_p` zero on four of five
    > states. Zero mismatches, before and after the bounds-check decorators.
    > **Note:** the arc cost is computed scalar-wise inside the `i` loop. That is the
    > fusion `_viterbi.py`'s comment anticipates ("Phase 2 fuses this into the inner
    > loop"), **not** the ADR 0015 hoist — the value still depends on both endpoints
    > and is never lifted out of either loop.
    > **Note:** `uv sync` alone does **not** pick up a new `.pyx`, and this will bite
    > again at the next kernel. The `fs.exists()` guard is evaluated at meson
    > *configure* time, so a build directory configured when the file was absent stays
    > unaware of it; `uv sync` reported "Checked 26 packages" and changed nothing, and
    > the import then failed `ModuleNotFoundError`. `uv sync --reinstall-package
    > pfsmgraph-hmm` forces the reconfigure. Note this is the *opposite* footgun from
    > the one `core.md` records for non-editable installs — there a source edit is
    > served stale; here a whole new build target is.
  - [x] Apply the deferred comma-form indexing fix to the tokalign template **before**
        it is used as the reference template. That file is deliberately untracked, so
        this goal closes with no diff to point at — record it as a note
    > **Note:** applied — **41 replacements** across `M`, `X`, `Y`, `T` in
    > `.scratch/align-poc/tokalign/src/tokalign/algorithms/needleman_wunsch/_cython.pyx`,
    > every 2-D access moved from `A[i][j]` to `A[i, j]`. Two `][` occurrences remain
    > and are correct: `aligned_a_np[:k][::-1]` and `aligned_b_np[:k][::-1]` are numpy
    > slices on the output arrays, not memoryviews. `git status` showed nothing
    > afterwards — the file is ignored by `.scratch/align-poc/.gitignore:170`, so the
    > subgoal's "no diff to point at" is literal. **It cannot be compile-checked here**:
    > `tokalign` is not a workspace member and nothing under `.scratch/` is built, so
    > this is verified by inspection. That limitation is inherent to the item.
    > **Note:** the fix did not end up mattering for *this* `.pyx`. The template is a
    > Needleman-Wunsch alignment over four 2-D matrices; the Viterbi kernel was written
    > from the phase-0 specification and `_viterbi.py`, not by copying it. Applying it
    > first was still right — PRD §11 designates that file the reference for every
    > future kernel and wavefront pass, so phases 3 and 4 and `align` are the consumers,
    > and a defect in a template is copied forward looking like house style.
  - [x] Confirm the emitted `.pyx` carries `# dp-compile: derived-from` naming
        `_viterbi.py` with its current hash. The kernel is headerless as a root; the
        `.pyx` is not a root and must record its edge
    > **Note:** confirmed by recomputation rather than by eye — the header names
    > `_viterbi.py` and records `sha256:48666ca8…`, and hashing that file with its own
    > provenance line stripped (there is none) yields the same digest. So the `.pyx` is
    > `fresh` and the graph now has two recorded edges: formalization ← python, and
    > python → cython, branching at the root exactly as `phase-detection.md` draws it.
- [ ] Make it build, install and import
  - [ ] `packages/pfsmgraph-hmm/meson.build`: un-dormant the extension block and name
        every new source in `install_sources` — meson does not glob, and
        `tests/test_meson_sources.py` is what catches the omission
  - [ ] Add the backend row to `_backends.py`; a backend implemented but not importable
        is a hard failure under ADR 0003, never a skip
  - [ ] `dp-compile`'s `PreToolUse` hook arms on this path from the first commit that
        stages it — `[commands] build` and `test` run before `git commit` and block on
        failure. Expect it inside `/smart-commit`; that is the design, not a leak
- [ ] Establish equivalence
  - [ ] Against phase 1 on constructed inputs, including the exact-tie case the learned
        fixtures cannot exhibit — there are 0 exact ties in 3804 oracle positions, so a
        differential suite alone would accept a last-wins port
  - [ ] Against the three oracles, one of which differs at position 0 by design: the
        δ-seeding defect is fixed rather than reproduced
- [ ] Update whatever the change makes stale
  - [ ] `core.md`: the backend matrix stops having one row, and the four-phase lifecycle
        gains its first compiled occupant
  - [ ] `claude.md` only if something names a Claude Code feature; the `dp-compile`
        section already describes the hook and the manifest
  - [ ] Neither is edited directly for `core.md` — `/agents-docs-build` regenerates the
        artifacts, and `/smart-commit` runs `/agents-docs-update` ahead of the commit
