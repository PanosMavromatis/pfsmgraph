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
- [x] Make it build, install and import
  > **Commit:** deferred at the checkpoint — the next `/hitl-step` was invoked before it
  > was answered — then committed on its own after all, once goal 5 was closed. So this
  > goal has its own commit despite the pause, and the two goals are not entangled in
  > the history.
  > **Done:** the extension builds, installs and imports, and the ADR 0003 header now
  > reads `backends: python ✓ · cython ✓` — the matrix's first compiled row. Suite 282 →
  > 283. Only one of the three subgoals needed an edit; the other two were verifications,
  > and both found the answer already correct for reasons worth recording.
  - [x] `packages/pfsmgraph-hmm/meson.build`: un-dormant the extension block and name
        every new source in `install_sources` — meson does not glob, and
        `tests/test_meson_sources.py` is what catches the omission
    > **Note:** **no edit was needed, and that is a finding rather than a shortcut.** The
    > extension block was already written as `if fs.exists(viterbi_pyx)`, so it
    > un-dormanted itself the moment the `.pyx` landed — written on 2026-09-04 for a file
    > that did not yet exist, and correct. Nor does the `.pyx` belong in
    > `install_sources`: it is built by `py.extension_module`, and
    > `tests/test_meson_sources.py` filters the on-disk set to `.py` plus
    > `PACKAGE_DATA_NAMES`, so a `.pyx` is deliberately outside what that test requires.
    > It stayed green throughout, at 16 parameterized cases.
    > **Note:** the self-activating guard did not remove the manual step, it *moved* it.
    > `fs.exists()` is evaluated at meson **configure** time, so a build directory
    > configured before the file existed never sees it: `uv sync` reported "Checked 26
    > packages", changed nothing, and the import failed `ModuleNotFoundError`.
    > `uv sync --reinstall-package pfsmgraph-hmm` forces the reconfigure. A fresh clone is
    > unaffected — no build directory to be stale — so this bites exactly the people who
    > had the repository before the kernel landed. Revision 03's `_baum_welch_cython.pyx`
    > will hit it again; phases 3 and 4 add `.py` files and will not.
  - [x] Add the backend row to `_backends.py`; a backend implemented but not importable
        is a hard failure under ADR 0003, never a skip
    > **Note:** added as `Backend("cython", "pfsmgraph.hmm._viterbi_cython")` with
    > `hardware=None`, like `python` — but the clause means something different on this
    > row, and this is the first row where it does any work. A compiled backend can be
    > *present in the source tree and absent from the environment* in a way a pure-Python
    > one cannot: the `.pyx` is committed, so the phase is unambiguously implemented,
    > while the extension exists only if it was built. `hardware` is for absences that are
    > legitimate in an environment — no CUDA device — and a missing build is not one of
    > those. So an unbuilt extension errors the session at startup instead of reporting a
    > green run over one backend.
    > **Note:** five assertions in `tests/test_backends.py` pinned the matrix at exactly
    > one row and failed as designed — the previous version of the first one said in its
    > own comment that "adding the *second* row should break this one the same way", and
    > it did. Updated rather than worked around, which is a different act from editing a
    > test to make an implementation pass: these assert what the matrix *is*, and the
    > matrix changed deliberately. Note *how* the resolve test failed — on the assertion,
    > not with a `BackendError` — which is itself the evidence that the `cython` probe
    > imported successfully. One test added (19 in that file, suite 283): an unbuilt
    > compiled backend escalates rather than skipping, exercised against a synthetic row
    > so the real one stays untouched.
  - [x] `dp-compile`'s `PreToolUse` hook arms on this path from the first commit that
        stages it — `[commands] build` and `test` run before `git commit` and block on
        failure. Expect it inside `/smart-commit`; that is the design, not a leak
    > **Ran:** verified empirically in both directions rather than assumed, by feeding the
    > hook script a `{"tool_input":{"command":"git commit -m x"}}` payload. **Control**,
    > nothing staged: silent, exit 0, no build and no test — the script exits 0 with no
    > output at all when no staged path matches, which is why the real commit at `c9df9c6`
    > looked like nothing had happened. **Armed**, with the kernel staged: "dp-compile: 1
    > kernel file(s) staged; running the suite", then `uv sync` and `uv run pytest` (283
    > passed), then "dp-compile: suite passed", exit 0. The throwaway staged change was
    > reverted exactly.
    > **Note:** the goal-3 report said the hook's firing could not be confirmed from the
    > commit output. It now can, and the explanation is display rather than behaviour:
    > hook stdout reaches the transcript, not the `Bash` tool result. Worth knowing before
    > anyone concludes from silence that the gate is absent.
    > **Note:** the scoping is what makes a blocking hook tolerable, and it is narrower
    > than "touches the package". `kernel_paths()` expands `[phases]` against the
    > `[algorithms]` keys rather than wildcarding, precisely because `_{algorithm}.py`
    > would become a pattern matching every private module in a flat layout — `_numeric.py`
    > and `_params.py` included. `formalization` is excluded outright: a Markdown
    > specification compiles to nothing, so staging it cannot break a backend.
- [x] Establish equivalence
  > **Done:** six test functions added to `test_viterbi.py`'s labelled non-shared
  > section — eight tests, since the oracle one is parameterized over all three models.
  > File 57 → 65, suite 283 → 291. These are the only assertions in this repository
  > entitled to the word *equivalence*: they call both kernels and compare. The shared
  > cases above the line still run once, and will until `align` brings the selection
  > seam, so a green run must not be read as "every test ran against both backends".
  > **Note:** the ad-hoc check during goal 3 (405 comparisons) proved the translation but
  > left **no test behind**, which is the failure `test-patterns.md` names in one line —
  > "a check outside the suite is a check nobody runs". This goal is that check moved
  > inside. Nothing new was discovered by moving it; that is the point, and the reason to
  > do it before the branch merges rather than after.
  - [x] Against phase 1 on constructed inputs, including the exact-tie case the learned
        fixtures cannot exhibit — there are 0 exact ties in 3804 oracle positions, so a
        differential suite alone would accept a last-wins port
    > **Note:** four tests cover this: a seeded 200-model property test, the `N = 0` and
    > `S = 1` shape boundaries, the exactly-uniform tie model, and one asserting the
    > frozen `HMMParams` arrays reach the typed memoryviews directly. Equality is
    > **exact**, never `approx` — both kernels perform the same two float64 operations in
    > the same order, so an `abs=` tolerance would hide precisely the defect class the
    > test exists to find: a reassociated expression that is close everywhere and wrong
    > at a tie.
    > **Note:** **randomly generated models tie far more than the learned ones do, and
    > that was not expected.** Measured over the property test's own 200 models: **126
    > exact ties in 15,735 `(state, position)` reductions**, with 36 of 200 models
    > exhibiting at least one — against **0 in 3804** on the three tracked fixtures. The
    > mechanism was not chased, so this is a measurement rather than an explanation. The
    > practical consequence is what matters: the fixtures' tie-freedom is a property of
    > *learned* parameters, and any generator-based suite is exercising the tie-break
    > constantly whether or not it means to.
    > **Ran:** mutation-tested, because this repository already learned that a passing
    > differential suite can leave a tie-break unpinned. Flipping the recurrence's
    > `cand < best` to `<=` in the `.pyx` failed **two** tests (the property test and the
    > tie test); flipping the backtrace's `delta[n, j] < best` failed the tie test alone.
    > Both mutations reverted, suite back to 291. So neither tie-break is vacuously
    > pinned, and the uniform-model test is the one that catches both — which is exactly
    > the test the fixtures could never have motivated.
  - [x] Against the three oracles, one of which differs at position 0 by design: the
        δ-seeding defect is fixed rather than reproduced
    > **Note:** two tests. The first runs the compiled kernel against each `.vpath.xls`
    > through the existing module-scoped `oracle` fixture, and asserts agreement with the
    > saved decode after position 0 *and* exact agreement with phase 1's states. The
    > second pins the divergence at exactly `[0]` on `m008_0001_008` — `flatnonzero`
    > compared to a literal list, not a count or a fraction. `test-patterns.md`'s rule is
    > the reason: an oracle whose defect the port deliberately fixed is *wrong* exactly
    > where the fix applies, so widening the assertion to "mostly agrees" would let a
    > genuine phase-2 defect hide inside an allowance made for a known one.
    > **Note:** this is the subgoal that breaks the circle, and it matters more here than
    > it would for a forward-built kernel. The formalization was recovered from
    > `_viterbi.py` and its test cases were extracted from the same kernel, so every other
    > comparison in this file agrees with phase 1 by construction. The three `.vpath.xls`
    > files came out of a Lush runtime that no longer exists here — running the compiled
    > backend against them is what stops phase 2 inheriting that circularity unexamined.
- [x] Update whatever the change makes stale
  > **Done:** twelve stale claims corrected across five files — `core.md` (5), `codex.md`
  > (4), `README.md` (2), `claude.md` (1) — plus `DEFERRED.md`'s first-`.pyx` trigger
  > discharged, and `AGENTS.md` / `AGENTS.override.md` regenerated. Suite 291, header
  > `backends: python ✓ · cython ✓`.
  > **Note:** **a diff-driven sweep would have missed most of this, which is why the goal
  > exists as its own goal.** `/agents-docs-update` evaluates docs against the *staged
  > diff*, and goals 4 and 5 were already committed by the time this ran — so the changes
  > that falsified these sentences were no longer visible to it. Two of the twelve were
  > caught earlier, inside `/smart-commit`, precisely because they were staged at the time.
  > The rest needed a deliberate grep. Worth remembering when judging whether a doc sweep
  > can be delegated to the commit path: it can, but only for what is in that commit.
  > **Note:** the last stale claim was found by grepping *after* believing the sweep was
  > finished — `codex.md:258` still said "the matrix holds one row as of 2026-09-04". Nine
  > files had been read carefully by then. The grep cost nothing and the belief was wrong.
  - [x] `core.md`: the backend matrix stops having one row, and the four-phase lifecycle
        gains its first compiled occupant
    > **Note:** five edits. Three are counts that move together and are easy to fix by
    > halves — the suite total, the per-directory split, and the root breakdown's
    > backend-matrix figure — now 291 = 74 + 175 + 42, with 19 of the root 42 covering the
    > matrix. The fourth rewrites the matrix paragraph for two rows. The fifth is the ADR
    > 0003 invariant, and it is the one that needed thought rather than substitution: the
    > old text said `backends: python ✓` means the kernel imports, "not that any suite ran
    > twice", which was exactly right for one row and becomes *misleading* at two, since a
    > reader could take two ticks as evidence of parameterization. It now says both kernels
    > import, the shared cases still run once, and the labelled section holds the only
    > assertions entitled to the word equivalence.
    > **Note:** `hmm` is now described as "three Python modules, one Cython kernel" rather
    > than "three modules". The distinction is load-bearing for anyone counting: the `.pyx`
    > is a module in every sense that matters to the backend matrix, and in none of the
    > senses that matter to `install_sources`.
  - [x] `claude.md` only if something names a Claude Code feature; the `dp-compile`
        section already describes the hook and the manifest
    > **Note:** the qualifier earned its keep — there *was* something, and it was wrong
    > rather than merely absent. The hook paragraph said the gate "arms on exactly four
    > paths — `_viterbi.py` and its three unwritten siblings"; one of those siblings is
    > written now, so the sentence has to name it. Corrected to `_viterbi.py`,
    > `_viterbi_cython.pyx`, and the two remaining Numba paths.
    > **Note:** one genuinely new Claude Code fact added, from goal 4's measurement: the
    > gate prints **nothing at all** when no kernel is staged, and its output goes to the
    > transcript rather than into the `Bash` tool result. So a session can watch a gated
    > commit succeed and be unable to tell from its own output whether the gate ran — which
    > is exactly what happened at `c9df9c6`, and is recorded so the next reader does not
    > conclude from silence that the hook is missing.
  - [x] Neither is edited directly for `core.md` — `/agents-docs-build` regenerates the
        artifacts, and `/smart-commit` runs `/agents-docs-update` ahead of the commit
    > **Note:** honoured — every edit went to a `docs/agents/` source and the script
    > produced the artifacts, twice (once mid-sweep, once after the late `codex.md` fix).
    > `AGENTS.md` 486 → 494 lines and `AGENTS.override.md` 529 → 547. The
    > `protect-agent-docs.py` hook would have blocked a direct edit to either, but it was
    > never reached, which is the correct relationship to a guard rail.
    > **Note:** `DEFERRED.md` was swept in the same pass though this subgoal does not name
    > it. Its **`## Trigger: the first `.pyx`` has now fired**: the meson-python revert was
    > closed earlier this branch, and the comma-form indexing fix is closed here. The
    > `numba-cuda` pin is left open deliberately and the mismatch is written down — its own
    > text defers it to "once the wavefront backend lands", ADR 0002 phase 4, so it is
    > filed under a trigger that is not its condition. Naming that is better than moving
    > it, since a fired trigger showing an open entry should be explicable rather than
    > surprising.
    > **Note:** closing the comma-form entry surfaced something the entry could not have
    > anticipated: the template was fixed in time and then **not used**. The phase-2 kernel
    > was written from the phase-0 formalization, not by copying a Needleman-Wunsch kernel,
    > so the first consumer of the corrected template is still ahead of us. Fixing it early
    > was still right — a defect in a template propagates looking like house style — but
    > the entry's premise, "apply it while moving the file", described a move that did not
    > happen.
