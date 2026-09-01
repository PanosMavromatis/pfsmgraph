# feat/backend-matrix-header

**Status**: merged — PR #6 — 2026-09-01
**Created**: 2026-09-01
**Subgoal**: Write the first test suite to the ADR 0003 standard, including the `pytest_report_header` hook that prints the backend matrix and names every excluded backend with its reason (revision `01-dataseq-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Decide what the header prints when the matrix is empty
  - [x] Re-read ADR 0003's "no parameter at all" rule against ADR 0002's definition of a backend, and state whether `dataseq`'s pure Python is one
  - [x] Put the options to the user and log the answer inline
  - [x] Record the decision where it will be found -- an ADR amendment, or a comment at the hook, whichever the answer warrants
  > **Done:** recorded as an in-place amendment to ADR 0003 rather than a comment at
  > the hook, because it resolves an under-specified corner of that ADR's own *Mechanism*
  > in that ADR's own direction -- it reverses nothing, so it warrants neither a new
  > number nor an index footnote, and the status stays Accepted. Two edits, +22/-5: the
  > empty-matrix rule added to *Mechanism*, and the stale "no tests exist yet" bullet
  > retired from *Open* into a new *Resolved* section following ADR 0010's convention,
  > which keeps a settled question visible rather than deleting it. Amending before the
  > hook exists is safe because an ADR specifies rather than promises -- the failure this
  > branch has to clean up in `pyproject.toml` is a *scheduling* promise ("lands with the
  > first test suite"), which reality could and did falsify.
  > **Found:** it is not. ADR 0002 scopes the three phases to "wherever dynamic
  > programming appears", and `dataseq` contains none -- it is a container and an
  > encoder. Its pure Python is therefore the implementation, not a phase-1 backend.
  > The consequence is stronger than "not yet": `dataseq` will *never* contribute a
  > backend however mature it gets, so the matrix stays empty until `align` or `hmm`
  > writes a recurrence. This is permanent behaviour being specified, not a placeholder.
  > **Q:** The matrix is empty and stays empty until the first DP kernel. What should
  > `pytest_report_header` print -- an explicit empty line, nothing at all, a
  > `python` backend registered today, or defer the hook to the first `.pyx`?
  > **A:** The explicit empty line:
  > `backends: none registered -- no DP kernel has reached ADR 0002 phase 1`.
  > It satisfies ADR 0003's governing principle -- absence is reported, never silent --
  > without asserting a lifecycle that does not exist, and it proves on every run that
  > the hook is wired, so the line is trusted the first time it carries real content.
  > Printing nothing was rejected because silence is indistinguishable from a hook that
  > is missing, misplaced, or raised and was swallowed, which is the same failure the
  > `-ra` half of the mechanism exists to prevent. Registering `python` was rejected as
  > actively false, not merely premature: it would claim `dataseq` sits at phase 1 and
  > make the first `.pyx` read as *its* phase 2. Phase 1, not phase 2, is the threshold
  > for a non-empty matrix -- a pure-Python DP kernel is already a backend parameter.

- [x] Write the shared availability fixture and site the `conftest.py`
  - [x] Choose the location: root `conftest.py` versus one per package, given that ADR 0003 requires availability be determined **once** so every suite reports identically
  - [x] Define how a backend registers itself, so that adding one later is adding a parameter value and not editing the hook
  - [x] Confirm the choice does not make `packages/` depend on repo-root files that no wheel ships
  > **Measured:** the siting was not a matter of taste. `pytest_report_header` is a
  > *startup* hook and nested conftests are loaded during collection, so a hook defined
  > in `packages/*/tests/conftest.py` is registered too late and discarded **silently**.
  > Probed directly (pytest 9.1.1, Python 3.14.7): two conftests each defining the hook,
  > only the root one printed. The failure mode is the nasty kind -- "hook in the wrong
  > directory" and "no backends to report" produce identical output -- which is a second,
  > independent argument for goal 1's always-print rule, since it makes the mistake
  > self-diagnosing.
  > **Q:** Given the hook must sit at the repo root, how should the code be organised --
  > one root `conftest.py`, a root `conftest.py` over a helper module, or a dev-only
  > pytest plugin?
  > **A:** Root `conftest.py` holding only the hook, over a `_backends.py` beside it
  > carrying the registry, the probe and the `PFSMGRAPH_REQUIRE_BACKENDS` logic. Goal 3
  > requires the hook itself be tested, and a plain importable module is testable where a
  > conftest is not; a plugin was rejected as a whole distribution's worth of structure
  > for one header line.
  > **Q:** How should a backend become visible to the matrix once kernels exist --
  > a probe table in the test infra, a declaration module shipped by each distribution,
  > or entry points?
  > **A:** A probe table in the test infra. Both alternatives put backend enumeration
  > into *shipped* source or metadata, which is most of what a runtime backend-selection
  > API needs -- and ADR 0003 leaves that question explicitly open, so shipping the
  > enumeration now would prejudge it. Adding a backend stays a one-line table edit.
  > **Found (subgoal 3, and it is worse than the subgoal asked):** nothing under
  > `packages/` depends on a repo-root file, because no shipped artifact contains a test
  > at all -- the wheel packages only `src/pfsmgraph`, verified by building it. But the
  > **sdist does ship `tests/`**, and ships them with neither half of the ADR 0003
  > mechanism: the sdist's `pyproject.toml` is the member's own 39-line file whose only
  > `[tool.*]` table is `hatch.build.targets.wheel`, so `addopts = "-ra"` -- which lives
  > in the workspace-root `pyproject.toml` -- does not travel. A packager running pytest
  > from the sdist gets bare `s` for skips and no header. ADR 0003's Open section guesses
  > the opposite ("if they do, distro packagers ... inherit this policy"); both halves of
  > that guess are now measured false. Filed as a goal-4 record fix.
  > **Done:** this goal produced the design and the two measurements behind it, but wrote
  > no code. `_backends.py` cannot be written in two halves -- its registry is inseparable
  > from the escalation rules goal 3 owns -- so splitting the file across two goals would
  > leave `main` reachable at a half-written module if the branch were interrupted. Goal 3
  > creates both files.

- [x] Implement `pytest_report_header` and the two availability rules
  - [x] The header line, in ADR 0003's exact format
  - [x] Implemented-but-not-importable is a hard failure, never a skip (the stale-`.so` case)
  - [x] `PFSMGRAPH_REQUIRE_BACKENDS` escalates an absent-hardware skip to a failure for the named backends
  - [x] Test the hook itself -- the reporting mechanism is the one thing no other test covers
  > **Done:** three files -- `_backends.py` (127 lines, the registry, probe and
  > escalation), `conftest.py` (41, the hook and nothing else), and
  > `tests/test_backends.py` (126, 13 tests). The suite goes 74 -> 87.
  > The three-way split ADR 0003 requires is expressed by the shape of a registry row,
  > not by probing the filesystem: a row that imports is `✓`, a row that fails to import
  > but names its `hardware` is a reported skip, a row that fails with `hardware=None` is
  > a hard failure, and a phase not yet reached is *no row at all*. Probing for build
  > products was rejected for a specific reason -- a Cython backend whose `.so` was never
  > built would then read as "not written yet" rather than "broken", silently downgrading
  > the one case ADR 0003 insists must be a hard failure.
  > `check_required` validates the names in `PFSMGRAPH_REQUIRE_BACKENDS` *before* testing
  > availability, the same shape as ADR 0011's encoder validating `on_unknown` before the
  > loop. Without it, CI setting `=cuda` against today's empty matrix would pass, because
  > nothing named `cuda` was missing.
  > **Ran:** `uv run pytest` -> 87 passed, header line present:
  > `backends: none registered — no DP kernel has reached ADR 0002 phase 1`.
  > `PFSMGRAPH_REQUIRE_BACKENDS=cuda uv run pytest` -> exit **4** (pytest's UsageError),
  > `ERROR: PFSMGRAPH_REQUIRE_BACKENDS names ['cuda'], which are not in the matrix
  > (registered: none).` -- one clean line, no traceback, and non-zero so CI fails.
  > Two of the 13 tests guard the wiring rather than the logic, because goal 2 measured
  > that misplacing the conftest fails silently: one asserts `conftest.py` and
  > `_backends.py` are at the repo root, the other copies both into a `pytester` temp
  > rootdir, runs pytest in a subprocess and asserts the line appears in the output.

- [x] Record the outcome
  - [x] Close the master-plan subgoal, or say why it stays `[~]`
  - [x] Fix the root `pyproject.toml` comment promising the hook "lands with the first test suite"
  - [x] Update `core.md` if any claim it makes about the suite has moved
  - [x] Correct ADR 0003's Open bullet on sdist/wheel: tests *do* ship in the sdist and the policy does *not* travel with them (measured in goal 2)
  > **Done:** four records, three of them falsified by *completion* rather than by a code
  > change -- the direction that goes unnoticed, since nothing fails. The root
  > `pyproject.toml` comment promising the hook "lands with the first test suite" is an
  > instance of the exact class `DEFERRED.md`'s release trigger already names, "a
  > future-tense promise stranded after it was kept"; it outlived the tests it predicted
  > by three commits. `core.md` was discharged early inside `/smart-commit`, whose
  > docs pass reached it from the staged diff.
  > The sdist finding is filed in **two** places, following the pattern ADR 0003's other
  > open question already sets: the record states the question and the measurement, and
  > `DEFERRED.md` carries the trigger that surfaces it. ADR 0003 says runtime backend
  > selection is settled "when `align` acquires a backend-selection API" and `DEFERRED.md`
  > has a trigger heading of that name; the sdist question now sits under *the first real
  > release* for the same reason. An ADR's *Open* section alone is not read on a schedule.
  > **Ran:** `uv run pytest` -> 87 passed after the `pyproject.toml` edit, confirming the
  > comment change did not disturb `[tool.pytest.ini_options]`.
  > **Note for `/smart-merge`:** its step 7 closes the master-plan subgoal itself. That is
  > already done -- the item is `[x]` with a second `> **Done:**` beneath the backlink --
  > so it should append only the PR number rather than write a third note.
