# 0003. One parameterized test suite per algorithm, run against every backend

- **Status:** Accepted
- **Date:** 2025 (proof-of-concept); formalized 2026-08-29
- **Source:** PRD §1.2 — inherited from the proof-of-concept alignment library

## Context

[ADR 0002](0002-three-phase-algorithm-lifecycle.md) commits the family to three
implementations of every dynamic-programming kernel — pure Python, Cython, and a Numba
CUDA anti-diagonal wavefront — all of which must compute the same function. That is an
enormous standing invariant, and it is the kind that decays silently: a Cython kernel
that disagrees with its Python reference on one edge case produces plausible-looking
output indefinitely.

The obvious way to test three implementations is to write three test files. That fails
in a specific and predictable way: the suites drift. A test added when a bug is found in
the Python path does not get added to the Cython path, so the very case that revealed
the bug is the case the fast path is never checked against. Coverage ends up inversely
proportional to how heavily a backend is used.

Backend equivalence is the property that makes the whole lifecycle honest. It needs to
be enforced by construction, not by diligence.

## Decision

**Each algorithm has exactly one test suite, parameterized over backends, and it is run
automatically against every backend available in the environment.**

A test is written once, against the algorithm's public API, and the backend is a
fixture parameter. Adding a backend adds a parameter value, not a file. Adding a test
case tests every backend, unavoidably and without the author choosing to.

The pure-Python implementation is the oracle: where an assertion needs a known-correct
value, that value comes from the reference implementation's semantics
([ADR 0002](0002-three-phase-algorithm-lifecycle.md)).

## Policy: unavailable backends

This policy is **developer-facing**: it governs the test suite only, not the runtime
behavior of the library. What the public API does when a caller requests a backend that
is not present is a separate decision, not settled here (see `Open`).

A backend can be missing for two reasons, and they are not treated alike.

- **Absent hardware** — no CUDA device — is a fact about the environment, not a defect.
  The backend is skipped, and the skip is **reported, never silent**: the session header
  names every backend detected and every one excluded, with the reason, and skip reasons
  appear in the run summary. A run that did not exercise CUDA says so, in its own
  output, before it says `passed`.
- **A backend that is implemented but not importable** — most often a missing or stale
  Cython build — is a **hard failure**, never a skip. It means the working copy is
  broken rather than the machine being modest, and the stale-`.so` problem recorded in
  [ADR 0008](0008-per-package-build-backends.md) is precisely what a skip here would
  conceal.

A third case is not a skip at all: an algorithm that has not yet reached a given phase of
[ADR 0002](0002-three-phase-algorithm-lifecycle.md) contributes no parameter for it.
Absence from the matrix is the honest representation of "not written yet"; skipping is
reserved for "written, but not runnable here." Yellow output should always mean
something, or it stops being read.

Setting **`PFSMGRAPH_REQUIRE_BACKENDS`** (comma-separated backend names) escalates the
first case to a failure for the named backends, so CI can assert that a run it believes
to be GPU-capable actually exercised the GPU. Without it, a CI runner that loses its
device degrades to a green run whose header nobody reads. It is unset in normal
development.

### Mechanism

Backend availability is determined once, in a shared fixture, so every algorithm's suite
reports it identically:

- **`pytest_report_header`** prints the matrix once at session start —
  `backends: python ✓ · cython ✓ · cuda ✗ (no CUDA device detected)`. One line, before
  anything runs. Per-test skip messages would give N identical lines and be ignored.
- **An empty matrix still prints.** Until a DP kernel reaches phase 1 of
  [ADR 0002](0002-three-phase-algorithm-lifecycle.md) there is no backend to name, and
  the header says so in as many words —
  `backends: none registered — no DP kernel has reached ADR 0002 phase 1`. Printing
  nothing instead would be this policy's own failure committed one level up: a missing
  line is indistinguishable from a hook that is absent, misplaced, or that raised and
  was swallowed, so the mechanism could not be trusted the first time it carried real
  content. Note that `dataseq` contributes no backend at any maturity — ADR 0002 scopes
  the phases to wherever dynamic programming appears, and `dataseq` has none — so an
  empty matrix is the correct steady state until `align` or `hmm` lands a recurrence,
  not a placeholder awaiting the next commit. *(Added 2026-09-01.)*
- **`addopts = "-ra"`** in the root configuration, so skip reasons reach the summary.
  Pytest's default prints a bare `s` and swallows the reason, which is exactly the
  silence this policy exists to prevent.

## Consequences

### Positive

- **Backend equivalence is enforced rather than assumed.** This is the phrase the PRD
  uses, and it is precise: the suite cannot pass while backends disagree on a covered
  case.
- **Test coverage cannot be skewed toward the backend the author was thinking about.**
  Every case applies everywhere by construction.
- **Adding a backend is cheap and safe.** The wavefront implementation of a kernel
  arrives already covered by the accumulated case history of the Python and Cython
  phases — which is exactly when that history is most valuable.
- **It makes [ADR 0002](0002-three-phase-algorithm-lifecycle.md) survivable.** Without
  this control, three implementations per algorithm would be a maintenance liability
  rather than a strategy.

### Negative / costs

- **Tests must be written against the public API only.** A test that reaches into one
  backend's internals cannot be parameterized, so genuinely backend-specific concerns
  (memory layout, kernel launch configuration) need a separate, explicitly non-shared
  home.
- **Assertions must accommodate legitimate backend differences** — floating-point
  accumulation order differs between a sequential scan and a wavefront, so exact
  equality is not always the right assertion for probability-domain results, and
  tolerances have to be chosen deliberately rather than by default.
- **Ties and other under-specified outcomes must be pinned down.** Where a DP traceback
  has multiple optimal paths, the algorithm's tie-breaking rule becomes part of the
  contract, because otherwise two correct backends legitimately disagree.
- **The suite gets slower as backends multiply**, since every case runs N times.

## Alternatives considered

- **One test file per backend.** Rejected: this is the drift failure described above,
  and it is the default outcome of not deciding.
- **Test the Python reference thoroughly, smoke-test the others.** Rejected: it inverts
  the risk. The compiled and GPU paths are where the subtle bugs live, precisely because
  they are the ones doing manual index arithmetic with bounds checking disabled.
- **A dedicated cross-backend equivalence test, separate from the functional suite.**
  Rejected as insufficient on its own: it checks equivalence only on the cases someone
  remembered to put in it, which reintroduces the drift problem one level up.

## Open

- **Runtime backend selection is undecided and deliberately out of scope.** What the
  public API does when a caller explicitly requests an unavailable backend — raise, or
  fall back to the next-fastest — is a user-facing question with different obligations
  from a test suite's: the suite reports honestly on an environment it does not control,
  while the API honors or refuses a request. Note that
  [ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md) has already taken
  a house position on the general shape of this question, choosing strictness on the
  grounds that silently absorbing a problem produces work that merely looks fine —
  and "silently ran on the CPU path" is that same failure in another guise. Settle this
  when `align` acquires a backend-selection API; it warrants its own record.

## Resolved

- **No tests existed when this record was written** — `uv run pytest` collected nothing,
  which made the ADR a standard for work not yet done rather than a description of it.
  It bound `dataseq` first
  ([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)) and binds every
  DP kernel thereafter. **Settled 2026-08-31**: the `dataseq` merge landed 74 tests, the
  first in the repository, in `packages/pfsmgraph-dataseq/tests/`. The two halves of
  *Mechanism* followed separately — `addopts = "-ra"` with those tests, and
  `pytest_report_header` on 2026-09-01.

- **The mechanism is repo-local, and no sdist inherits it.** Measured 2026-09-01 against
  `pfsmgraph-dataseq` and reproduced at the 0.1.0 release: the wheel packages only
  `src/pfsmgraph`, so it contains no test at all and `pytest --pyargs` has nothing to
  collect; the sdist *does* carry `tests/`, but the `pyproject.toml` beside them is the
  member's own, whose sole `[tool.*]` table is `hatch.build.targets.wheel`. Neither half
  of *Mechanism* is present — `addopts = "-ra"` lives in the workspace-root
  `pyproject.toml` and `pytest_report_header` in the root `conftest.py`, and no sdist
  contains either. **Settled 2026-09-01: tests keep shipping, and the policy is
  repo-local by decision rather than by accident.**

  Both remedies that would make an sdist self-sufficient were rejected. Excluding
  `tests/` from the sdist buys the guarantee by removing the run, and it costs most
  exactly where the run is worth most: distro packagers build from the sdist and execute
  the suite to validate the build, which for the compiled members is the single most
  valuable place a test can run. Duplicating the mechanism into all five members makes it
  five things that can disagree — the drift failure this record exists to prevent,
  committed one level up, on the policy instead of on the tests.

  What that costs is stated here rather than left to be discovered: someone running
  pytest from an sdist gets a bare `s` for every skip and no header. Today it costs
  nothing, because `dataseq` registers no backend at any maturity and its matrix is
  empty. It starts costing something at the first member that registers one, where a
  green run could conceal a backend that is implemented but not importable — the case
  *Policy* makes a hard failure. That member is `align`, and the obligation to revisit is
  filed in `docs/plan/DEFERRED.md` under the `align` migration rather than left to
  memory.
