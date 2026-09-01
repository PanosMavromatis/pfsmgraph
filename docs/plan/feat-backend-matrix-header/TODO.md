# feat/backend-matrix-header

**Status**: active
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

- [ ] Write the shared availability fixture and site the `conftest.py`
  - [ ] Choose the location: root `conftest.py` versus one per package, given that ADR 0003 requires availability be determined **once** so every suite reports identically
  - [ ] Define how a backend registers itself, so that adding one later is adding a parameter value and not editing the hook
  - [ ] Confirm the choice does not make `packages/` depend on repo-root files that no wheel ships

- [ ] Implement `pytest_report_header` and the two availability rules
  - [ ] The header line, in ADR 0003's exact format
  - [ ] Implemented-but-not-importable is a hard failure, never a skip (the stale-`.so` case)
  - [ ] `PFSMGRAPH_REQUIRE_BACKENDS` escalates an absent-hardware skip to a failure for the named backends
  - [ ] Test the hook itself -- the reporting mechanism is the one thing no other test covers

- [ ] Record the outcome
  - [ ] Close the master-plan subgoal, or say why it stays `[~]`
  - [ ] Fix the root `pyproject.toml` comment promising the hook "lands with the first test suite"
  - [ ] Update `core.md` if any claim it makes about the suite has moved
