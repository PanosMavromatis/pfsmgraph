# feat/backend-matrix-header

**Status**: active
**Created**: 2026-09-01
**Subgoal**: Write the first test suite to the ADR 0003 standard, including the `pytest_report_header` hook that prints the backend matrix and names every excluded backend with its reason (revision `01-dataseq-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [ ] Decide what the header prints when the matrix is empty
  - [ ] Re-read ADR 0003's "no parameter at all" rule against ADR 0002's definition of a backend, and state whether `dataseq`'s pure Python is one
  - [ ] Put the options to the user and log the answer inline
  - [ ] Record the decision where it will be found -- an ADR amendment, or a comment at the hook, whichever the answer warrants

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
