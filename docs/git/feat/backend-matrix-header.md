# feat/backend-matrix-header

**Created**: 2026-09-01
**Base**: main at fc59815
**Status**: active

## Purpose

Land the second half of the [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md)
reporting mechanism: the `pytest_report_header` hook that prints the backend matrix once
at session start, together with the shared availability fixture it reads and the
`conftest.py` that has never existed in this repository. The first half -- `addopts = "-ra"`
in the root `pyproject.toml`, so skip *reasons* reach the run summary -- landed with the
`dataseq` merge (PR #2) and carries a comment promising the hook "lands with the first test
suite". It did not, and this branch is that promise being kept.

The master-plan subgoal was deliberately left `[~]` rather than `[x]` for exactly this
reason, so revision `01-dataseq-v0.1.0` cannot close while the hook is owed. Closing it is
this branch's deliverable.

## Scope

- Decide what the header prints when the matrix is legitimately **empty** -- the open
  design question, see Context.
- Write the shared backend-availability fixture, determined **once** as ADR 0003 requires,
  and site the `conftest.py` that hosts it (root vs per-package).
- Implement `pytest_report_header`, plus the two rules that are cheap now and expensive to
  retrofit: `PFSMGRAPH_REQUIRE_BACKENDS` escalating an absent-hardware skip to a failure,
  and implemented-but-not-importable being a hard failure rather than a skip.
- Close the records -- the `[~]` subgoal, the now-false `pyproject.toml` comment, and
  `core.md` if any claim it makes about the suite moves.

## Context

- [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) is
  authoritative. It specifies the header's format verbatim
  (`backends: python ✓ · cython ✓ · cuda ✗ (no CUDA device detected)`), both halves of the
  mechanism, and the three-way distinction between a skip, a hard failure, and absence from
  the matrix.
- [ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md) defines what a
  "backend" is: a phase of one DP kernel's lifecycle. This is what makes the empty-matrix
  question real -- see below.
- Root `pyproject.toml` lines 31-38 already carry `[tool.pytest.ini_options]` with `-ra`
  and the comment this branch falsifies.
- No `conftest.py` exists anywhere outside `.scratch/`; verified before branching.

**The open question.** ADR 0003 rules that a lifecycle phase not yet reached "contributes
no parameter at all -- absence from the matrix is the honest representation of *not written
yet*". There are currently zero DP kernels: `dataseq` is pure Python, but it is not a
*backend* in the ADR 0002 sense, because it has no Cython or CUDA sibling for it to be
equivalent to. The matrix is therefore genuinely empty, and a header printing
`backends: python ✓` would assert a lifecycle that does not exist. What an empty matrix
should print is goal 1, and the answer needs the user.

The ADR's own hazard rule bears on the hook as much as on skips: "yellow output should
always mean something, or it stops being read". A line printed on every run that carries no
information decays the same way -- by the time the first `.pyx` makes it informative, nobody
is reading it.

## Notes

- 2026-09-01: branch created from `main` at `fc59815`, immediately after PR #5
  (reserved-block renumbering) merged. Master plan backlinked under the `[~]` subgoal,
  placed *after* its existing `> **Done:**` block so the item reads chronologically: PR #2
  did the 74 tests, this branch resumes for the hook.
- `docs/plan/refactor-reserved-block-renumber/` is still unfiled at the top level of
  `docs/plan/`. `/file-plans` sweeps it into `01-dataseq-v0.1.0/` and makes its own
  `chore/file-plans` branch; it is independent of this work and can run at any time.
