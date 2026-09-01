# chore/release-dataseq-0.1.0

**Created**: 2026-09-01
**Base**: main at 3a66581
**Status**: active

## Purpose

Publish `pfsmgraph-dataseq` 0.1.0 to PyPI, replacing the content-free `0.0.0`
placeholder, and close the last open subgoal of revision `01-dataseq-v0.1.0`. This is
the first artifact of this project that anyone outside the repository can install, and
the first one whose mistakes cannot be taken back: PyPI versions are immutable and
yanking does not free the number.

## Scope

- Settle ADR 0003's sdist/wheel question — three candidate remedies are recorded, none picked
- Semantic sweep of the prose claims about repository state, ahead of first outside readers
- Honest lower bounds on the intra-family dependencies naming `pfsmgraph-dataseq`
- Drop `.dev0` from `pfsmgraph-dataseq` only, build, publish, and tag `pfsmgraph-dataseq-v0.1.0` by hand

## Context

- `docs/plan/TODO.md:86` — the master-plan subgoal this branch executes
- `docs/plan/DEFERRED.md` — `## Trigger: the first real release` carries five obligations, all of which land here rather than after
- [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) — the sdist/wheel question, measured on the previous branch and settled here; now in its `## Resolved` section
- [ADR 0006](../../design/adr/0006-single-repository-as-a-uv-workspace.md) — the workspace footgun that makes a wrong version bound invisible locally
- `docs/agents/claude.md` — `/smart-commit` cannot produce this project's per-package tags under any configuration; the tag is manual

## Notes

- **2026-09-01 — ADR 0003's sdist/wheel question is settled: the mechanism is
  repo-local.** Tests keep shipping in the sdist; neither `addopts = "-ra"` nor
  `pytest_report_header` travels with them, and the record now says so along with what it
  costs. The two self-sufficiency remedies were rejected in the ADR rather than merely
  passed over. The obligation to revisit is re-filed in `DEFERRED.md` under the `align`
  migration, since `align` is the first member whose backend matrix will have a row in it.
  No packaging change was made, which is stated in the record so it does not read as an
  oversight.
- **2026-09-01 — found while re-measuring: the sdist ships `.gitignore` and no README or
  LICENSE file**, though `license = "MIT"` is declared as metadata. The PyPI page would be
  blank on first publish, and `0.1.0` cannot be re-cut to fix it. Added as a subgoal to
  the branch plan's release goal.
