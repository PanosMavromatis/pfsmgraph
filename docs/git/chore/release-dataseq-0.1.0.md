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
- [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) — its `Open` section holds the sdist/wheel question, measured on the previous branch
- [ADR 0006](../../design/adr/0006-single-repository-as-a-uv-workspace.md) — the workspace footgun that makes a wrong version bound invisible locally
- `docs/agents/claude.md` — `/smart-commit` cannot produce this project's per-package tags under any configuration; the tag is manual

## Notes
