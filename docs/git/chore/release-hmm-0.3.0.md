# chore/release-hmm-0.3.0

**Created**: 2026-09-18
**Base**: main at b05ec98
**Status**: active

## Purpose

Revision 04's last subgoal: release `pfsmgraph-hmm` 0.3.0, the version that adds
topology search by state merge and split, scored by minimum description length. It
follows 0.2.0's CI path — `just release-ci`, since `pfsmgraph-hmm` is in
`ci_built_packages` and `just release` refuses it — and it is the first release to take
that path from a version the tree bumped at the revision's opening (`0.3.0.dev0`) rather
than at the release commit.

What makes it unlike 0.2.0 is what it does *not* add. [ADR 0025](../../design/adr/0025-topology-search-not-exported.md)
keeps `__all__` at the ten names 0.2.0 shipped, so the headline capability has no
supported entry point, and §7 requires the release notes to say so plainly rather than
leave a reader to discover it from `__all__`. And `HMMLIB-ACCOUNT.md` §12 predicts no new
ADR 0002 lifecycle phase, since the original compiled none of its search methods — which
this branch verifies rather than assumes.

## Scope

- Verify what ships: no new backend row, the ADR 0003 header byte-identical to 0.2.0's,
  the four-file release invariant and `license-files` intact.
- Decide the two `_trials` helpers only the tests call (`_suggest_split`,
  `_suggest_merge`), handed on by PR #52.
- Release notes meeting ADR 0025 §7. No `CHANGELOG` exists and 0.2.0 wrote none, so where
  they live is itself a decision.
- The root `README.md`'s status line ("released at **0.2.0**", the search as a later
  revision) and `codex.md`'s review guide, which still describes `hmm` as "three modules
  and 167 tests" as of 2026-09-04 — both handed on by PR #52.
- A CI dry run of the wheels and sdist, verified against a clean install, before any tag.
- The release commit, the tag push by the user, digest confirmation, and the record
  commit.

## Context

- Master plan: `docs/plan/TODO.md`, revision `04-hmm-v0.3.0`, the last open subgoal.
- Previous release: `chore/release-hmm-0.2.0` (PR #40), whose plan is filed under
  `docs/plan/03-hmm-v0.2.0/` and is the template for goals 3 and 4 here.
- Audit that precedes this: `chore/hmm-0.3.0-audit` (PR #52), which handed on the README,
  release notes, `codex.md` and `_trials` items.
- Runbook: `docs/ops/release.md`. Versioning rules: `docs/agents/core.md`, "Versioning" —
  in particular the window between a release commit and the next `.devN`, missed at both
  0.1.0 and 0.2.0.

## Notes
