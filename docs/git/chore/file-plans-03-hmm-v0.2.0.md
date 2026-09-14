# chore/file-plans-03-hmm-v0.2.0

**Created**: 2026-09-14
**Base**: main at b487e09
**Status**: active

## Purpose

File the first merged branch plan of revision `03-hmm-v0.2.0` into its revision directory.
`fix-hmm-viterbi-log2` landed on `main` with PR #28 as a flat `docs/plan/` directory, and
`/file-plans` places it under `docs/plan/03-hmm-v0.2.0/` by reading its `> **Branch:**`
backlink in the master plan. This is the revision's first filing, so it also creates the
revision directory `/open-revision` deliberately left uncreated.

## Scope

- `mkdir -p docs/plan/03-hmm-v0.2.0/` and
  `git mv docs/plan/fix-hmm-viterbi-log2 docs/plan/03-hmm-v0.2.0/fix-hmm-viterbi-log2`,
  exactly as `file-plans.sh` proposed (1 to file, 0 skipped).
- No lookup needs updating: every command resolves a plan by directory name, and PR #28's
  body names the plan rather than giving its path.

## Context

- PR #28 (`fix/hmm-viterbi-log2`), merged 2026-09-14.
- PR #26, the precedent for a filing branch (`chore/close-revision-02-hmm-v0.1.0`).
- Left alone deliberately: `DEFERRED.md`'s "revision 03 or 04 opening" item asking whether
  `docs-hmm-migration-plan` should move out of `02-hmm-v0.1.0`. It was not chosen when the
  revision opened and stays open under its trigger.

## Notes
