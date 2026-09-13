# chore/release-hmm-0.1.0

**Created**: 2026-09-13
**Base**: main at ef03f57
**Status**: active

## Purpose

Release `pfsmgraph-hmm` 0.1.0, the last subgoal of revision 02-hmm-v0.1.0 and the first
wheel this project publishes from meson-python. Reconciling `docs/api/hmm/` (PR #24) found
that the package declares `pfsmgraph-align>=0.1`, which no module imports and nothing on
PyPI satisfies, so the release first removes it. The removal is recorded as ADR 0019
amending ADR 0009, whose graph states the edge, and the question returns under a
`DEFERRED.md` trigger at the version that seeds HMM topology from sequence alignment
(expected 0.4.0).

## Scope

- Remove the `align` dependency; ADR 0019, ADR 0009's status pointer, the ADR index, a
  `DEFERRED.md` entry with its trigger, relock, and every document stating the edge.
- Commit the `justfile` default (`pfsmgraph-hmm`) and sweep docs naming the old one.
- Settle what 0.1.0's immutable metadata says: the accelerator extras, the `description`,
  honest lower bounds.
- Ship the four files (README, LICENSE copy, `Typing :: Typed`, `py.typed` in
  `install_sources`), bump `.dev0`, build, verify the wheel in a clean venv, publish, tag.

## Context

- Master plan: `docs/plan/TODO.md`, revision 02-hmm-v0.1.0, "Release `pfsmgraph-hmm`
  0.1.0 via `just release 0.1.0 pfsmgraph-hmm`…", including its blocker note.
- PR #24 (docs/hmm-api), which found the dependency; PR #8, the `dataseq` release.
- ADR 0009 (the graph), ADR 0018 (meson-python), `docs/ops/release.md`, the root `justfile`.
- PRD §1 "Alignment as a training accelerant", the reason the edge exists as intent.

## Notes

- The `justfile` default was changed by hand to the package under development, so an
  omitted package argument cannot touch an already-published one. Its comments still use
  `dataseq` as the illustrative default, deliberately.
