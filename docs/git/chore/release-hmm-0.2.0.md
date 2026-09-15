# chore/release-hmm-0.2.0

**Created**: 2026-09-15
**Base**: main at 9850b43
**Status**: active

## Purpose

Release `pfsmgraph-hmm` 0.2.0: Baum-Welch training, batched Viterbi decoding, and runtime
backend selection over four lifecycle phases plus torch. It is the first version whose
public API reaches a compiled kernel, so it cannot ship 0.1.0's pure `py3-none-any` wheel
without every pip user seeing `cython ✗`. The branch settles the wheel shape, adapts the
release tooling to it, verifies what a consumer installs, and publishes.

## Scope

- Decide the wheel shape and where it is built (cibuildwheel in CI, locally, or pure again)
- Drop `-Dcompiled=false`, restore `Programming Language :: Cython`, teach `preflight` platform tags
- Review 0.2.0's immutable metadata: version, extras bounds, `dataseq` lower bound, README
- Verify each built wheel in a clean venv, including the compiled kernels' bit-exactness
- Publish and tag `pfsmgraph-hmm-v0.2.0` (user-run steps marked **User**)

## Context

- Master plan `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, the release subgoal and its two notes
- Precedent: `docs/plan/02-hmm-v0.1.0/chore-release-hmm-0.1.0/TODO.md` (PR #25)
- `docs/plan/DEFERRED.md`: "a backend-selection API" (the pure-wheel ending) and "CI existing"
  (trusted publishers, `PFSMGRAPH_REQUIRE_BACKENDS`)
- `docs/ops/release.md`, the root `justfile`, `packages/pfsmgraph-hmm/meson.build` / `meson.options`
- ADR 0018 (meson-python), ADR 0020 (no fused multiply-add), ADR 0021 (backend selection)

## Notes
