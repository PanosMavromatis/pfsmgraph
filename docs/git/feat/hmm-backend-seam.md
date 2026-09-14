# feat/hmm-backend-seam

**Created**: 2026-09-14
**Base**: main at 6d4fd8a
**Status**: active

## Purpose

Decide, in an ADR, how a caller of `pfsmgraph-hmm` chooses a backend at runtime and what happens when the requested one is unavailable; then build that seam. With it, the ADR 0003 suites can finally be parameterised over backends against the public API. Until now the two requirements have been jointly unsatisfiable, so a green `backends:` header has meant only that each kernel imports. Restore the `cpu-parallel` and `gpu` extras that 0.1.0 withheld, since a selectable backend gives them something to promise.

## Scope

- ADR 0021: the public selection shape, backend names, the default, and whether an unavailable backend raises or falls back
- The seam in the package: one registry read by both the runtime and the repo-root test header; `viterbi` first, with forward-backward and EM taking the same parameter shape at phase 1
- Parameterise the ADR 0003 suites over the seam; fold `test_viterbi.py`'s labelled non-shared section into the shared tests
- Restore `cpu-parallel = ["numba>=0.61"]` and `gpu` (with numba-cuda's `numpy<2.5` cap)
- Docs: `docs/api/hmm/`, `core.md`, the ADR index, and closing the `DEFERRED.md` entries

## Context

- Master plan: `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, the backend-selection subgoal
- `docs/plan/DEFERRED.md`, `## Trigger: align acquiring a backend-selection API` (three items)
- ADR 0003 (its Open section routes this question here), ADR 0011 (the house position on strictness), ADR 0016 (the four phases)
- The repo-root `_backends.py` / `conftest.py`: today's test-only registry
- It follows PR #32 (hmmlearn oracle); the `torch` backend subgoal comes next and depends on this seam

## Notes
