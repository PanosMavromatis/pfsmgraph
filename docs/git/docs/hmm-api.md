# docs/hmm-api

**Created**: 2026-09-13
**Base**: main at 4a50003
**Status**: active

## Purpose

Write the `docs/api/` documents for `pfsmgraph-hmm` 0.1.0, the revision-02 subgoal that
sits between the last Viterbi kernel (PR #23) and the release. Under
[ADR 0013](../../design/adr/0013-api-documentation-layout-and-tooling.md) the pages are
hand-written Markdown under a per-distribution subdirectory, normative for **contracts**
while docstrings stay normative for signatures, and every code block is executed with its
output pasted from the run. `pfsmgraph.hmm` exports four names — `HMMParams`, `viterbi`,
`ViterbiPath`, `ImpossibleSequenceError` — and `docs/api/README.md` still lists the member
as "no code yet".

## Scope

- Settle the page layout under `docs/api/hmm/` and which contracts each page carries:
  arc emission (`output_p[i, j, k]`), min-sum over bits, the `N + 1` state geometry, the
  exactly-zero reserved fibres, impossibility as `ImpossibleSequenceError`, the
  smallest-index tie-break, and that backends are not selectable in 0.1.0.
- Write the pages with executed code blocks and pasted output, green under
  `tests/test_api_docs.py`.
- Update `docs/api/README.md`'s distribution table; decide how the `gpu` and
  `cpu-parallel` extras and the `numpy<2.5` cap appear from a consumer's side.
- Check every page against the docstrings — a disagreement is a bug in one of them.

## Context

- Master plan: `docs/plan/TODO.md`, revision `02-hmm-v0.1.0`, "Write the `/docs/api/`
  documents that pertain to this release."
- Model to follow: `docs/api/dataseq/` (`README.md`, `container.md`, `encoder.md`) and
  its verifier `tests/test_api_docs.py`.
- Contracts to draw on: ADR 0015 (arc emission), ADR 0017 (frozen parameter value),
  ADR 0011 (reserved block), `docs/design/algorithms/viterbi/FORMALIZATION.md`.
- The next subgoal — releasing 0.1.0 — ships a member `README.md` as the PyPI long
  description, which the same verifier reads; that file belongs to the release, not here.

## Notes

