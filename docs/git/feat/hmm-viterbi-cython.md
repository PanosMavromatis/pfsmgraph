# feat/hmm-viterbi-cython

**Created**: 2026-09-09
**Base**: main at 68bb7d7
**Status**: active

## Purpose

Implement Viterbi at ADR 0002 phase 2 (Cython) — the first `.pyx` in any
pfsmgraph distribution, and the first compiled artifact the meson-python build
backend has actually had to build. Phase 1 landed 2026-09-04 as `_viterbi.py`;
phase 0's specification was recovered from it on 2026-09-09. This phase is
written against the specification, not against the phase-1 source.

## Scope

- Settle two questions that are open on `main` before any code is written
- `_viterbi_cython.pyx`, per the `dp-compile.toml` phase template
- Un-dormant the `hmm` meson extension block; list the new sources
- Register the backend in `_backends.py`
- Equivalence against phase 1 and the three `.vpath.xls` oracles
- Close or advance the live `first .pyx` DEFERRED items

## Context

- Master plan `docs/plan/TODO.md`, revision 02-hmm-v0.1.0, line 200
- `docs/design/algorithms/viterbi/FORMALIZATION.md` — phase 0, contract for the
  min-plus objective and the smallest-index tie-break
- ADR 0002 / 0016 (lifecycle), 0003 (one parameterized suite), 0015
  (arc-emission), 0017 (frozen parameters), 0018 (family-wide meson-python)
- `dp-compile.toml` — the manifest; phases 2-4 run under the plugin
- `docs/plan/DEFERRED.md` — "Trigger: the first `.pyx`"

## Notes
