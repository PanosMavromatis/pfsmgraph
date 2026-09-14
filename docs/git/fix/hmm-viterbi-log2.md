# fix/hmm-viterbi-log2

**Created**: 2026-09-14
**Base**: main at 85ba698
**Status**: active

## Purpose

Make every Viterbi backend take its arc costs from the same `log2`, so the backends agree
bit for bit and the first-wins tie-break is a contract independent of the CPU. Today
phase 1 (`_viterbi.py`) and phase 4's host-side table use numpy's vectorised `log2`,
while phases 2 and 3 (`_viterbi_cython.pyx`, `_viterbi_cpu_parallel.py`) call libm's
scalar one. On an AVX-512 host the two differ by one ulp in about 0.1% of inputs, so
`total_bits` can differ by an ulp and a near-tie could resolve differently. Phases 2 and 3
nonetheless claim bit-exactness with phase 1 in their docstrings. This is the first subgoal
of revision `03-hmm-v0.2.0` because its forward pass is the next consumer of `bits`, and a
sum reduction would otherwise inherit the split as a tolerance nobody chose.

## Scope

- Confirm the plugin's view of `viterbi` and reproduce the split on this host.
- Choose the fix: one numpy table for every backend, or libm in phase 1 too. Record which
  `log2` is contract.
- Change the backends and regenerate the `dp-compile` provenance chain.
- Replace phase 4's one-ulp bound with exact equality, and construct a near-tie that falls
  in the disagreeing inputs.
- Update `FORMALIZATION.md`, the phase 2/3 docstrings, `DEFERRED.md` and `core.md`.

## Context

- `docs/plan/DEFERRED.md`, `## Trigger: revision 03 or 04 opening`, "Unify the logarithm
  across the Viterbi backends": where the defect was filed, with both candidate fixes.
- `docs/design/algorithms/viterbi/FORMALIZATION.md` ("Which host `log2` is contract
  remains open").
- PR #23 (`feat/hmm-viterbi-cuda`): the phase-4 differential test that found it, and
  phase 4's host-side table.
- `docs/plan/TODO.md`, revision `03-hmm-v0.2.0`, first subgoal.

## Notes
