# feat/hmm-batched-decode

**Created**: 2026-09-14
**Base**: main at ac93c65
**Status**: active

## Purpose

Give `viterbi` a batched form over `pad_collate`'s padded records, with each of the four ADR 0016 phases' kernels batched rather than looped per record. A single-record decode cannot show phase 4's advantage: it pays a fixed launch cost per timestep over only `O(S²)` work, and a batch axis is what gives the device enough work per launch. Because a min-sum decode performs only `+` and `<`, every batched kernel is held **bit-exact** with the per-record decode, not within a tolerance, and the first-wins tie-break stays contract.

## Scope

- Decide the public surface: a batched call beside `viterbi` or a widened `viterbi`, what it returns per record, how an impossible record in a batch is reported, and whether a `batch_size=` knob (as on `baum_welch`) applies.
- Batch the `python` reference kernel over `(B, L)` codes and `lengths`; padded steps leave δ and ψ untouched, and each record's backtrace starts at its own last position.
- Batch the `cython`, `cpu_parallel` and `cuda` kernels, each bit-exact row by row with the per-record decode on ragged batches, on the shared ADR 0003 suite through `backend=`.
- Measure the batch against the per-record decode per phase, with the host named, and record whether phase 4's crossover moves.
- Document the call in `docs/api/hmm/viterbi.md`, update `viterbi/FORMALIZATION.md`'s decomposition section, and amend ADR 0021's Open item.

## Context

- Master-plan subgoal "Batch the decode across all four ADR 0016 phases" (revision 03-hmm-v0.2.0), deferred from `feat/hmm-batched-training` (PR #35).
- `viterbi/FORMALIZATION.md` "The two decompositions not taken": a batch of independent sequences, "the best speed story … unavailable at this signature".
- ADR 0016 (phase 3 parallelises states within one timestep), ADR 0021 (`backend=`, no fallback; Open: "A batched `viterbi`").
- Revision 02's per-phase branches: `feat-hmm-viterbi-{python,cython,cpu-parallel,cuda}` under `docs/plan/02-hmm-v0.1.0/`.
- `dp-compile` gate: staging any of the four `_viterbi*` kernel paths runs the build and test commands before the commit.

## Notes
