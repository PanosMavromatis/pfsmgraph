# feat/hmm-batched-decode

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Batch the decode across all four ADR 0016 phases (revision 03-hmm-v0.2.0)

## Goals

- [ ] Decide the batched decode's public surface: a new call or a widened `viterbi`, its per-record result, how an impossible record in a batch is reported, and whether `batch_size=` applies
- [ ] Batch the `python` reference kernel over `pad_collate`'s padded records
  - [ ] Each row is bit-exact with the per-record `_viterbi`, including ties, empty and length-1 records, and an impossible record beside possible ones
  - [ ] Padding is shown load-bearing, with a mutation check that a padded step reaching δ or ψ is caught
- [ ] Batch the `cython` kernel, bit-exact with the batched reference
- [ ] Batch the `cpu_parallel` kernel, settling which axis `prange` takes now that records are independent
- [ ] Batch the `cuda` kernel, bit-exact with the batched reference, and measure the batch against the per-record decode with the host named
- [ ] Document the batched decode and amend `FORMALIZATION.md` and ADR 0021 where the decisions land
