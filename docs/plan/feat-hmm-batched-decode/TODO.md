# feat/hmm-batched-decode

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Batch the decode across all four ADR 0016 phases (revision 03-hmm-v0.2.0)

## Goals

- [x] Decide the batched decode's public surface: a new call or a widened `viterbi`, its per-record result, how an impossible record in a batch is reported, and whether `batch_size=` applies
  > **Q:** Should the batched decode be a new public call or a widened `viterbi`?
  > **A:** A new call, `viterbi_batch(params, records, *, backend="python", batch_size=None, on_impossible="raise")`, taking `records` as `baum_welch` does. `viterbi` stays one record in, one path out, so neither call's return type depends on its input.
  > **Q:** What should it return for each record?
  > **A:** `list[ViterbiPath]`, one per record in record order with labels kept. No new public type, and each row is directly comparable with `viterbi(params, record)`.
  > **Q:** How should an impossible record in the batch be reported?
  > **A:** An `on_impossible=` policy validated before any work, as `encode`'s `on_unknown` is: `"raise"` (the default) raises `ImpossibleSequenceError` naming the record's index and its dead symbol; `"none"` puts `None` in that slot and decodes the rest, which is what revision 04's topology search needs.
  > **Q:** Should `batch_size=` apply to the batched decode?
  > **A:** Yes, with `baum_welch`'s meaning: `None` is one kernel call over every record, an integer bounds how many are padded together, and it changes no result. It bounds memory, since ψ grows as `B·L·S`.
  > **Done:** surface settled. The public call stays one validated wrapper over a purely numeric batched kernel, so the kernel still neither validates nor raises and returns `inf` per row; `viterbi_batch` range-checks every record before the first kernel call and turns an infinite row into the policy's outcome. It joins `__all__` as a tenth name.
  > **Note:** the empty record (length 0) is allowed in a batch, as it is in `viterbi`, and `pad_collate` pads it like any other; an empty `records` has no batch to collate, and what it returns (an empty list, or `ValueError` as `pad_collate` raises) is left to goal 2's tests rather than decided here. ADR 0021's Open item is amended with these answers in goal 6.
- [ ] Batch the `python` reference kernel over `pad_collate`'s padded records
  - [ ] Each row is bit-exact with the per-record `_viterbi`, including ties, empty and length-1 records, and an impossible record beside possible ones
  - [ ] Padding is shown load-bearing, with a mutation check that a padded step reaching δ or ψ is caught
- [ ] Batch the `cython` kernel, bit-exact with the batched reference
- [ ] Batch the `cpu_parallel` kernel, settling which axis `prange` takes now that records are independent
- [ ] Batch the `cuda` kernel, bit-exact with the batched reference, and measure the batch against the per-record decode with the host named
- [ ] Document the batched decode and amend `FORMALIZATION.md` and ADR 0021 where the decisions land
