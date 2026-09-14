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
- [x] Batch the `python` reference kernel over `pad_collate`'s padded records
  > **Q:** Where does `viterbi_batch`'s kernel register in the backend table while only `python` has one?
  > **A:** A new `viterbi_batch` key in `_TABLE`, with a `python` row now and one row added by each of goals 3-5, and `viterbi_batch` joins `_PUBLIC`. ADR 0021 keys the table by public call, and a second function on the `viterbi` rows would let a row be available for one call and not the other, which `backends()` cannot report.
  > **Q:** What should `viterbi_batch(params, [])` do?
  > **A:** Return `[]` without calling a kernel, after validating `backend`, `batch_size` and `on_impossible`, so a bad argument fails identically on any input.
  > **Done:** `_viterbi_batch(init_state_p, transition_p, output_p, codes, lengths) -> (states (B, L+1), total_bits (B,))` beside `_viterbi`, and the public `viterbi_batch` over it, registered as `_TABLE["viterbi_batch"]` with a `python` row and exported as the tenth name. `viterbi`'s range and impossibility messages moved into `_range_error` and `_impossible_message`, byte-identical, so the batch prefixes `record {i}: ` to the same wording. `tests/test_viterbi_batch.py` holds 28 tests under its own `backend` fixture; the suite is 930.
  > **Note:** registering a public call changes two pasted outputs under `docs/api/hmm/`, both caught by `test_api_docs.py`: `backends.md`'s refusal lists the public calls, and `README.md` pastes `__all__` (with a "exactly nine names" count and a surface table beside it). Those three lines moved here so the gated commit stays green; the documentation proper is still goal 6.
  - [x] Each row is bit-exact with the per-record `_viterbi`, including ties, empty and length-1 records, and an impossible record beside possible ones
    > **Note:** measured before writing, on a scratch prototype: 2470 rows over 400 models up to 39 states, every tenth exactly uniform, 0 mismatches in states or in `total_bits` bytes. The one numeric risk was `bits` over an `(R, S, S)` array built by fancy indexing and `moveaxis` rather than the per-record C-ordered `(S, S)`, since numpy's `log2` has differed by layout here before; on this host it does not. The suite pins it on 200 models up to 40 states, and against the 1269-symbol corpus padded beside short records.
  - [x] Padding is shown load-bearing, with a mutation check that a padded step reaching δ or ψ is caught
    > **Note:** the δ half is load-bearing and the ψ half cannot be. Mutation-tested by swapping a mutant in before the table's first probe: taking the padded step (δ and ψ) fails 11 of 28 tests, including `test_padding_stays_inert_where_pad_is_emittable`, where raw arrays make `PAD` the cheapest symbol so a padded step is finite rather than infinite. Writing ψ at padded steps while carrying δ passes all 28, and must: every backtrace starts at its record's own `n_b` and reads only `psi[b, 1..n_b]`, so ψ past a record's end is unread by construction. The δ carry is made observable by one choice in the kernel, reading the final argmin from column `L` for every row rather than from each row's `n_b`; a kernel that read `n_b` would make the carry an equivalent mutant too, which the compiled phases should not "optimise" towards.
- [ ] Batch the `cython` kernel, bit-exact with the batched reference
- [ ] Batch the `cpu_parallel` kernel, settling which axis `prange` takes now that records are independent
- [ ] Batch the `cuda` kernel, bit-exact with the batched reference, and measure the batch against the per-record decode with the host named
- [ ] Document the batched decode and amend `FORMALIZATION.md` and ADR 0021 where the decisions land
