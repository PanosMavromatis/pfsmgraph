# feat/hmm-batched-training

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Batch the trainer over sequences (revision 03-hmm-v0.2.0)

## Goals

- [x] Decide the batched surface: what `baum_welch` takes beyond `backend=` (device, batch size), and whether batching is internal or a new argument
  > **Q:** How should batching appear on `baum_welch`'s public surface: a `batch_size=` keyword, internal only, or the caller passing `pad_collate` batches?
  > **A:** A `batch_size: int | None = None` keyword, `None` meaning the whole corpus as one batch. It is a performance and memory knob only: EM already sums counts over records before the M-step, so the result must not depend on it (bit-exact on `python`, within ADR 0020's tolerance on `torch`), and a test asserts that.
  > **Q:** Does the public call gain a device argument now?
  > **A:** Yes: `device: str | None = None`, torch only. `None` is CPU on every backend; a non-CPU device with `backend="python"` raises `ValueError`, and a device torch cannot reach raises `BackendUnavailableError`. No fallback and nothing environmental, per ADR 0021. Implemented in goal 3.
  > **Q:** What does a `baum_welch` backend row point at once batching exists?
  > **A:** A batched kernel replaces `_e_step` in the table: `_e_step_batch(init_state_p, transition_p, output_p, codes, lengths, device)` over `pad_collate`'s `(B, L)` codes, returning the counts summed over the batch and the description length per record. One record is the `B = 1` case, so each backend keeps one kernel and `backends()` gains no key. The per-record `_e_step` stays as a private helper, since `test_baum_welch.py`, `test_baum_welch_backends.py` and the hmmlearn oracle call it.
  > **Done:** `baum_welch(params, records, *, backend="python", batch_size=None, device=None, ...)`; the result is invariant under `batch_size`, `device` is torch's alone, and each row's kernel is batched. Whether `viterbi` batches too is goal 4.
- [x] Batch the E-step over a padded corpus with `pad_collate`'s mask, on the `python` reference
  > **Note:** a batched numpy pass can be bit-exact with the per-record reference *within* a record: `np.add.accumulate` over a state axis is elementwise across a leading batch axis, a padded step keeps α with `scale = 1` (`bits(1.0)` is `-0.0`, which adds exactly), and ξ at padded positions is selected to exact `0.0`. It cannot be bit-exact *across* records if a kernel sums its batch: `(c₀ + c₁) + c₂ != c₀ + (c₁ + c₂)` in 213 of 1000 random triples, so regrouping by `batch_size` moves the last bits.
  > **Q:** How do the batched kernels hand counts back, given that regrouping the sum over records changes bits?
  > **A:** Per record: `(B, S)`, `(B, S, S)` and `(B, S, S, A)` counts plus `(B,)` bits, which `_corpus_step` adds in ascending record order as it does today. The `python` path is then bit-exact with the per-record reference at every `batch_size`; memory is `O(B·S²·A)` per batch, which is exactly what `batch_size` bounds. This supersedes goal 1's "summed over the batch". `torch` needs per-record leaves to get per-record gradients (goal 3).
  > **Done:** `_e_step_batch` in `_forward_backward.py` is the `python` row. `baum_welch(..., batch_size=None)` feeds `pad_collate` batches through `_corpus_step`, which sums the per-record counts in record order. `torch`'s row is a temporary per-record loop with the same signature. The backend kernel tests now run each kernel as a one-record batch. Suite 833 → 842.
  - [x] Padded positions contribute exactly zero to every count and to the description length
    > **Note:** mutation-tested. Leaving the scale factor unreset at a padded step fails 6 of 6 cases, and seeding padded β with 1 fails 5 of 6. Three mutants are equivalent. Not carrying α past a record's end changes nothing, since nothing reads it. Dropping ξ's mask changes nothing on any valid `HMMParams`, because `PAD`'s fibre is zero, so padding already weighs 0: ADR 0011's block masks it arithmetically. Multiplying by the mask instead of `np.where` changes nothing either, while α and β stay finite. The `pad-emittable` case passes raw arrays with `PAD` emittable, and so makes the mask load-bearing; it catches the dropped mask.
  - [x] Batched counts match per-record counts on ragged corpora, with the tolerance or exactness stated
    > **Note:** exact. Each row of `_e_step_batch` equals `_e_step` under `assert_array_equal` on six ragged cases: empty and length-1 records, `(200, 1, 1, 1)`, an `UNK`-impossible record, dead arcs, and emittable `PAD`. `baum_welch` is bit-identical at `batch_size` 1, 2, 3 and `None` over 25 cycles. `_corpus_description_length`, the convergence check every `batch_cycles`, stays per-record and unbatched.
- [ ] Batch the `torch` E-step, and settle device placement
- [ ] Decide whether `viterbi` gains a batched form on this branch
- [ ] Document the batched call and amend ADRs where the decisions land
