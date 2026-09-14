# feat/hmm-batched-training

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Batch the trainer over sequences (revision 03-hmm-v0.2.0)

## Goals

- [ ] Decide the batched surface: what `baum_welch` takes beyond `backend=` (device, batch size), and whether batching is internal or a new argument
- [ ] Batch the E-step over a padded corpus with `pad_collate`'s mask, on the `python` reference
  - [ ] Padded positions contribute exactly zero to every count and to the description length
  - [ ] Batched counts match per-record counts on ragged corpora, with the tolerance or exactness stated
- [ ] Batch the `torch` E-step, and settle device placement
- [ ] Decide whether `viterbi` gains a batched form on this branch
- [ ] Document the batched call and amend ADRs where the decisions land
