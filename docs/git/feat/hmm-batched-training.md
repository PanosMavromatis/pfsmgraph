# feat/hmm-batched-training

**Created**: 2026-09-14
**Base**: main at f17a6ca
**Status**: active

## Purpose

Batch `baum_welch`'s E-step over records, revision 03's "Batch the trainer over sequences"
subgoal. It is the first place `pad_collate`'s mask does real work: padded timesteps must
contribute nothing to the expected counts, and a mask bug is silent, since it shifts the
estimates rather than raising.

## Scope

- Decide the batched call's surface beyond `backend=` (device index? batch size?) under ADR 0021
- Batched E-step on the `python` reference and the `torch` backend, fed by `pad_collate`
- Hold batched against per-record counts on ragged corpora, including records where padding dominates
- Decide in-plan whether `viterbi` gains a batched form here
- Docs: `docs/api/hmm/baum_welch.md`, ADR amendments as needed

## Context

- `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, and its two notes on this subgoal
- ADR 0020 (evaluation order, torch tolerance), ADR 0021 (backend selection; leaves batching and devices here)
- Builds on PR #34 (`baum_welch`, torch backend) and PR #33 (backend seam)

## Notes
