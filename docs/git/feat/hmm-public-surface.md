# feat/hmm-public-surface

**Created**: 2026-09-03
**Base**: main at 0db06f3
**Status**: active

## Purpose

Settle the public surface of `pfsmgraph.hmm` 0.1.0 and where it meets `dataseq`: what a
caller constructs, what Viterbi is a method *on* given there is no trainer object in this
release, and which of `SymbolTable`, the record container and `pad_collate` it consumes.
Apply *encode at the boundary* (ADR 0001) by naming the exact entry and exit points where
strings are still permitted. Decide the class architecture explicitly — Lush's
`hmm`/`hmm-param`/`hmm-trainer` split, or something else — rather than defaulting to it by
not deciding.

## Scope

- Decide what a caller constructs and what Viterbi is a method on (no trainer object
  exists in 0.1.0 per `HMMLIB-ACCOUNT.md` §15's falsifier-3 finding)
- Decide the class architecture: Lush's three-way split vs. an alternative, weighing the
  GUI-undo motivation (`keep-model`/`reset-model`) against modern-framework idioms
  (immutable/frozen parameter objects, PyTorch conventions)
- Name the exact string/integer boundary per ADR 0001
- Settle `SymbolTable` consumption: take it explicitly, don't reproduce Lush's
  file-coupled alphabet read out of `.sds` (`HMMLIB-ACCOUNT.md` §4)
- Settle whether Viterbi 0.1.0 consumes a single `dataseq` record directly, leaving
  `pad_collate` out of scope for this subgoal (deferred to revision 03's batched training)
- Record the class-architecture decision as an ADR, on the order of ADR 0010/0015, per
  the master plan's framing

## Context

Master plan subgoal at `docs/plan/TODO.md:109`, revision `02-hmm-v0.1.0`. Follows
`docs/hmmlib-account` (PR #13) and `docs/cpu-parallel-phase` (PR #14, ADR 0016). The
class-architecture question was raised and discussed at length earlier this session —
leaning toward an immutable/frozen parameter object over Lush's mutable `hmm`/`hmm-param`
split, pending a check of revision 04's actual accept/reject ratio for the topology
search.

## Notes

_(running log — filled in as work proceeds)_
