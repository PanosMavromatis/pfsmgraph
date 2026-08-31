# feat/dataseq-merge

**Created**: 2026-08-31
**Base**: main at 73e15a0
**Status**: active

## Purpose

Implement `pfsmgraph.dataseq` by merging the three data sequence implementations that
already exist outside this repo, taking the `dl` version as the base (PRD §3.5), and
settle the encoder API that the merge forces a decision on. This is the base layer of
the family: the other four packages are blocked on the symbol↔code encoder this branch
delivers, so the API settled here is the one `align`, `hseg`, `hmm`, and `dl` all consume
at their boundaries.

Executes the first two subgoals of revision `01-dataseq-v0.1.0`. They are on one branch
because they are mutually dependent — the merge surfaces the constraints the API has to
satisfy, and the API decision determines which of the three implementations' shapes
survives. Splitting them would mean landing an encoder and immediately rewriting it.

## Scope

- Bring the three implementations into the repo and read them side by side; tabulate
  where they diverge on container semantics, encoder shape, and vocabulary handling.
- Land the merged container in `packages/pfsmgraph-dataseq/pfsmgraph/dataseq/`, with the
  `dl` version as the base: `Dataset` conformance such that a stock `DataLoader` works
  without subclassing.
- Settle the encoder API — constructor signature, the spelling of the strictness switch,
  and how `align` consumes the mapping at its boundary — and implement encoder/decoder
  against the fixed reserved block.
- Promote ADR 0010 `Proposed` → `Accepted` and update its row in `docs/design/adr/README.md`.

**Not in scope**, though they belong to the same revision and get their own branches:
renumbering the proof-of-concept alignment code to the reserved block; the ADR 0003 test
suite; the `dependencies = []` fix; the 0.1.0 release.

## Context

- Master plan: `docs/plan/TODO.md`, revision `01-dataseq-v0.1.0` — this branch is
  backlinked under subgoals 1 and 2.
- [PRD §3.5](../../design/PRD.md) — names the `dl` implementation as the merge base.
- [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md) —
  `Proposed`, and `Proposed` *solely* because the encoder API is unresolved. Settling it
  here is what promotes it.
- [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) —
  fixed reserved block `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5, user symbols
  from 6, not configurable; strict encoding by default with `UNK` fallback opt-in. Settled;
  not up for relitigation on this branch.
- [ADR 0009](../../design/adr/0009-dataseq-as-the-base-layer.md) — why `dataseq` has no
  intra-family dependencies.
- [ADR 0008](../../design/adr/0008-per-package-build-backends.md) — `dataseq` presumes
  hatchling. If the merge turns up a compiled inner loop, that presumption breaks and the
  build-backend subgoal stops being a formality.
- `docs/plan/DEFERRED.md`, trigger "the `dataseq` merge" — the register of what this
  revision must not leave behind.

**Dependency:** the preexisting code is not in this repo yet. The first goal cannot start
until it is imported, and will be marked `[!]` if that has not happened.

## Notes

Running log — decisions made, things tried, things deferred.

- 2026-08-31 — Branch created from `main` at 73e15a0. No code imported yet.
