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

**Dependency:** each import goal waits on its own source arriving in `.scratch/`. The
merge base (goal 2) landed on 2026-08-31 — MelodyHPO, `main` at `5f42311`; the Lush and
rudimentary implementations (goals 3–4) are still outside the repo and will be marked
`[!]` if they have not arrived when their turn comes. Goal 1, creating the scratch
location, never depended on any of them and is complete.

## Notes

Running log — decisions made, things tried, things deferred.

- 2026-08-31 — Branch created from `main` at 73e15a0. No code imported yet.
- 2026-08-31 — Import goal split into a prelude, one goal per implementation in merge
  order, and a postlude. Under the Lush import, the semantic account of the original now
  precedes the translation: a close translation is defined by fidelity to something you
  must already understand, so translating first would run the comparison against our
  reading of the original rather than the original itself.
- 2026-08-31 — Merge base imported. It is **MelodyHPO**, not a project named `dl`: that
  label is this repository's package slot, applied retroactively to a standalone and now
  defunct project. Copied whole (2.2 GB) and reduced to 33 tracked files by a
  deny-by-default `.scratch/dl/.gitignore`. Two renames inside the copy were forced
  rather than chosen — `.git` → `.git-disabled`, since an embedded repository makes
  `git add` on interior paths a silent no-op that exits `0`; and `CLAUDE.md` →
  `CLAUDE.md.orig`, since a nested `CLAUDE.md` is loaded when reading files beside it and
  would have injected another project's agent instructions into sessions here.
- 2026-08-31 — Scratch location is `.scratch/` at the repo root, tracked, with one
  subdirectory per implementation. The leading dot is load-bearing: `uv run pytest` has no
  `testpaths`, so its walk reaches everywhere, and the dot matches pytest's default
  `norecursedirs` entry `.*` — no config to add now or remove at cleanup. Verified with a
  deliberately-failing canary inside `.scratch/` (not collected) against the same file at
  the repo root (collection error). Root `pyproject.toml` untouched.
- 2026-08-31 — Merge base analysed; `.scratch/dl/ANALYSIS.md`. The base fuses two different
  kinds of encoder into one map: a dense vocabulary index (`PAD`/`BOS`/`EOS` at 0/1/2) and a
  stateless structured code (`PitchCode`, `100*chromatic + diatonic`, range 1207-13176).
  **Decided:** `dataseq` owns the dense index per ADR 0011; structured codecs are demoted to a
  symbol canonicaliser running before vocabulary assignment. The decisive argument is `hmm`,
  whose V x V transition matrix would be ~1.74e8 entries at V = 13,177 for an alphabet of a few
  dozen real symbols. This is the substance of what ADR 0010 is `Proposed` pending, so goal 7
  now has its decision section in draft.
