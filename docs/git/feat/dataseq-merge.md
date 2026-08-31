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
- 2026-08-31 — Lush implementation imported to `.scratch/hmm-lush/`; 135 tracked files
  (≈129 KB) out of 929 MB, again deny-by-default. Not version-controlled, so provenance
  is dates rather than a revision: sources span 2008–2011. Neither of the MelodyHPO
  renames applied — no `.git`, no nested `CLAUDE.md` — both checked before reading.
  The import already answers the question goals 3–4 were given: this encoder is a
  **dense vocabulary index**, built in first-appearance order, and its reserved block is
  `0 begin`, `1 end` with user symbols from **2**. That is a third offset, and the first
  one that collides with ADR 0011 destructively rather than merely differing: the loader
  packs ragged sequences into a dense `size × seq_size_max` integer matrix, so the unused
  tail of every short row is `0`, which here means `begin`. The original is safe only
  because `_seq_sizes` is consulted at every read. This is the ADR 0011 `PAD`=0 rationale
  observed in the wild rather than argued from first principles.
- 2026-08-31 — Precedence between the ADRs and the three imports stated explicitly, as a
  new invariant in `docs/agents/core.md`: the `dl` version is the merge's *base* because
  it is the most mature, but base means starting point, not authority. Where any import
  disagrees with an Accepted ADR the ADR wins, unless that ADR carves out an exception in
  its own text — as ADR 0011 does for the *spelling* of the strictness switch while
  holding its semantics settled. Prompted by re-reading goals 3–4, whose "every point
  where that base must be overridden" reads, on its own, as licence to let an import
  reopen a closed decision; it means the points the ADRs leave open. Sharpest instance:
  the three imports start user symbols at 3 (`dl`), 2 (Lush) and 4 (proof-of-concept),
  and none of them has a `GAP` code at all, so there was never a majority to defer to —
  ADR 0011 is supplying something all three lack rather than breaking a tie. This also
  reclassifies the dense-matrix finding logged above: it is an illustration of the
  failure ADR 0011 prevents, not evidence for `PAD`=0, and goal 3 now carries a note
  saying so before its semantic account is written.
- 2026-08-31 — Goal 3 subgoals 2 and 3 done. `.scratch/hmm-lush/ACCOUNT.md` describes the
  Lush original in its own terms, before any translation; `.scratch/hmm-lush/translation/`
  is a runnable, stdlib-only rendering of it. Three findings changed the shape of the merge
  question. **(1)** `format-sds` is a free `de`, not a method, so the vocabulary is a
  build-time artefact and a loaded container has no encoding path at all — which is why the
  strict-vs-`UNK` question never arises in the original. It is a structural absence, not a
  lenient choice, and that reframes what ADR 0011's strictness clause adds. **(2)** The dense
  `size × seq_size_max` matrix is a staging buffer, not the corpus representation:
  `hmm-trainer.lsh:66` is the only caller of `fprop-all` and nothing else reads `seq-data`,
  so `load` unpacks ragged data into a rectangle whose sole consumer immediately repacks it
  ragged. 71% of `set01z0`'s matrix is `begin`-valued padding bought for nothing — the
  hazard was taken on without even the performance argument that usually excuses it.
  **(3)** The `%l`/`%s` split between the two `_alphabet` writers is not an oversight but a
  downstream cost of the compile boundary: `alphabet` is `-idx1- (-gptr-)` because
  `fprop-all` and `set-alphabet` are what `dhc-make` compiles, and `dsource-seq.lsh:83`
  (`str-ptr (symbol->string s)`) is where the knowledge that a name ever needed delimiters
  is discarded. `format-sds`, outside the class, keeps symbols and can still use `%l`. Dates
  agree: 2009-07-05 precedes 2009-07-15. Corrected in the account after review — the tracked
  corpora are fine, `|…|` is multiple-escape and round-trips, and only the unmeasured half
  (that `%s` emits the string bare) was ever at issue; it is now flagged as inference.
  The translation's rebuild check re-derives both corpora from their own `_raw_data` and
  reproduces the 2009 output exactly, so first-appearance ordering, `begin`/`end` wrapping
  and the dense packing are confirmed rather than assumed. The goal-2 gitignore trap fired
  again — deny-by-default `/*` swallowed the whole `translation/` directory — and was caught
  before the commit this time, by running `git check-ignore` per file rather than assuming.
- 2026-08-31 — Goal 3 complete. `.scratch/hmm-lush/COMPARISON.md` sets the Lush
  implementation against the `dl` base, with §2 (overrides, on matters the ADRs leave open)
  and §3 (divergences, ADR-settled) kept in separate sections so the two cannot blur when
  this feeds ADR 0010. Five overrides: first-appearance ordering rather than set iteration;
  per-sequence true lengths as container state; vocabulary persistence with the format owning
  its quoting rule; frozen-as-explicit-state; decode on the tested surface. Two of those turn
  arguments already made in `ANALYSIS.md` into evidence — the ordering fix was proposed there
  on stability grounds and is now backed by the translation reproducing the 2009 `_alphabet`
  code for code, and `dependencies = []` was argued there and is now supported by a
  stdlib-only translation that handles both corpora.
  The finding neither analysis could have produced alone: **both implementations bake a
  consumer's view into the container** — `dl` next-token `(input, target)` pairs, Lush the
  flat concatenated stream Baum-Welch wants. Two unrelated consumers, the same category
  error, arrived at independently, which promotes `ANALYSIS.md` §2.2's separation from a
  judgement call about layering to a correction of a mistake made twice.
  Scoping honoured: `seq-state` is named as an `hmm` object, not a `dataseq` one. What
  `dataseq` owes it is a per-sequence view an annotation can align against, which is the
  concrete reason the true-lengths override is not merely tidiness.
  The branch plan also gained a `## Constraint on every comparison below` header block, and
  three subgoal criteria were reworded to carry the ADR-precedence rule inline. The notes
  alone were not enough: `/hitl-step` treats subgoals as the acceptance criteria and notes as
  context, so a constraint stated only in commentary does not bind the executor.
