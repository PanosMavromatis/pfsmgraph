# The reserved block across four implementations and ADR 0011

The first point at which the whole picture exists. `COMPARISON.md` in `hmm-lush/` built this
table with three rows and a placeholder for the fourth; goal 4 supplies it, so the table is
restated here in full rather than amended in place. Nothing below reopens
[ADR 0011](../docs/design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) —
it is Accepted, and this is the divergence record it should carry, not a case against it.

## 1. The table

| Source | Reserved codes | User symbols from | `GAP`? | Strict? | Decodes? |
|---|---|---|---|---|---|
| `dl` (MelodyHPO) | `PAD` 0, `BOS` 1, `EOS` 2 | **3** | no | no — silent `NaN` | no |
| Lush (`hmm`) | `begin` 0, `end` 1 | **2** | no | no — nothing offered | no |
| `segalign` (rudimentary) | `:EOS` 0, `:PAD` 1 | **2** | no — `'.'` *deleted* | no — unknown → `:PAD` | no |
| `tokalign` (proof-of-concept) | pad 0, BOS 1, EOS 2, gap 3 | **4** | **yes**, index 3 | **yes** — `KeyError` | **partial** |
| **ADR 0011 — authoritative** | `PAD` 0, `UNK` 1, `BOS` 2, `EOS` 3, `GAP` 4, `MSK` 5 | **6** | yes, index 4 | yes, `UNK` opt-in | required |

Source lines: `melody_hpo/data/control.py`; `dsource-seq.lsh:83` and `format-sds.lsh`;
`segalign/src/segalign/seq/dataset.py:66-67, 105, 124-130, 295-296`;
`tokalign/src/tokalign/_types.py:21-46`.

## 2. Reading the table

**Four sources, three distinct offsets — and the collision is between two containers.**
Lush and `segalign` both start user symbols at 2, and they are the only pair that agrees on
anything here. The agreement is worthless: Lush's low pair is `begin`/`end` and `segalign`'s
is `:EOS`/`:PAD`, so the *same two integers carry different meanings*, and `0` means `begin`
in one and `:EOS` in the other. There was still never a majority to defer to.

**This corrects two claims that stood until now**, in `docs/agents/core.md`'s reserved-block
invariant and in `hmm-lush/COMPARISON.md` §3.1, which it derives from. Both said "every one
of the three uses a different offset — `dl` at 3, the Lush original at 2, the proof-of-concept
at 4", and both said none of the three has a `GAP` code.

- The **trio was miscounted**. It named two containers and the proof-of-concept, which is the
  fourth source rather than the third container. The three containers are `dl` 3, Lush 2,
  `segalign` 2 — so among the containers the offsets *collide*, and it is only across all
  four sources that three distinct values appear.
- **`tokalign` does have a `GAP` code**, at index 3. That is precisely why its user symbols
  start at 4, so the offset the claim cites and the gap it denies are the same fact.

Neither was careless. Both were written before `segalign` and `tokalign` had been read: the
proof-of-concept row was filled in from `DEFERRED.md`'s recollection, which predicted the
offset correctly and the gap wrongly. Measurement was the only thing that could settle it,
and this table is the first point at which all four sources had been read.

**Only `tokalign` has a gap code, and only `tokalign` was written to align sequences.** That
is not a coincidence and it is the whole argument for `GAP` being a reserved slot rather
than a user symbol: a gap is the one symbol an aligner *emits* and no corpus ever contains.
The three containers had no reason to invent it, and did not.

**`segalign` supplies the sharpest evidence for `GAP`, by deleting it.** Its corpus uses
`'.'` as a no-pitch sentinel, and `Dataset` strips it in both directions — out of every
extracted sequence (`dataset.py:105`) and out of the vocabulary before codes are assigned
(`del collected_tokens['.']`, lines 295-296). `tokalign`'s `Alphabet` defaults `gap_symbol` to
**`'.'`** — the same character, from the same corpus lineage, given a reserved index. One
implementation destroys the information the other reserves a code for. ADR 0011 sides with
`tokalign`, and this is the observation that makes that a finding rather than a preference.

**Only `tokalign` is strict, and each of the other three fails differently.** `dl` returns a
silent `NaN` from `Series.map`, which also promotes the column to `float64` so integer codes
become floats. Lush offers no path at all. `segalign` maps every unseen symbol to `:PAD`
(`dataset.py:124-130`) — and unlike the other two this is **deliberate and pinned by a test**
(`test_dataset.py:282`, `assert encoded[0] == [2, 1, 3]  # UNKNOWN mapped to :PAD`). It is
the failure ADR 0011's separate `UNK`=1 exists to prevent, implemented on purpose: after
encoding, "symbol I have never seen" and "no symbol here" are the same integer, and nothing
downstream can tell them apart.

## 3. What the renumbering costs

ADR 0011 lands as part of the merge, not after it. Per source:

| Source | Cost of moving to `PAD` 0 … `MSK` 5, users from 6 |
|---|---|
| `dl` | Mechanical. Codes are assigned in one place; no persisted vocabulary exists to migrate (`ANALYSIS.md` §3.4) |
| Lush | Not migrated — the translation is a reading aid, and its `0 = begin` collision is recorded as evidence, not carried |
| `segalign` | Mechanical, but reverses a *tested* contract: the unknown → `:PAD` test must be rewritten to assert a raise |
| `tokalign` | Renumbering only — see `align-poc/COMPARISON.md` §2. `gap_index` moves 3 → 4 and `test_gap_index_is_reserved_count` changes with it |

No source persists a vocabulary to disk, so **no stored artefact anywhere encodes an old
offset**. The renumbering is a source-only change with no migration path to write — which is
exactly why ADR 0011 requires it to land now rather than at the first release.
