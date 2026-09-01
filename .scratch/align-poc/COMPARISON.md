# `tokalign`'s `Alphabet` against the merged encoder

The **encoder** half of goal 4. The container half is `../py-rudimentary/COMPARISON.md`; the
reserved block for all four sources is `../RESERVED-BLOCK.md`.

This document is read under a rule that runs the other way from the other three. `tokalign`
is the proof-of-concept library PRD §1.2 describes and
[ADRs 0001-0004](../../docs/design/adr/README.md) derive from, so it agrees with the record
by construction and a divergence is more likely a later decision than a defect. The ADRs are
still the later word — but "`tokalign` does X and the merge does not" has to be checked
against the record written *from* it before it counts as a finding.

It is also not a fourth `dataseq` implementation. There are three containers; `tokalign`
contributes the encoder, which is why
[ADR 0010](../../docs/design/adr/0010-dataseq-composition-merging-three-implementations.md)
requires reconciling this `Alphabet` as part of a merge whose own title says three.

Per the goal-4 Q&A, this **tabulates and proposes**. Goal 6 settles and implements the
encoder API; goal 7 promotes ADR 0010.

## 1. The axis: the same mapping, built once and frozen

`Alphabet` (`_types.py:9-80`) is a `frozen=True` dataclass holding `symbols: tuple[str, ...]`
and deriving both directions in `__post_init__`. Set against the gaps the other analyses
name, it already answers four of them:

| Gap named elsewhere | `Alphabet` |
|---|---|
| `ANALYSIS.md` §3.5 — no vocabulary object at all | **is** one, and the only one of the four |
| `ANALYSIS.md` §3.2, `hmm-lush` §2.4 — frozen is a structural accident | `frozen=True` + `tuple` — immutable by construction |
| `ANALYSIS.md` §3.3, `hmm-lush` §2.5 — nothing decodes | `decode` exists and is tested (§3.2 below qualifies this) |
| `RESERVED-BLOCK.md` — strictness | `encode` raises `KeyError` with the offending symbol and the valid set |

This is why the goal-4 reword calls it the most developed of the four, and why the
reconciliation is a merge into the merged encoder rather than a rewrite of it.

## 2. Renumbering, not repair

`RESERVED_INDICES = 3` with the gap at 3 and user symbols from 4; ADR 0011 fixes `PAD` 0,
`UNK` 1, `BOS` 2, `EOS` 3, `GAP` 4, `MSK` 5 and user symbols from 6. Mechanically:

- `gap_index` moves 3 → 4
- `size` becomes `6 + len(symbols)`, from `RESERVED_INDICES + 1 + len(symbols)`
- `UNK` and `MSK` are new — `Alphabet` has **no `UNK` slot at all**, so its strictness is
  total rather than a default with an opt-in. ADR 0011 wants strict-by-default *with* an
  `UNK` fallback available, which is strictly more than this offers
- `test_gap_index_is_reserved_count` and `test_user_symbols_start_after_gap` change with it,
  and `test_encode_pair`'s literal `[4, 5]` / `[6, 7]` become `[6, 7]` / `[8, 9]`

No stored artefact encodes the old offset (`RESERVED-BLOCK.md` §3), so there is no migration
to write. **This is the renumbering the branch plan anticipates, and nothing here reopens it.**

## 3. Where the merged encoder must not follow

Two genuine defects. Both survive the "source of the ADRs" caveat, because neither is a
decision — one is an annotation slip and the other is an unfinished table.

### 3.1 The reserved block is per-instance configurable, by accident

`RESERVED_INDICES: int = 3` is annotated as a plain field, not a `ClassVar`, so it is part
of the generated `__init__` — positionally, third:

```python
>>> a = Alphabet(symbols=("D3", "F3"))
>>> b = Alphabet(("D3", "F3"), ".", 7)
>>> b.gap_index, b.size, b._sym_to_idx
(7, 10, {'.': 7, 'D3': 8, 'F3': 9})
>>> a == b
False
```

Two alphabets over identical symbols, with disjoint code assignments, comparing unequal —
and since `Alphabet` is `frozen` it is also hashable, so both can key the same dict. The
comment directly above the field says the indices are reserved for interop; the annotation
says they are a constructor argument.

ADR 0011 requires the block to be **fixed, not configurable**, and this is the clearest case
in the whole comparison of the ADR winning over an import without any tension: the import
does not *intend* to offer this. It is one missing `ClassVar[int]`. The merged encoder must
make the block structurally unreachable from the constructor, and — because this failed
silently here — that should be asserted by a test rather than left to the annotation.

### 3.2 `decode` cannot round-trip a padded buffer

`_idx_to_sym` is populated from the gap index upward (`_types.py:40-45`); the reserved
indices are never inserted. So:

```python
>>> a.decode([0, 4, 5])
KeyError: 0
```

The tests cover the round trip and the gap index, and stop there — no test decodes a
reserved index, which is how this survived.

It matters more than it looks. `PAD` = 0 exists precisely so PyTorch's zero-fill idioms
(`pad_sequence`, `torch.zeros()`) mean "absent", per the invariant in
`docs/agents/core.md` — so a zero-padded batch is **the** array shape a caller most wants to
decode, and it is the one that raises. `hmm-lush/COMPARISON.md` §2.5 already requires decode
to be a live path; this adds that it must be **total over the full code range**, reserved
codes included, and that `decode(encode(x))` round-tripping is not sufficient evidence.

The fix is not just filling in the table — it is a decision goal 6 owns: whether `decode`
renders `PAD` as a symbol, drops it, or takes a flag. That is the same question
`ANALYSIS.md` §3.6 raises as "padding is emitted but never masked", arriving from the
opposite direction.

## 4. What `tokalign` contributes that no container has an analogue for

### 4.1 `ScoringMatrix` — what "encode at the boundary" is *for*

`ScoringMatrix` (`_types.py:82-152`) stores an `(alphabet.size, alphabet.size)` `float64`
array, is built from a human-readable `dict[tuple[str, str], float]`, and is read in the hot
loop as `score(i, j)` on two integers — with `score_symbols(a, b)` kept as an explicitly
non-hot-path convenience.

That pairing is the invariant in executable form: strings at the construction boundary,
integers everywhere inside. `../py-rudimentary/COMPARISON.md` §5 has the counter-example in
the same lineage — `ss2_alignment(seq1: List[Any], ..., subst_cost: Callable)`, which reaches
into Python object space in the inner loop of an O(mn) dynamic program. The two are the same
problem solved twice, five years apart, and the difference is exactly what ADRs 0001-0004
record.

`ScoringMatrix` belongs to `align`, not `dataseq`. It appears here because it is the reason
the encoder's integer codes must be **dense and low**: the matrix is `size × size`, so a
sparse or structured code space costs quadratically. That independently corroborates the
goal-2 decision that `dataseq` owns the dense vocabulary index and demotes `PitchCode`-style
structured codes to a pre-vocabulary canonicaliser (`ANALYSIS.md` §1) — reached there from
`hmm`'s V × V transition matrices, and here from `align`'s scoring matrix.

### 4.2 The private mapping is already public API

`ScoringMatrix.from_dict` and `score_symbols` both index `alphabet._sym_to_idx` directly
(`_types.py:115-116, 148-150`). The underscore is nominal: a sibling class in the same
package is a consumer, and in the family the consumer would be a *different distribution*
(`pfsmgraph-align` reaching into `pfsmgraph-dataseq`).

Goal 6 has to publish this deliberately — some public, documented, stable accessor from
symbol to code that `align` and `hmm` can rely on across a package boundary — rather than
inherit a private name that three packages reach into. Naming it is goal 6's business; that
it must be named is settled here.

## 5. The proposed reconciliation

Offered for goal 6, not decided. The shape that follows from the above:

1. **Take `Alphabet`'s structure as the base of the merged encoder** — frozen dataclass,
   `tuple` of symbols, both mappings derived once at construction. It is the only one of the
   four that is a vocabulary object, and it already satisfies four gaps the others leave open.
2. **Renumber to ADR 0011** (§2), adding `UNK` and `MSK`, and moving strictness from total
   to strict-by-default-with-opt-in.
3. **Make the reserved block unreachable** — `ClassVar[int]`, plus a test asserting the
   constructor rejects any attempt to relocate it (§3.1).
4. **Make `decode` total** over the whole code range, with the reserved-code rendering
   decided explicitly rather than by omission (§3.2).
5. **Add what `Alphabet` lacks and the containers need**: first-appearance ordering as the
   assignment rule (`hmm-lush` §2.1), persistence with its own escaping rule
   (`hmm-lush` §2.3), and a frequency reordering offered explicitly
   (`../py-rudimentary/COMPARISON.md` §2.3).
6. **Publish the symbol → code accessor** as cross-distribution API (§4.2).

Points 1-4 are `Alphabet` reconciled; 5-6 are where the containers and the ADRs supply what
it does not have.

## 6. What this hands forward

| Goal | Inherits |
|---|---|
| 5 (land the container) | §4.1 — the container's codes must stay dense and low, because `align` and `hmm` both build `size × size` matrices over them |
| 6 (encoder API) | §5 in full; §3.1 and §3.2 as defects to fix rather than inherit; §4.2 as an API surface that must be named |
| 7 (promote ADR 0010) | §1 — the `Alphabet` reconciliation the ADR requires is tabulated here; §4.1 corroborates the dense-index decision from `align`'s side, independently of `hmm`'s |
| `align` 0.1.0 (later) | `ScoringMatrix`, `AlignmentResult`, `_backends.py` and the algorithms behind the commented Phase 3 block in `.gitignore` — none of it `dataseq`'s business |
