# Lush against the `dl` base

**Subgoal 4 of goal 3**, `docs/plan/feat-dataseq-merge/TODO.md`. The account of the Lush
original in its own terms is [`ACCOUNT.md`](ACCOUNT.md); the base is analysed in
[`../dl/ANALYSIS.md`](../dl/ANALYSIS.md). This document compares them and names what the
base must be overridden on.

**The constraint this document works under.** The ADRs outrank all three implementations,
the `dl` base included. So "override" here means only the points the ADRs leave *open*. A
point an Accepted ADR settles is recorded below as a divergence of the original —
information about what was tried, never a candidate to adopt. The two categories are kept
in separate sections (§2 and §3) so they cannot be confused when this feeds ADR 0010.

---

## 1. The axis: both are dense vocabulary indices

`ANALYSIS.md` §5 set the question to ask of each further implementation: *is its encoder a
vocabulary or a codec, and where does its reserved block start?*

**Lush is a pure dense vocabulary index.** `format-sds` assigns the next free integer to
each novel symbol and stores the mapping in a table; the code means nothing except
"position in this corpus's alphabet". There is no arithmetic structure anywhere, and no
second kind of encoder to conflate it with.

That makes the Lush implementation **independent corroboration of the decision taken on
2026-08-31**, arrived at fifteen years earlier and under no influence from it: `dataseq`
owns the dense vocabulary index, and structured codecs like `PitchCode` are demoted to
canonicalisers upstream of vocabulary assignment. The `dl` base is the *only* one of the
three that fuses a codec into the vocabulary, and the fusion is legible there as
`PitchCode`'s own design succeeding at a job — one model's embedding table — that is not
this one.

Corroboration is not proof, and it is worth being precise about what it is worth: Lush had
no `align`, no shared namespace, and no second consumer, so it was never *tested* against
the pressure that makes the distinction matter. What it shows is that the plain design is
sufficient for a working HMM trainer over a real corpus, which is the weaker claim and the
one that was actually in doubt.

## 2. Where the base must be overridden

Five points. Each is checked against the ADRs first: all five concern matters no Accepted
ADR settles, so all five are live.

### 2.1 Vocabulary ordering must be first-appearance, not set iteration

| | `dl` base | Lush |
|---|---|---|
| Registration | iterate `symbol_set - self.alphabet`, a **`set`** (`data.py:133-134`) | append on first sight, in document order |
| Determinism | not guaranteed once codes are counter-assigned | total |

`ANALYSIS.md` §1a identifies this as a latent reproducibility bug: today `PitchCode.encode`
is a pure function of the string so iteration order cannot matter, but **under a dense
index a counter assigns codes in insertion order, and CPython randomises `str` hashing per
process.** The same corpus would yield different codes on every run, silently invalidating
any checkpoint or persisted vocabulary.

The analysis proposed first-appearance order on the argument that it is stable under corpus
growth where sorted order is not. **Lush supplies the prior art, and the translation
supplies evidence rather than argument:** re-deriving both tracked corpora from their own
`_raw_data` reproduces the 2009 `_alphabet` exactly, code for code. A sixteen-year gap
between the two runs is a stronger determinism test than anything we would have written.

**Override:** first-appearance order, and the merged implementation must not register
symbols by iterating an unordered collection.

### 2.2 The container must carry per-sequence true lengths

| | `dl` base | Lush |
|---|---|---|
| Ragged storage | pad to corpus maximum (`data.py:241-243`) | pad to `seq_size_max` |
| True lengths | **not stored anywhere** | `seq_sizes`, a first-class slot |
| Who consults them | nothing | every reader, without exception |

Both implementations pad to a global maximum and neither produces a mask
(`ANALYSIS.md` §3.6). But they fail differently, and the difference is the override.

`dl` discards the information: once a document is padded, its true length is not
recoverable from the container, so a mask cannot be *derived* even by a caller willing to
do the work. Lush keeps it, and `save`, `fprop-all` and the translation's `decode` all loop
to the row's own size rather than the matrix width. That discipline is the only reason the
original is correct at all, since its fill value is `begin` — a real symbol
(§3.1 below).

**Override:** per-sequence true lengths are part of the container's state, not an artefact
of how it was built. Whether the *mask* lives in `dataseq` or in `dl` stays open where
`ANALYSIS.md` §3.6 left it — but the base must at minimum make one derivable, and today it
does not.

### 2.3 The vocabulary must be persistable, and persistence must own its escaping

`ANALYSIS.md` §3.4: `dl` never serializes `encoder_map` / `decoder_map`, so a checkpoint
trained on those codes is uninterpretable without the exact corpus and load order that
produced them.

Lush persists the alphabet as `_alphabet`, a two-column text file, and reconstructs it on
every load. That is prior art for the gap — and it also carries the lesson about *how*,
which is the more useful half. The original has two writers that disagree: `format-sds`
prints the Lisp symbol with `%l` and so emits the multiple-escape delimiters `|…|`, while
`dsource-seq save` prints `(ptr-str …)` — a bare string whose symbolhood `symbol->string`
discarded at load — with `%s`. The asymmetry is a downstream cost of the `-gptr-` alphabet
slot, which exists so the compiled methods can cross the DH boundary (`ACCOUNT.md` §4).

The general form, and the reason it survives translation into any language: **once a name
has been unwrapped into a plain string, the information about whether it needed quoting is
gone.** A vocabulary that persists strings rather than symbols cannot borrow a reader's
escaping and must own the rule itself. The translation does exactly that — one writer,
inspecting the name — and round-trips `set11a_dInt`'s `E. -2nd` and `H _` intact.

**Override:** the vocabulary is a persistable artefact with a defined format, and the
format owns its own quoting rule. One regression test: round-trip a symbol containing a
space.

### 2.4 "Frozen" must be an explicit state, not a structural accident

`ANALYSIS.md` §3.2: `dl`'s vocabulary grows as documents are read with no way to freeze it,
so train/test splits are inexpressible — the test set silently extends the alphabet the
model trained on.

Lush has the property `dl` lacks, but gets it by *not having an encoder at all*: encoding
happens once inside `format-sds`, at corpus-build time, in a process that then exits, and a
loaded `dsource-seq` can decode but cannot encode. This is the account's central structural
finding (`ACCOUNT.md` §1), and it explains something that looked like a design choice and
is not — **the strict-vs-`UNK` question never arises in the original because there is no
runtime encoding path to be strict in.** An absence, not a lenient default.

So neither implementation offers a model to copy: `dl` has an encoder that cannot freeze,
Lush has a freeze achieved by having no encoder. What Lush contributes is the demonstration
that a frozen vocabulary is *sufficient* for a real training workflow, and that the two
phases — build the vocabulary, then use it — are genuinely separable rather than
intertwined by necessity.

**Override:** frozen is an explicit, queryable state of the vocabulary object, and both
phases are expressible against one API. The `dl` base's growing-only behaviour is replaced,
not extended.

### 2.5 Decoding must be a live path

`ANALYSIS.md` §3.3: `dl` builds `decoder_map` and keeps it in sync, but **never reads it
anywhere**. ADR 0001 requires ints → strings at exit; that half does not exist.

Lush's decode path is live and exercised — `view-string` renders symbols through
`(ptr-str (alphabet (symbol-data pos-i)))`, and it is how a trained model's output was
actually read. Nothing dramatic follows, but it is the difference between a maintained path
and dead code that will be wrong the first time anyone calls it.

**Override:** decode is part of the tested surface. `dl` contributes the map; Lush
contributes the evidence that it needs to be used.

## 3. Divergences that are *not* candidates

Recorded because the comparison would be incomplete without them, and flagged so they
cannot be mistaken for §2 material.

### 3.1 The reserved block

| Implementation | Reserved | User symbols from |
|---|---|---|
| `dl` (MelodyHPO) | `PAD` 0, `BOS` 1, `EOS` 2 | **3** |
| Lush | `begin` 0, `end` 1 | **2** |
| Proof-of-concept (`align`) | padding/BOS/EOS low, gap next | **4** |
| **ADR 0011 — authoritative** | `PAD` 0, `UNK` 1, `BOS` 2, `EOS` 3, `GAP` 4, `MSK` 5 | **6** |

Three implementations, three offsets, and **not one of the three has a `GAP` code at all**.
There was never a majority to defer to: ADR 0011 supplies something all three lack rather
than breaking a tie between them. `GAP` is the symbol `align` exists to produce, and no
single-purpose implementation had any reason to invent it.

Lush's `0 = begin` is the sharpest case, because it collides with ADR 0011 destructively
rather than merely differing. `load` allocates a zero-filled `size × seq_size_max` matrix
and writes only each row's real cells, so **every short row's tail holds `begin` — a real
symbol, indistinguishable from data by inspection.** In `set01z0_100.sds` that is 1418 of
2000 cells, 71%, and 35 of the 100 sequences are empty (`begin`/`end` and nothing between).

The original is correct because every reader consults `seq_sizes`. But this is ADR 0011's
`PAD` = 0 rationale observed in working code rather than argued from first principles, and
it is worth stating what makes it a *good* illustration: the padding is not merely
ambiguous, it is **gratuitous**. `hmm-trainer.lsh:66` is the only caller of `fprop-all` and
nothing else reads `seq-data`, so the matrix is a staging buffer — ragged data unpacked
into a rectangle whose sole consumer immediately repacks it ragged. The hazard was taken on
without even the performance argument that usually excuses it.

**Not a candidate.** It is evidence about the cost of the alternative, and it is already
recorded as such.

### 3.2 Strictness

Settled by ADR 0011: strict by default, `UNK` fallback an explicit opt-in. `dl`'s accidental
behaviour is a silent `NaN` from `Series.map` on an unmapped symbol — which also promotes
the whole Series to `float64`, so integer codes quietly become floats (`ANALYSIS.md` §3.2).
Wrong default, silently. Lush offers nothing, per §2.4.

Only the **spelling** of the switch is open, and ADR 0011 says so in its own text, deferring
it to the ADR 0010 reconciliation while holding the semantics settled. That is goal 6's
second criterion, and neither implementation informs it.

## 4. The error both made, independently

Worth separating from the point-by-point comparison, because it is the one finding that
neither implementation's analysis could have produced alone.

**Both bake a consumer's view into the container.** `dl`'s `DatasetSW` and `DatasetDoc`
emit `(input_ids, target_ids)` shifted one position — next-token language modelling, a
training objective (`ANALYSIS.md` §2.2). Lush's `fprop-all` emits one flat concatenated
stream with `begin`/`end` inline as the only sequence delimiters, which is what
Baum-Welch wants and nothing else.

Two implementations, two unrelated consumers, the same category error. That converts
`ANALYSIS.md` §2.2's proposal from a judgement call into something better supported: the
separation is not a preference for tidiness but a correction of a mistake made twice
independently, by people solving different problems, each of whom would have had to undo it
to serve the other's consumer.

**For goal 5:** `dataseq` yields sequences. The windowing and `(input, target)` pairing move
to `pfsmgraph.dl`; the flat concatenated stream with inline boundary markers moves to
`pfsmgraph.hmm` as a view. Neither is a container property.

## 5. What Lush contributes that `dl` has no analogue for

### 5.1 `seq-state`: a sequence and its annotation, in parallel arrays

```
symbol_data    size        the observed sequence
path_states    size + 1    the state path over it
path_entropy   size + 1    per-position uncertainty
```

The `+1` is exact rather than defensive: a path through an HMM that emits N symbols visits
N+1 states, so the annotation arrays are indexed by the *gaps between* symbols while the
symbols are indexed by themselves. `view-string` renders the offset directly, states
bracketed beneath symbols.

`dl` has no analogue — nothing in the base pairs a sequence with a model's reading of it.

**This is not a `dataseq` feature**, and the scoping decision recorded in the plan is why:
the `hmm` side matters only insofar as it shows how the `hmm` model implementation must be
modified. `seq-state` is an `hmm` object. What `dataseq` owes it is a per-sequence view an
annotation can be aligned against — which follows from §2.2's true-lengths requirement, and
is the concrete reason that requirement is not merely tidiness.

**For the eventual `hmm` translation:** `seq-state` is the seam. `pfsmgraph.hmm` will carry
it, consuming `dataseq` sequences and owning `path_states` / `path_entropy` itself.

### 5.2 Positive evidence for `dependencies = []`

`ANALYSIS.md` §4 closed the `DEFERRED.md` dependency question in the negative — torch leaves
with the dataset views, pandas with ingestion, and neither the container nor a dense
vocabulary needs a third-party package — but by argument.

The translation is evidence: it is **stdlib-only**, and it loads, rebuilds, round-trips and
flattens both tracked corpora, including the 1449-symbol single-sequence one. That is not
proof for the merged package, which will do more, but it does close off the possibility
that something in the container's job quietly requires numpy.

## 6. What neither implementation supplies

The merge inherits none of these from anywhere, and goals 5 and 6 must build them:

- **`GAP`, `UNK`, `MSK`** — absent from both. `GAP` is the load-bearing one.
- **A vocabulary object.** `dl` has a bare `set[str]` beside two dicts (`ANALYSIS.md` §3.5);
  Lush has a pointer array beside a separately-stored size. Neither has anywhere to put the
  reserved block, the strictness switch, ordering, persistence, or the invariant that the
  parallel structures stay in sync. This is what ADR 0010 is `Proposed` pending.
- **Validation.** Lush's intended consistency check is an unimplemented comment in `load`;
  `dl` validates nothing either. Both trust their own metadata — Lush trusts
  `_alphabet_size` and `_size`, and self-indexes from the file so a duplicate silently
  overwrites and a gap silently leaves a zero.
- **Masking.** Neither masks; §2.2 is about making one derivable.
- **Tests for the container.** `dl`'s only test covers `PitchCode`, which attaches to the
  demoted canonicaliser rather than to `dataseq`'s encoder. Lush has none.
- **Sharing one vocabulary across corpora.** Same gap as §2.4 seen from another side.

## 7. What this hands forward

| Goal | Inherits |
|---|---|
| 4 (rudimentary Python) | Ask the same two questions (§1). Its reserved block is the fourth data point; `DEFERRED.md` anticipates user symbols from 4 |
| 5 (land the container) | §2.2 true lengths, §4's separation of consumer views, and `ANALYSIS.md` §2.1–2.6 |
| 6 (encoder API) | §2.1 ordering as a correctness constraint; §2.3 persistence and its quoting rule; §2.4 frozen as explicit state; §2.5 decode on the tested surface |
| 7 (promote ADR 0010) | §1 corroborates the dense-index decision; §4 strengthens the container/view separation; §3 is the divergence record the ADR should carry |
| `hmm` translation (later) | §5.1 — `seq-state` is the seam, and it belongs to `hmm` |
