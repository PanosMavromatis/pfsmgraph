# The `dl` merge base — what carries, what doesn't, what's missing

Analysis of `.scratch/dl/MelodyHPO/melody_hpo/data/`, the base the other two `dataseq`
implementations are merged onto per PRD §3.5. Provenance is in `.scratch/README.md`.

Line references are to the files as imported at MelodyHPO `5f42311`.

## 0. Inventory

| File | Lines | What it is |
|---|---|---|
| `data.py` | 252 | `MiniCorpus` (ingest + encode), `DatasetSW` (sliding window), `DatasetDoc` (whole document) |
| `encoder/music.py` | 23 | `MusicCode`, an ABC: `range` + static `encode`/`decode` |
| `encoder/pitch.py` | 185 | `PitchCode`, a structured pitch codec. 74 lines of docstring, and they are the design rationale |
| `encoder/control.py` | 21 | `_control_sym_encode` = `PAD` 0, `BOS` 1, `EOS` 2; reverse map derived |
| `data/tokenizer.py` | 1 | A comment. Nothing is implemented |
| `tests/unit/test_pitch_encoder.py` | 45 | The only test in the project. Covers `PitchCode` only |

The container itself — `MiniCorpus`, `DatasetSW`, `DatasetDoc` — has no tests.

## 1. The central finding: two kinds of encoder, conflated

This is the finding everything else in the merge depends on, so it goes first.

`MiniCorpus` builds a single `encoder_map: dict[str, int]` (`data.py:84`) from two sources
that are not the same kind of thing:

- **A dense vocabulary index.** `_control_sym_encode` (`control.py:11-15`) assigns
  `PAD` 0, `BOS` 1, `EOS` 2 — positions in an alphabet, meaningful only relative to each other.
- **A stateless structured code.** `PitchCode.encode` (`pitch.py:132`) computes
  `100 × chromatic + diatonic`, so `C4` → 6035 and `Gb4` → 6639. The integer *is* the pitch,
  recoverable by arithmetic with no table at all. `PitchCode.decode` inverts it (`pitch.py:149`)
  without reference to any corpus.

The result is a code space occupied at 0, 1, 2 and then nothing until 1207
(`PitchCode.range`, `pitch.py:101-104`). The `pitch.py` docstring is explicit that this is
deliberate: it reserves 0–1206 for control tokens, notes compatibility with `[PAD]`/`[BOS]`/`[EOS]`
at 0/1/2, and observes that the 13,176 ceiling keeps the whole vocabulary inside `uint16`.

**As a design for one model's embedding table, this is coherent and well argued.** As the base
of this package family it does not survive contact with the dependents:

- `hmm` builds V×V transition matrices. V = 13,177 gives ~1.74 × 10⁸ entries — about 1.4 GB in
  float64 — for a corpus whose real alphabet is a few dozen symbols. Baum-Welch over that is not
  slow, it is infeasible, and topology search by state merge/split makes it worse.
- ADR 0011 fixes user symbols to start at 6. That presumes contiguous dense allocation. A scheme
  whose first user symbol is 1207 cannot satisfy it without the reserved block becoming a
  fiction — it would not *collide*, but "user symbols from 6" would describe nothing.
- Embedding-table width is a `dl` concern. `align` and `hseg` never embed anything.

**Decision taken (2026-08-31):** `dataseq` owns the **dense vocabulary index** — reserved block
0–5, user symbols from 6, per ADR 0011. Structured codecs like `PitchCode` are demoted to a
**symbol canonicaliser** that runs *before* vocabulary assignment: it normalises and validates a
domain symbol, and the vocabulary then assigns that symbol a dense code. Musical structure stays
available where it is useful, without dictating the alphabet the whole family sees.

This makes `MusicCode` (`music.py:11-22`) the wrong abstraction to carry over as-is. Its contract
is `str → int` with a `range`, which is the structured-codec contract, not the vocabulary contract
(`str → int` *relative to a corpus*, plus `int → str`, plus a size, plus persistence).

### 1a. A trap the decision walks straight into

`MiniCorpus` registers new symbols by iterating a **set**:

```python
new_symbol_set = symbol_set - self.alphabet     # data.py:133
for new_symbol in new_symbol_set:               # data.py:134
    new_symbol_code = encoder.encode(new_symbol)
```

Today this is harmless, because `PitchCode.encode` is a pure function of the string — iteration
order cannot affect the result. **Under a dense index it becomes a reproducibility bug**, because
a counter-based vocabulary assigns codes in *insertion order*, and CPython randomises `str` hashing
per process by default. The same corpus would then yield different code assignments on every run,
silently invalidating any checkpoint or serialized vocabulary from a previous one.

Fix when the dense index lands: iterate deterministically (sorted, or first-appearance order in
the document). First-appearance order is the better choice — it is stable under corpus growth in a
way sorted order is not, since adding one symbol re-sorts everything after it.

## 2. `dl`-specific features that may not serve the other packages

Each with a proposal. None of these are defects in MelodyHPO; they are correct choices for a
single-project DL codebase that become wrong at the base of a five-package family.

### 2.1 Hard `torch` dependency in the container

`data.py:17` imports `torch.utils.data` at module scope; both dataset classes subclass
`torch.utils.data.Dataset` (`data.py:154`, `data.py:215`) and emit `torch.long` tensors.

`dataseq` is the base layer — `align`, `hseg`, and `hmm` all depend on it and none of them need
torch. Making it a hard dependency forces every `align` user to install a multi-gigabyte package
to get a sequence container. It also cuts against the `core.md` invariant that "GPU" means two
unrelated things: `torch` for `dl`, `numba-cuda` for the DP packages.

**Proposal.** The container is torch-free. This costs nothing, because **`torch.utils.data.Dataset`
is not an enforced interface** — for a map-style dataset `DataLoader` requires only `__len__` and
`__getitem__`, found by duck typing. Subclassing buys the `+` operator (`ConcatDataset`) and
nothing else. So goal 5's criterion "conforms to `torch.utils.data.Dataset`; a stock `DataLoader`
works without subclassing" is satisfiable with **zero torch imports in `dataseq`** — worth
verifying with an actual `DataLoader` in the test suite rather than asserting.

### 2.2 The training objective is baked into the container

`DatasetSW` and `DatasetDoc` both emit `(input_ids, target_ids)` with the target shifted one
position ahead (`data.py:203-204`, `data.py:242-243`). That is next-token language modelling — a
training objective, not a container property. `align` wants the raw sequence; `hmm` wants
observation sequences; `hseg` wants segments.

**Proposal.** `dataseq` yields sequences. The shift-by-one, the windowing, and the
`(input, target)` pairing move to `pfsmgraph.dl` as views over a `dataseq` container.

### 2.3 Padding written as a literal, never as `PAD`

`[0] * pad_len` appears at `data.py:196`, `197`, `242`, `243`. It agrees with ADR 0011's `PAD` = 0
only by coincidence of a literal — nothing references the reserved block.

This is exactly the hard-coded index assumption `docs/plan/DEFERRED.md` says must be audited as
part of the merge rather than after it. **Proposal:** every padding site references the symbolic
constant; the literal `0` never appears as a code.

### 2.4 pandas as the ingestion substrate

`MiniCorpus` reads TSV via `pd.read_csv` (`data.py:103`), stores `pd.Series` per document
(`data.py:96`), and encodes with `Series.map` (`data.py:141`).

pandas is heavy for a package whose stated job is "container + encoder", and `align`/`hmm`/`hseg`
have no use for it.

**Proposal.** Ingestion is not `dataseq` core. The core accepts sequences of symbols from any
source; a pandas/CSV reader is an optional extra or lives in `dl`.

### 2.5 `MiniCorpus` is a music-corpus loader, not a container

Its constructor takes one dict with `data_dir`, `doc_paths`, `df_name`, `filters`, `encoder`
(`data.py:73-78`). It selects columns, applies a per-column regex mask, and joins surviving cells
with tabs to form one symbol per row (`data.py:124`). `minicorps/…/layer01.py` shows the intended
use — a definition file executed by `runpy.run_path()`, reading `groups.csv` to build `doc_paths`.

That is a domain-specific corpus assembler and belongs downstream. **Proposal:** keep the
*concept* — a named collection of documents sharing one vocabulary — and drop the loading
mechanics from the base.

### 2.6 BOS/EOS wrapping is unconditional

`data.py:127` prepends `BOS` and appends `EOS` to every document, with no way to opt out. For a
language model that is right. For `align`, sentinel symbols inside the sequences being aligned are
an active nuisance — they will be matched, gapped, and scored.

**Proposal.** Wrapping is a caller's choice, defaulting off in the base.

## 3. Essential gaps

Things absent from the merge base that `dataseq` must have. These are the substance of what goals
5 and 6 have to build; the merge does not inherit them from anywhere.

### 3.1 The reserved block is wrong *and* incomplete

`dl` has `PAD` 0, `BOS` 1, `EOS` 2. ADR 0011 requires `PAD` 0, `UNK` 1, `BOS` 2, `EOS` 3,
`GAP` 4, `MSK` 5.

So `BOS` and `EOS` both **shift**, and three symbols are missing outright. **`GAP` matters most**:
it is the symbol `align` exists to produce, and the base layer cannot serve `align` without it.

Note this is a *different* wrong base from the one `DEFERRED.md` anticipates. That entry warns the
proof-of-concept alignment code allocates user symbols from 4; this one allocates from 3. Two
implementations, two offsets, one audit — and the Lush implementation is a third data point still
to come.

### 3.2 No frozen vocabulary, so no inference path

The vocabulary grows as documents are read (`data.py:132-138`) and there is no way to freeze it.
Encoding a held-out document against an existing vocabulary is not expressible, which means
**train/test splits do not work**: the test set would silently extend the alphabet the model was
trained on.

This is also where a latent silent failure lives. `symbols.map(self.encoder_map)` (`data.py:141`)
returns `NaN` for an unmapped symbol rather than raising — and one `NaN` promotes the whole Series
to `float64`, so integer codes quietly become floats. Today it cannot fire, because every symbol is
registered immediately before the map. It fires the moment a frozen vocabulary exists, which is
precisely what goal 6 must introduce.

ADR 0011 requires strict-by-default with `UNK` fallback as explicit opt-in. The base has neither
behaviour, and its accidental one is the wrong default.

### 3.3 Nothing decodes

`decoder_map` is built (`data.py:85`) and kept in sync (`data.py:137`) but **never read** anywhere
in the codebase. ADR 0001's encode-at-the-boundary invariant requires ints → strings at exit; that
half does not exist. `PitchCode.decode` exists and is tested, but that decodes a *structured code*,
not a vocabulary index — a different operation, as §1 sets out.

### 3.4 The vocabulary is not persistable, and not shared

There is no serialization of `encoder_map` / `decoder_map`. A checkpoint trained on these codes is
uninterpretable without the exact corpus and load order that produced them — and per §1a, load
order is not even stable across runs once the index is dense.

The vocabulary is also per-`MiniCorpus`: two corpora built from the same symbols get independent,
unrelated maps. Nothing lets two corpora share one alphabet, which every train/test or
multi-corpus workflow needs.

### 3.5 No vocabulary object at all

`alphabet` is a bare `set[str]` (`data.py:86`) alongside two dicts. There is no object owning the
symbol↔code relation, so there is nowhere to put size, ordering, the reserved block, the strictness
switch, persistence, or the invariant that the three structures stay in sync. Keeping three
parallel containers consistent by hand is a maintenance hazard the merge should close.

This object is what ADR 0010 is `Proposed` pending, and its constructor signature is goal 6's first
criterion.

### 3.6 Padding is emitted but never masked

`DatasetDoc` pads every document to the corpus maximum (`data.py:241-243`) and produces no mask.
The final real target position predicts `PAD`, and loss is taken on padded positions. ADR 0011's
rationale for pinning `PAD` = 0 is precisely that PyTorch zero-fill must unambiguously mean
"absent" — but "absent" is only meaningful if something honours it.

Whether the mask belongs in `dataseq` or in `dl` is open. It is listed here because the base
provides no way to *derive* one either.

### 3.7 No tests for the container

`test_pitch_encoder.py` is thorough about `PitchCode` — round-trips, enharmonics, both range
bounds, invalid input. It is also the whole suite. `MiniCorpus`, `DatasetSW`, and `DatasetDoc` are
untested, including the sliding-window arithmetic and the short-document padding branch
(`data.py:187-200`).

Per ADR 0003 the merged container needs a suite in this repo regardless; nothing testable carries
over except the `PitchCode` cases, and those attach to the demoted canonicaliser rather than to
`dataseq`'s encoder.

## 4. Dependency findings

For the `DEFERRED.md` entry "fix `dataseq`'s third-party runtime dependencies", whose answer is
supposed to be determined by what the merge base actually needs.

MelodyHPO declares `requires-python >= 3.14`, `torch >= 2.10.0`, `pandas >= 3.0.0`
(`pyproject.toml`). pfsmgraph targets `>= 3.10`, so nothing here can be carried across on the
assumption that a 3.14 floor is available.

Under the proposals in §2, **`dataseq`'s current `dependencies = []` is very likely correct**:
torch leaves with the dataset views (§2.1), pandas leaves with ingestion (§2.4), and neither the
container nor a dense vocabulary needs a third-party package. `numpy` may earn its way in later for
the code arrays; nothing in the base requires it today.

That closes the `DEFERRED.md` question in the negative, which is worth stating explicitly, since
"no change needed" is otherwise indistinguishable from "not yet looked at".

## 5. What this hands to the later goals

| Goal | What it inherits from here |
|---|---|
| 3, 4 (Lush, rudimentary) | §1's dense-vs-structured distinction is the axis to compare on. Ask of each: is its encoder a vocabulary or a codec? Where does its reserved block start? |
| 5 (land the container) | §2.1–2.6 are the separations to make. §2.3's literal-`0` audit lands here |
| 6 (encoder API) | §3.1–3.5 are the requirements. §1a is a correctness constraint on the implementation, not a design choice |
| 7 (promote ADR 0010) | §1's decision and its rationale are the ADR's decision section |
| 8 (cleanup) | This file is one of the artefacts the retention decision covers |
