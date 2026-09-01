# Container

`SequenceRecord`, `SequenceDataset`, and `pad_collate`. See [README.md](README.md) for the
contracts these implement, and [encoder.md](encoder.md) for the vocabulary they encode
against.

Every example on this page runs against the same dataset:

```python
from pfsmgraph.dataseq import SequenceDataset, SymbolTable, pad_collate

sequences = [["D3", "F3", "G3"], ["F3", "E3"]]
vocab = SymbolTable.from_sequences(sequences)
ds = SequenceDataset.from_symbols(sequences, vocab, labels=["V01", "V02"])
```

```python
>>> ds
SequenceDataset(n_sequences=2, vocabulary_size=10)
```

## `SequenceRecord`

One sequence: its codes, its true length, and an optional label. A frozen dataclass.

```python
SequenceRecord(codes: np.ndarray, label: str | None = None)
```

```python
>>> ds[1]
SequenceRecord(length=2, label='V02')
>>> ds[1].codes
array([7, 9], dtype=int32)
>>> ds[1].length
2
>>> len(ds[1])
2
```

### It never holds padding

`length` is always the **true** length. It is not a stored field — it is
`codes.shape[0]` — so there is no second number that could disagree with the first.

This is structural rather than conventional, and deliberately so. The Lush original
packed ragged sequences into a dense matrix whose short rows were tail-filled with a code
that meant a real symbol, and stayed correct only because every reader remembered to
consult a parallel lengths array. Miss one reader and the bug is silent. Here, padding
exists only in a collated batch, alongside the mask that identifies it.

### `codes` is read-only

```python
>>> ds[1].codes[0] = 99
ValueError: assignment destination is read-only
```

Set at construction, and the array is **copied before being frozen** — `np.asarray` may
have returned the caller's own array, and making that read-only would mutate an object the
record does not own. So passing an array in and then modifying your copy does not
retroactively change the record.

`codes` must be 1-D; anything else raises `ValueError` at construction, naming the shape
it got.

## `SequenceDataset`

A read-only collection of encoded sequences over one shared vocabulary.

### `from_symbols` — the encoding boundary

```python
SequenceDataset.from_symbols(
    sequences: Iterable[Sequence[str]],
    vocabulary: Vocabulary,
    labels: Sequence[str] | None = None,
) -> SequenceDataset
```

This is where encoding happens, once, for the whole dataset; everything downstream is
integer-only. Unseen symbols raise, because the vocabulary is strict — `from_symbols` does
not expose an `on_unknown` passthrough, so a dataset built this way cannot silently
contain `UNK`. If you want lenient encoding, call `vocabulary.encode(..., on_unknown="unk")`
yourself and construct `SequenceRecord`s from the result.

`labels` is optional; when given it must be the same length as `sequences`, and the
mismatch error says both numbers.

### `__init__`

```python
SequenceDataset(records: Iterable[SequenceRecord], vocabulary: Vocabulary)
```

The lower-level constructor, for records you built yourself. Every element is type-checked
and a non-record raises `TypeError` naming the offending index — worth it because the
alternative failure surfaces much later, inside a collate.

### This is not a corpus loader

There is no directory walking, no file format, and no pandas. Ingestion is the part that
differs most between projects: two of the imported implementations made the loader a
constructor, and neither could reuse the other's. `from_symbols` is the seam. Produce
sequences of symbols however you like — parse MIDI, read a CSV, query a database — and
hand them over.

### Accessors

```python
>>> len(ds)
2
>>> ds.lengths          # true lengths, in item order
(3, 2)
>>> ds.vocabulary
SymbolTable(size=10, user_symbols=4)
>>> ds.decode(1)        # one item, back to symbols
['F3', 'E3']
```

`ds[i]` returns a `SequenceRecord`; the dataset is also iterable, yielding records in
order.

### Torch compatibility

`SequenceDataset` does **not** subclass `torch.utils.data.Dataset` and does not import
torch. `__len__` and `__getitem__` are the entire map-style contract, and it is duck-typed,
so a stock `DataLoader` accepts this directly. See
[README.md](README.md#isinstanceds-torchutilsdatadataset-is-false) for what that means
for `isinstance` checks.

## `pad_collate`

```python
pad_collate(batch: Sequence[SequenceRecord]) -> dict[str, np.ndarray]
```

Pads a batch of ragged records to the batch maximum. Pass it as `collate_fn`; no
subclassing is needed on either side.

```python
from torch.utils.data import DataLoader

loader = DataLoader(ds, batch_size=32, collate_fn=pad_collate)
```

Returns a dict of three arrays:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `codes` | `(B, L)` | `int32` | Codes, right-padded with `PAD` (0) |
| `lengths` | `(B,)` | `int64` | True length of each item |
| `mask` | `(B, L)` | `bool` | `True` at real positions, `False` at padding |

`L` is the maximum length *in this batch*, not a global maximum, so batches differ in
width and no sequence is ever truncated.

```python
>>> batch = pad_collate([ds[0], ds[1]])
>>> batch["codes"]
array([[6, 7, 8],
       [7, 9, 0]], dtype=int32)
>>> batch["lengths"]
array([3, 2])
>>> batch["mask"]
array([[ True,  True,  True],
       [ True,  True, False]])
```

### The mask is not optional

It is returned unconditionally rather than offered behind a flag. Padding that cannot be
distinguished from data is the exact failure the reserved block exists to prevent, and
emitting the padded array without its mask would reintroduce that failure one layer up.
There is no call that gives you `codes` without `mask`.

### Arrays, not tensors

The base layer does not depend on torch, so these are numpy arrays.
`torch.from_numpy(batch["codes"])` converts each one in a single call and without a copy.

### An empty batch raises

```python
>>> pad_collate([])
ValueError: cannot collate an empty batch
```

There is no meaningful width for a batch of nothing. Raising beats returning a `(0, 0)`
array that would propagate silently into a model.
