# pfsmgraph-dataseq

Data sequence container and symbol-to-code encoder — the base layer of the
[`pfsmgraph`](https://github.com/PanosMavromatis/pfsmgraph) family. It depends on nothing
else in the family, and everything else in the family depends on it.

It imports neither torch nor pandas. The container is compatible with
`torch.utils.data.Dataset` by being duck-typed rather than by subclassing it, so a stock
`DataLoader` works against a package that never imports torch; ingestion belongs to the
caller. Its one runtime dependency is numpy.

```bash
pip install pfsmgraph-dataseq
```

## Encoding is strict, and that is the point

Symbols are multi-character strings — words, note names, phoneme labels — not characters.
`SymbolTable` maps them to `int32` codes at the boundary, so every inner loop is
integer-only and a Cython or CUDA backend never touches a string type.

```python
from pfsmgraph.dataseq import SymbolTable

sequences = [["D3", "F3", "G3"], ["F3", "E3"]]
vocab = SymbolTable.from_sequences(sequences)
```

```python
>>> vocab.encode(["D3", "F3", "G3"])
array([6, 7, 8], dtype=int32)
```

An unseen symbol raises rather than silently becoming padding or a `NaN`, and the error
names the symbol, its position, and the way out:

```python
>>> vocab.encode(["D3", "Bb4"])
KeyError: '\'Bb4\' at position 1 is not in this vocabulary (4 user symbols). Encoding is strict; pass on_unknown="unk" to map unseen symbols to UNK.'
```

The fallback is spelled per call, not per table, so one mapping serves curated training
data and uncurated inference:

```python
>>> vocab.encode(["D3", "Bb4"], on_unknown="unk")
array([6, 1], dtype=int32)
```

Decoding is **total** over `range(size)`, reserved codes included — a padded batch is the
array most likely to be decoded:

```python
>>> vocab.decode([0, 6, 7])
['PAD', 'D3', 'F3']
```

## The reserved block is fixed

`PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6. It is not
configurable, by design.

```python
>>> from pfsmgraph.dataseq import RESERVED_SYMBOLS, USER_BASE
>>> RESERVED_SYMBOLS
('PAD', 'UNK', 'BOS', 'EOS', 'GAP', 'MSK')
>>> USER_BASE
6
```

`PAD` must be 0 because PyTorch's zero-fill idioms — `pad_sequence`, a `torch.zeros()`
buffer — would otherwise silently mean something other than "absent".

## Records are ragged; padding lives in the collate function

A `SequenceRecord` carries its true length and never padding. Padding is introduced only
by `pad_collate`, and is always returned with the mask that says where it is.

```python
from pfsmgraph.dataseq import SequenceDataset, pad_collate

ds = SequenceDataset.from_symbols(sequences, vocab, labels=["V01", "V02"])
```

```python
>>> ds[0]
SequenceRecord(length=3, label='V01')
>>> pad_collate([ds[0], ds[1]])
{'codes': array([[6, 7, 8],
       [7, 9, 0]], dtype=int32), 'lengths': array([3, 2]), 'mask': array([[ True,  True,  True],
       [ True,  True, False]])}
```

Stock `DataLoader`, no subclassing:

```python
# DataLoader(ds, batch_size=32, collate_fn=pad_collate)
```

## Documentation

The contracts a caller may rely on — the invariants, why they hold, and the seams between
this distribution and the rest of the family — are in
[`docs/api/dataseq/`](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/api/dataseq/README.md).
Docstrings are normative for signatures; that directory is normative for contracts. Every
code block in both, this README included, is executed and its output pasted from the run.

The decisions behind them are recorded in
[`docs/design/adr/`](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/README.md).

## License

MIT.
