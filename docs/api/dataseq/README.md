# `pfsmgraph.dataseq`

The data sequence container and the symbol↔code encoder. This is the base layer of the
`pfsmgraph` family: every other member depends on it, and it depends on none of them.

- **Encoder** — [`encoder.md`](encoder.md): `Vocabulary`, `SymbolTable`, the reserved
  block, `CODE_DTYPE`.
- **Container** — [`container.md`](container.md): `SequenceRecord`, `SequenceDataset`,
  `pad_collate`.

Its only runtime dependency is `numpy`. It imports neither `torch` nor `pandas`, and a
test asserts that in a subprocess rather than trusting it.

## The one example worth reading first

Everything this package is careful about shows up in a single batch:

```python
from pfsmgraph.dataseq import SequenceDataset, SymbolTable, pad_collate

sequences = [["D3", "F3", "G3"], ["F3", "E3"]]
vocab = SymbolTable.from_sequences(sequences)
ds = SequenceDataset.from_symbols(sequences, vocab, labels=["V01", "V02"])

batch = pad_collate([ds[0], ds[1]])
```

```python
>>> batch["codes"]
array([[6, 7, 8],
       [7, 9, 0]], dtype=int32)
>>> batch["lengths"]
array([3, 2])
>>> batch["mask"]
array([[ True,  True,  True],
       [ True,  True, False]])
```

The trailing `0` in the second row is `PAD`. Read the `codes` array alone and there is no
way to tell that cell from a real symbol — which is exactly why `PAD` is reserved at 0 and
why the mask is returned unconditionally alongside it. The two arrays are produced
together, by one function, and there is no call that gives you the first without the
second.

## The contracts

These hold for every caller and are not configurable. Each is stated where it belongs and
repeated here because together they are the package.

1. **The reserved block is fixed.** `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5;
   user symbols from 6. There is no constructor parameter, class attribute, or setting
   anywhere that relocates it. `PAD` must be 0 because PyTorch's zero-fill idioms
   (`pad_sequence`, a `torch.zeros()` buffer) write zeros into absent positions, and any
   other value would make "absent" silently mean a real symbol.
   ([ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md))
2. **Encoding is strict by default.** An unseen symbol raises. `on_unknown="unk"` is the
   explicit, per-call opt-in that maps it to `UNK` instead. There is no third policy and
   no way to make leniency the default.
3. **Decoding is total** over `range(vocab.size)`, reserved codes included. The array most
   likely to be decoded is a padded batch, which is full of reserved codes by
   construction, so a decoder that handled only user codes would fail on the commonest
   input.
4. **A record carries its true length and never holds padding.** `SequenceRecord.length`
   and `len(record.codes)` cannot disagree, because there is only one of them.
5. **`pad_collate` is the single place padding is introduced**, and it always returns the
   mask with it. Padding is a property of a batch, never of the container.

Encoding happens once, at the boundary — `SequenceDataset.from_symbols` — and everything
downstream is integer-only. That is what will make the Cython and CUDA backends in the
other packages mechanical to write: they never touch a string type.
([ADR 0001](../../design/adr/0001-encode-at-the-boundary.md))

## The public surface

`pfsmgraph.dataseq.__all__` is exactly 15 names. Anything underscore-prefixed — the
`_vocabulary`, `_record`, `_dataset`, `_collate`, `_reserved` modules and every attribute
beginning with `_` — is private, out of contract, and may change without notice. Import
from the package, never from a module inside it.

The 15 are not 15 peers. They are three groups:

**Five API objects** — the surface you program against.

| Name | Kind | Documented in |
|---|---|---|
| `Vocabulary` | protocol | [encoder.md](encoder.md) |
| `SymbolTable` | class | [encoder.md](encoder.md) |
| `SequenceRecord` | class | [container.md](container.md) |
| `SequenceDataset` | class | [container.md](container.md) |
| `pad_collate` | function | [container.md](container.md) |

**Nine reserved-block constants** — the fixed block, plus the two derived views of it.

| Name | Value |
|---|---|
| `PAD` | `0` |
| `UNK` | `1` |
| `BOS` | `2` |
| `EOS` | `3` |
| `GAP` | `4` |
| `MSK` | `5` |
| `USER_BASE` | `6` |
| `RESERVED_SYMBOLS` | `('PAD', 'UNK', 'BOS', 'EOS', 'GAP', 'MSK')` |
| `RESERVED_CODES` | `{'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'GAP': 4, 'MSK': 5}` |

**One dtype** — `CODE_DTYPE`, which is `numpy.int32`. Every code array uses it. Fixing it
here is what keeps the downstream DP packages free of a conversion.

## Two boundaries you will otherwise trip on

Both are deliberate, and both look like bugs the first time you meet them.

### `isinstance(ds, torch.utils.data.Dataset)` is `False`

`SequenceDataset` does not subclass it, and does not import torch at all. It does not need
to: `torch.utils.data.Dataset` is duck-typed, and `__len__` plus `__getitem__` is the whole
contract for a map-style dataset. A stock `DataLoader` accepts a `SequenceDataset`
directly.

```python
from torch.utils.data import DataLoader
from pfsmgraph.dataseq import pad_collate

loader = DataLoader(ds, batch_size=32, collate_fn=pad_collate)
```

Keeping torch out is what lets `align`, `hseg`, and `hmm` — none of which have anything to
do with deep learning — depend on this package. If you have code that gates on
`isinstance`, gate on the two methods instead.

### `default_collate` raises `TypeError` on these items

The reason `pad_collate` ships. Items are ragged by construction, and the stock collate
function cannot stack arrays of differing lengths, nor does it know what `SequenceRecord`
is. Omitting `collate_fn` at any batch size above 1 fails, immediately and loudly:

```python
loader = DataLoader(ds, batch_size=32)   # TypeError from default_collate
```

This failure mode is the good one. The alternative design — a container that padded on
ingestion — would have produced a batch that *worked* and was silently wrong at every
padded position. See [`container.md`](container.md#pad_collate) for the returned arrays.

## Related records

- [ADR 0009](../../design/adr/0009-dataseq-as-the-base-layer.md) — why `dataseq` is the base layer.
- [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md) — the merge this package came out of, and the settled encoder API.
- [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) — the reserved block and strict encoding.
- [ADR 0001](../../design/adr/0001-encode-at-the-boundary.md) — encode at the boundary.
