# Encoder

`Vocabulary`, `SymbolTable`, the reserved block, and `CODE_DTYPE`. See
[README.md](README.md) for the contracts these implement.

Every example on this page runs against the same table:

```python
from pfsmgraph.dataseq import SymbolTable

sequences = [["D3", "F3", "G3"], ["F3", "E3"]]
vocab = SymbolTable.from_sequences(sequences)
```

```python
>>> vocab
SymbolTable(size=10, user_symbols=4)
>>> vocab.symbols
('D3', 'F3', 'G3', 'E3')
>>> vocab.size
10
```

Note the ordering: `E3` gets the last code even though it appears in the shorter sequence,
because ordering is **first appearance across the corpus**, and `E3` is first seen after
`D3`, `F3`, and `G3`. And `size` is 10, not 4 — it counts the reserved block, since it is
the number a downstream package needs when it allocates.

## The reserved block

Fixed by [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
and hard-coded as module constants. There is nothing to pass, override, or subclass.

| Symbol | Code | Meaning |
|---|---|---|
| `PAD` | 0 | Absent position in a collated batch. Must be 0. |
| `UNK` | 1 | A symbol not in the vocabulary. Only ever produced on explicit opt-in. |
| `BOS` | 2 | Beginning of sequence. |
| `EOS` | 3 | End of sequence. |
| `GAP` | 4 | An alignment gap, for `pfsmgraph-align`. |
| `MSK` | 5 | A masked position, for masked-prediction training in `pfsmgraph-dl`. |
| — | 6 | `USER_BASE`: the first code available to a user symbol. |

```python
>>> from pfsmgraph.dataseq import RESERVED_CODES, RESERVED_SYMBOLS, USER_BASE
>>> RESERVED_CODES
{'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'GAP': 4, 'MSK': 5}
>>> RESERVED_SYMBOLS
('PAD', 'UNK', 'BOS', 'EOS', 'GAP', 'MSK')
>>> USER_BASE
6
```

`RESERVED_CODES` is derived from `RESERVED_SYMBOLS` rather than written out, so the two
cannot disagree. Index `RESERVED_SYMBOLS` with any code below `USER_BASE`.

The reserved names are also refused as user symbols, so a corpus containing the literal
string `"PAD"` fails at construction rather than colliding silently:

```python
>>> SymbolTable(["D3", "PAD"])
ValueError: 'PAD' is a reserved symbol name and cannot be a user symbol. Reserved: PAD, UNK, BOS, EOS, GAP, MSK
```

## `CODE_DTYPE`

```python
>>> from pfsmgraph.dataseq import CODE_DTYPE
>>> CODE_DTYPE
<class 'numpy.int32'>
```

Every code array in the package uses it. `int32` is what a Cython or CUDA buffer wants, and
codes never approach its range. Fixing it in one place keeps the downstream DP packages
free of a conversion at their entry points.

## `SymbolTable`

A frozen, first-appearance-ordered symbol table. Immutable by construction — the symbols
are a tuple, the lookup maps are built once, and there is no method that adds a symbol.
A table therefore cannot drift after sequences have been encoded against it.

### Constructors

```python
SymbolTable(symbols: Iterable[str])
SymbolTable.from_sequences(sequences: Iterable[Sequence[str]])
```

`SymbolTable(symbols)` takes a flat iterable of symbols; duplicates are collapsed, keeping
first appearance. `from_sequences` takes a corpus — an iterable of sequences — and
flattens it. Use the second for a dataset and the first when you already have the symbol
list.

The name is deliberate. `Alphabet`, which the proof-of-concept used, implies single
characters; the symbols in this family are words (`"D3"`, `"V01"`), and one of them is
routinely several characters long.

### Ordering is a correctness constraint

First appearance, not insertion into a set and not frequency. Both alternatives were
found in the imported implementations and both are rejected:

- Iterating a `set` to assign codes is a **live reproducibility bug** — CPython randomises
  string hashing per process, so the same corpus produces different codes on different
  runs, and a model checkpoint stops matching its own vocabulary.
- Ordering by frequency is deterministic, but makes every code a function of the whole
  corpus: adding one file renumbers the alphabet. A frequency-ordered constructor is
  deferred rather than refused — see `docs/plan/DEFERRED.md`, trigger "a corpus large
  enough for code locality to matter" — and if it lands it will be a separate classmethod,
  never a change to this default.

### `size`, `symbols`, `__len__`

```python
>>> vocab.size          # reserved block included
10
>>> len(vocab)          # the same number
10
>>> vocab.symbols       # user symbols only, in code order
('D3', 'F3', 'G3', 'E3')
```

`size` is the allocation size a consumer needs; `symbols` is the user portion. They differ
by `USER_BASE`, always.

### `code(symbol)`

```python
>>> vocab.code("F3")
7
>>> vocab.code("ZZ")
KeyError: "'ZZ' is not in this vocabulary (4 user symbols). Encoding is strict."
```

Single-symbol lookup, for callers that want one code without building a sequence.

### `sym_to_code`

The symbol→code mapping, as a read-only view.

```python
>>> dict(vocab.sym_to_code)
{'D3': 6, 'F3': 7, 'G3': 8, 'E3': 9}
>>> vocab.sym_to_code["X"] = 9
TypeError: 'mappingproxy' object does not support item assignment
```

**This is public API across a distribution boundary, not a convenience.**
`pfsmgraph-align` builds an `(size, size)` scoring matrix from the whole mapping when a
matrix is constructed, and it must not have to reach into a private attribute of another
distribution to do it. The proof-of-concept did exactly that — its scoring matrix indexed
`alphabet._sym_to_idx` — which made the mapping cross-package API in everything but name.

It is a `MappingProxyType`, which is a **live view rather than a copy**. That is O(1) per
access where a fresh `dict` would be O(size), a difference that turns quadratic the moment
a consumer calls it in a loop. The view is only safe because there is no method that adds
a symbol: for this class, "live" and "immutable" are the same thing.

### `encode(symbols, on_unknown="raise")`

Maps symbols to a 1-D `int32` array. Strict by default.

```python
>>> vocab.encode(["D3", "E3"])
array([6, 9], dtype=int32)
```

An unseen symbol raises, and the error names the symbol, its position, and the way out:

```python
>>> vocab.encode(["D3", "ZZ"])
KeyError: '\'ZZ\' at position 1 is not in this vocabulary (4 user symbols). Encoding is strict; pass on_unknown="unk" to map unseen symbols to UNK.'
```

The opt-in is spelled per call:

```python
>>> vocab.encode(["D3", "ZZ"], on_unknown="unk")
array([6, 1], dtype=int32)
```

`1` is `UNK`, and it decodes:

```python
>>> vocab.decode(vocab.encode(["D3", "ZZ"], on_unknown="unk"))
['D3', 'UNK']
```

**Per call, rather than per table**, so that one mapping serves both curated training data
(where an unseen symbol is a data bug you want to hear about) and uncurated inference
(where it is expected). The alternative — a `strict=` flag on the constructor — would have
forced two tables over the same symbols, which is two things that can disagree.

There is no third policy, and no way to make leniency the default. The direction of this
switch is fixed by ADR 0011; only its spelling was ever open.

> **Reading those `KeyError` messages.** The outer quotes in the tracebacks above are not a
> typo, and the inner `\'ZZ\'` is not double-escaping on this page. `KeyError` is unique
> among the builtins in defining `str(e)` as `repr(e.args[0])`, so a message that itself
> contains quotes comes back with them escaped. It matters when you handle the error:
> **`e.args[0]` is the message; `str(e)` is a repr of it.** Matching on `str(e)` will not
> find a substring you can plainly see in the traceback.

**The policy is validated before the loop, not at the first unknown symbol:**

```python
>>> vocab.encode([], on_unknown="UNK")
ValueError: on_unknown must be 'raise' or 'unk', not 'UNK'
```

Note the empty input — there is nothing to look up, so this error can only come from an
up-front check. That matters. Validating lazily would let a misspelled `on_unknown="UNK"`
behave exactly like the default for as long as every symbol happened to be known, passing
on clean test data and changing behaviour the first time production saw an unseen symbol.
For an encoder whose entire job is handling unseen symbols, that is precisely the wrong
moment to discover a typo.

### `decode(codes)`

Total over `range(size)`, reserved codes included:

```python
>>> vocab.decode([0, 1, 6, 7])
['PAD', 'UNK', 'D3', 'F3']
```

Reserved codes decode to their names, not to `None` and not to an empty string. This is
what makes a padded batch renderable — and a padded batch is the array most likely to be
handed to `decode`, so a decoder that covered only user codes would fail on its commonest
input. Totality is structural: the reserved block is inserted into the reverse map at
construction, so there is no branch that could omit it.

An out-of-range code raises, naming the position and the size:

```python
>>> vocab.decode([99])
KeyError: 'code 99 at position 0 is out of range for this vocabulary (size 10)'
```

## `Vocabulary`

The protocol the container programs against, so a different encoder can be substituted
without touching the container. It is a `runtime_checkable` `Protocol` — structural, not
nominal, so anything with the right members satisfies it with no registration and no
inheritance.

```python
>>> from pfsmgraph.dataseq import Vocabulary
>>> isinstance(vocab, Vocabulary)
True
```

Members: `size`, `sym_to_code`, `code`, `encode`, `decode`.

`sym_to_code` and `code` are on the **protocol**, not merely on `SymbolTable`, and that is
the load-bearing detail. The consumer is in a different distribution: were they omitted
here, a substitute vocabulary would satisfy `Vocabulary`, pass every `dataseq` test, and
then fail inside `pfsmgraph-align` when the scoring matrix tried to read the mapping. An
implementation must also not return something a consumer can mutate.

Note that `runtime_checkable` checks only for the *presence* of members, never their
signatures or their behaviour. `isinstance` returning `True` does not establish that a
substitute decodes totally or encodes strictly. Those are contracts, not types.

## Not implemented, deliberately

**Persistence.** A `SymbolTable` cannot be saved or loaded. Serialising a symbol table
means settling an escaping rule for symbols containing the delimiter, and that is a
decision worth making on its own rather than as a side effect of an encoder API. See
`docs/plan/DEFERRED.md`, trigger "a vocabulary outliving the process that built it".

**Frequency ordering.** Covered above, same file, trigger "a corpus large enough for code
locality to matter".
