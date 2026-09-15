# Decode

`viterbi` and its batched form `viterbi_batch`, the `ViterbiPath` they return, and the
`ImpossibleSequenceError` they raise. See
[README.md](README.md) for the contracts, and [params.md](params.md) for the model it
reads.

Every example on this page runs against the same model and record:

```python
import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceDataset, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, ImpossibleSequenceError, ViterbiPath, viterbi, viterbi_batch

vocab = SymbolTable(["a", "b"])

output_p = np.zeros((2, 2, vocab.size))
output_p[0, 0, USER_BASE:] = [0.9, 0.1]
output_p[0, 1, USER_BASE:] = [0.2, 0.8]
output_p[1, 0, USER_BASE:] = [0.5, 0.5]
output_p[1, 1, USER_BASE:] = [0.1, 0.9]

params = HMMParams([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], output_p, vocab)

ds = SequenceDataset.from_symbols([["a", "a", "b", "b"]], vocab, labels=["s1"])
record = ds[0]
path = viterbi(params, record)
```

## `viterbi`

```python
viterbi(params: HMMParams, record: SequenceRecord, *, backend: BackendName = "python") -> ViterbiPath
```

The most probable state path over one record under one model. `backend` chooses which of
four implementations runs it, all returning the same path; [backends.md](backends.md)
describes them.

```python
>>> record.codes
array([6, 6, 7, 7], dtype=int32)
>>> path
ViterbiPath(n_symbols=4, total_bits=5.0180, label='s1')
```

A free function, not a method: the model is a frozen value, and the decode reads it
without owning it. It takes **one record**, which never holds padding, so there is no mask
to consult. To decode many records padded together, see [`viterbi_batch`](#viterbi_batch).

### It minimises bits

The quantity minimised is the path's description length, `-log2` of its probability,
summed over the start and every arc crossed. You can recompute it from the path:

```python
>>> s, c = path.states, record.codes
>>> cost = -np.log2(params.init_state_p[s[0]]) + sum(
...     -np.log2(params.transition_p[s[t], s[t + 1]] * params.output_p[s[t], s[t + 1], c[t]])
...     for t in range(len(c)))
>>> bool(abs(cost - path.total_bits) < 1e-12)
True
```

Lower is better. An arc of probability 0 costs `+inf`, which is how impossibility
propagates without any special case.

The comparison is against a tolerance on purpose: `total_bits` is exact within one
implementation, but a different floating-point `log2` can place a sum one unit in the last
place away.

### Ties go to the smallest state index

When two predecessors give exactly the same cost, the lower-numbered one wins, at every
step and in choosing the final state. A model whose every probability is uniform ties
everywhere, so its path is all zeros:

```python
>>> uniform_out = np.zeros((3, 3, vocab.size))
>>> uniform_out[:, :, USER_BASE:] = 0.5
>>> uniform = HMMParams(np.full(3, 1 / 3), np.full((3, 3), 1 / 3), uniform_out, vocab)
>>> viterbi(uniform, record).states
array([0, 0, 0, 0, 0])
```

This rule is part of the contract, not an implementation detail. Without it, two correct
implementations could return different paths for the same input. A uniform model is also
not an edge case: it is the natural initialisation for training.

### Errors

Two different failures, and they deserve different handling.

**A code outside the model's symbol axis is a malformed input**, a plain `ValueError`
raised before the decode runs. The usual cause is a record encoded against a different
vocabulary:

```python
>>> other = SymbolTable(["a", "b", "c"])
>>> viterbi(params, SequenceRecord(other.encode(["c"])))
ValueError: record holds code(s) outside the model's symbol axis [0, 8): observed [8, 8]. The usual cause is a record encoded against a different vocabulary than the one output_p was sized against
```

Note that this check is only a range check. A record from another vocabulary whose codes
all happen to fall in range decodes without error, against the wrong symbols. Keep a
record and the model it is decoded against on one `Vocabulary`.

**A record with no finite-cost path is an `ImpossibleSequenceError`.** Every path crosses
some arc of probability zero. A reserved code always does this, since the reserved fibres
are zero, and `on_unknown="unk"` is how one gets into a record:

```python
>>> unk_record = SequenceRecord(vocab.encode(["a", "zz", "b"], on_unknown="unk"), label="u")
>>> unk_record.codes
array([6, 1, 7], dtype=int32)
>>> viterbi(params, unk_record)
ImpossibleSequenceError: no path over these 3 symbols has finite description length under a model of 2 state(s): every state became unreachable emitting symbol 1 of the record, code 1, which reaches path position 2. No arc carrying that code leaves a reachable state with positive probability, and bits(0) is +inf, which absorbs under addition
```

The message names the first symbol after which no state was reachable. The error is a
`ValueError` subclass, so `except ValueError` catches both failures:

```python
>>> issubclass(ImpossibleSequenceError, ValueError)
True
```

It is a separate type so that code can tell the two apart without matching on a message.
When decoding many sequences against a candidate model, an impossible sequence says
something about the model; an out-of-range code says something about the caller.

### An empty record

A record with no symbols decodes to a single state: the one with the highest initial
probability.

```python
>>> empty = viterbi(params, SequenceRecord(vocab.encode([])))
>>> empty
ViterbiPath(n_symbols=0, total_bits=0.7370)
>>> empty.states
array([0])
```

### Differences from the Lush original

The original seeded the recurrence with raw initial probabilities where it needed their
bit costs. Here the start is costed in bits like every arc, as above. So a decode here can
differ from a path the original saved at the first position, where the choice of start
state is decided; the rest of the recurrence is unchanged.

## `viterbi_batch`

```python
viterbi_batch(params: HMMParams, records, *, backend: BackendName = "python", batch_size: int | None = None, on_impossible: str = "raise") -> list[ViterbiPath | None]
```

Many records decoded as one padded array, one result per record in record order. `records`
is any iterable of `SequenceRecord`, a `SequenceDataset` included.

```python
>>> corpus = SequenceDataset.from_symbols([["a", "a", "b", "b"], [], ["b"], ["b", "a", "a", "b", "a"]], vocab, labels=["s1", "empty", "one", "s4"])
>>> paths = viterbi_batch(params, corpus)
>>> paths
[ViterbiPath(n_symbols=4, total_bits=5.0180, label='s1'), ViterbiPath(n_symbols=0, total_bits=0.7370, label='empty'), ViterbiPath(n_symbols=1, total_bits=2.2109, label='one'), ViterbiPath(n_symbols=5, total_bits=9.3578, label='s4')]
>>> paths[3].states
array([1, 0, 0, 0, 1, 0])
```

### Each result is `viterbi` on that record alone

Not approximately: the same states and the same `total_bits`, bit for bit, on every backend
and at every `batch_size`. The decode performs only `+` and `<`, and padding never enters
the arithmetic, so there is no tolerance to state.

```python
>>> all(np.array_equal(p.states, viterbi(params, r).states) and p.total_bits == viterbi(params, r).total_bits for p, r in zip(paths, corpus))
True
```

### `batch_size` bounds memory and nothing else

`None` pads every record into one call; an integer pads that many at a time. A call holds
`δ` and `ψ` of shape `(B, L + 1, S)` for the longest record `L` in it, so a large corpus of
long records wants a bound. Measured on a CPU, one unbounded call over 256 records at
`S = 160` was slower than decoding them one at a time; on a GPU the batch is the fast path.

```python
>>> [p.total_bits for p in viterbi_batch(params, corpus, batch_size=1)] == [p.total_bits for p in paths]
True
>>> viterbi_batch(params, [])
[]
```

### An impossible record

By default it raises, naming the record's index ahead of `viterbi`'s own diagnosis. The
first impossible record in record order is the one reported:

```python
>>> viterbi_batch(params, [record, unk_record])
ImpossibleSequenceError: record 1: no path over these 3 symbols has finite description length under a model of 2 state(s): every state became unreachable emitting symbol 1 of the record, code 1, which reaches path position 2. No arc carrying that code leaves a reachable state with positive probability, and bits(0) is +inf, which absorbs under addition
```

`on_impossible="none"` puts `None` in that record's place and decodes the rest. That is the
setting for decoding a corpus against candidate models, where an impossible record says
something about the model rather than being an error:

```python
>>> viterbi_batch(params, [record, unk_record], on_impossible="none")
[ViterbiPath(n_symbols=4, total_bits=5.0180, label='s1'), None]
```

### Errors before any work

Every argument and every record's codes are checked before the first record is decoded, so
a malformed call fails the same way whatever its records hold:

```python
>>> viterbi_batch(params, [record, SequenceRecord(other.encode(["c"]))])
ValueError: record 1: record holds code(s) outside the model's symbol axis [0, 8): observed [8, 8]. The usual cause is a record encoded against a different vocabulary than the one output_p was sized against
>>> viterbi_batch(params, [record], on_impossible="skip")
ValueError: on_impossible must be one of ['raise', 'none'], got 'skip'
>>> viterbi_batch(params, [record], backend="torch")
ValueError: viterbi_batch has no 'torch' backend: that lifecycle phase is not implemented for it. It has ['python', 'cython', 'cpu_parallel', 'cuda']
```

## `ViterbiPath`

```python
ViterbiPath(states: np.ndarray, total_bits: float, label: str | None = None)
```

A frozen dataclass.

| Field | Meaning |
|---|---|
| `states` | `(N + 1,)` `int64`, read-only. `states[t]` is the state occupied before symbol `t` is emitted |
| `total_bits` | the description length of this path. Finite in every path `viterbi` returns; the constructor does not check it |
| `label` | the record's `label`, carried through unchanged |
| `n_symbols` | property, `N`. One fewer than `len(states)` |

```python
>>> path.states
array([0, 0, 0, 1, 1])
>>> path.n_symbols
4
>>> path.total_bits
5.017980503380648
>>> path.label
's1'
>>> path.states[0] = 1
ValueError: assignment destination is read-only
```

`states` holds state indices, not vocabulary codes, so there is nothing to decode them to;
`label` is the only string that crosses this boundary.

### Per-position entropies are not stored

The original saved an entropy beside each state of a decoded path. Here it is one index
away, and derived:

```python
>>> params.state_entropies[path.states]
array([0.89317346, 0.89317346, 0.89317346, 0.82674637, 0.82674637])
```

A stored copy could disagree with the model it came from; this cannot.

## `ImpossibleSequenceError`

A subclass of `ValueError`, raised by `viterbi` when no path over the record has finite
description length. It carries no attributes beyond its message. See
[Errors](#errors) above.
