# Training

`baum_welch` and the `BaumWelchResult` it returns: Baum-Welch re-estimation of a model's
parameters over a fixed topology. See [README.md](README.md) for the contracts,
[params.md](params.md) for the model it trains, and [backends.md](backends.md) for
`backend=`.

Every example on this page runs against the same starting model and corpus:

```python
import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceDataset, SequenceRecord, SymbolTable
from pfsmgraph.hmm import BaumWelchResult, HMMParams, ImpossibleSequenceError, baum_welch

vocab = SymbolTable(["a", "b"])

output_p = np.zeros((2, 2, vocab.size))
output_p[0, 0, USER_BASE:] = [0.9, 0.1]
output_p[0, 1, USER_BASE:] = [0.2, 0.8]
output_p[1, 0, USER_BASE:] = [0.5, 0.5]
output_p[1, 1, USER_BASE:] = [0.1, 0.9]

params = HMMParams([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], output_p, vocab)

ds = SequenceDataset.from_symbols(
    [["a", "a", "b", "b", "a", "b"], ["b", "b", "a"], ["a", "a", "a", "b"]],
    vocab,
    labels=["s1", "s2", "s3"],
)
result = baum_welch(params, ds)
```

## `baum_welch`

```python
baum_welch(params: HMMParams, records, *, backend: BackendName = "python", batch_size: int | None = None, device: str | None = None, batch_cycles: int = 10, change_bits: float = 0.1, patience: int = 3, max_cycles: int | None = None) -> BaumWelchResult
```

Alternates an E-step, the expected counts of every start, arc crossing and emission under
the current model, with an M-step that turns those counts into new parameters, until the
stopping rule fires. The topology is fixed: an arc of probability 0 stays 0, because it
carries no count.

```python
>>> result.cycles
50
>>> result.converged
True
>>> result.params
HMMParams(n_states=2, n_symbols=8)
```

A free function over a frozen value, like `viterbi`: it **returns** a new `HMMParams` and
leaves `params` untouched.

### It trains on records, not on one stream

`records` is any sequence of `SequenceRecord`, a `SequenceDataset` included. Each record's
counts are computed separately and summed before every M-step, so no transition is ever
counted across the boundary between two records. A whole corpus concatenated into one
record is the special case, and trains exactly as the Lush original's single flat stream
did.

### It batches, and the result does not depend on how

`batch_size` is how many records go through the E-step together, padded to the longest of
them; `None`, the default, sends the whole corpus at once. Memory grows as
`batch_size · S² · A`, and nothing else changes: each record's counts come back separately
and are summed in record order, so on `"python"` a run at any batch size is identical to
the last bit:

```python
>>> one_at_a_time = baum_welch(params, ds, batch_size=1)
>>> one_at_a_time.description_lengths == result.description_lengths
True
>>> all(bool((a == b).all()) for a, b in zip(
...     (one_at_a_time.params.init_state_p, one_at_a_time.params.transition_p, one_at_a_time.params.output_p),
...     (result.params.init_state_p, result.params.transition_p, result.params.output_p)))
True
```

Padding cannot reach a count: `PAD`'s emission probability is zero in every model, and the
E-step masks padded positions besides.

### It stops by the original's rule

Every `batch_cycles` cycles it recomputes the corpus description length, in bits; a batch
that moved it by less than `change_bits` counts as unchanged, and `patience` unchanged
batches in a row stop training. The defaults are `run-converge`'s. The description length
never rises, so the loop terminates without `max_cycles`, which is a budget for callers
that want one; reaching it returns with `converged` false:

```python
>>> short = baum_welch(params, ds, max_cycles=5)
>>> short.cycles, short.converged
(5, False)
```

**The rule can stop on a symmetric saddle**, since it watches only how much each batch
moves. An exactly uniform starting model is one, and it stays uniform. Break symmetry when
initialising.

## `BaumWelchResult`

```python
BaumWelchResult(params: HMMParams, description_lengths: tuple[float, ...], cycles: int, converged: bool, degenerate_states: tuple[int, ...])
```

A frozen dataclass. `description_lengths[c]` is the corpus description length of the
model after `c` cycles, so it has `cycles + 1` entries, the first being the starting
model's:

```python
>>> len(result.description_lengths) == result.cycles + 1
True
>>> round(result.description_lengths[0], 4), round(result.description_lengths[-1], 4)
(13.4368, 10.4047)
```

**`degenerate_states` names states the last cycle could not re-estimate.** A state
occupied only at the last position of every record has no outgoing count, so the M-step
has nothing to divide by. Its previous row and emission fibres are kept, which leaves the
description length exactly what the original's all-zero row gives while keeping the model
valid. Any row maximises the objective for such a state, so no information is lost.

```python
>>> result.degenerate_states
()
```

## Backends

`baum_welch` has two backends, which differ in how the E-step is computed; the M-step and
the stopping rule are shared.

| Name | E-step | Needs |
|---|---|---|
| `"python"` | explicit forward and backward passes, numpy, the reference | nothing |
| `"torch"` | the forward pass only; the counts are reverse-mode gradients | the `torch` extra |

`torch` derives the counts rather than writing them out, which is why the two are worth
comparing: they share no code. **They agree within a tolerance, not bit for bit**, because
torch chooses its own order for the sums over states. Measured against the reference, each
expected count agrees within `N · eps · max(1, count)`, where `N` is the record's length and
`eps` float64's machine epsilon, and each description length within
`N · eps · max(1, bits)`. A whole training run stops at the same cycle, with the same
degenerate states:

```python
>>> theirs = baum_welch(params, ds, backend="torch")
>>> theirs.cycles == result.cycles and theirs.degenerate_states == result.degenerate_states
True
>>> all(bool(np.allclose(a, b, rtol=0, atol=1e-12)) for a, b in zip(
...     (theirs.params.init_state_p, theirs.params.transition_p, theirs.params.output_p),
...     (result.params.init_state_p, result.params.transition_p, result.params.output_p)))
True
```

That bound is relative to the size of each value and grows with `N`, and it is not "a few
units in the last place": a record that costs a fraction of a bit can differ by thousands of
them relative to itself while differing by `1e-14` bits outright. `torch` runs in float64 on
the CPU unless `device=` names a torch device such as `"cuda:0"`, which is probed before
training starts and never falls back; nothing picks a device for you. Only `torch` takes a
device other than `"cpu"`, and the tolerance above holds there too.

Neither backend is a lifecycle phase of the other, so `"cython"` and the rest are not
`baum_welch` backends:

```python
>>> baum_welch(params, ds, backend="cython")
ValueError: baum_welch has no 'cython' backend: that lifecycle phase is not implemented for it. It has ['python', 'torch']
```

## Errors

Validation happens before any cycle runs. A record with no path of finite description
length under the starting model cannot be trained on, since all of its counts are zero:

```python
>>> impossible = SequenceRecord(np.array([USER_BASE, 1]))
>>> baum_welch(params, [impossible])
ImpossibleSequenceError: record 0 has no path of finite description length under the starting model of 2 state(s), so every one of its expected counts is zero and it cannot be trained on
```

Code 1 is `UNK`, whose fibres are zero in every model, so no path emits it. A code outside
the model's symbol axis, a corpus with no symbols, and a stopping rule that cannot stop
raise `ValueError`.

So do a `batch_size` below 1 and a device the backend cannot use. Only `"torch"` runs
anywhere but the CPU, and a device name must be a string:

```python
>>> baum_welch(params, ds, batch_size=0)
ValueError: batch_size must be at least 1 or None, got 0
>>> baum_welch(params, ds, device="cuda")
ValueError: backend 'python' runs on the CPU only; device='cuda' needs backend='torch'
>>> baum_welch(params, ds, device=0)
TypeError: device must be a device name or None, got int
```

On `"torch"`, a name torch cannot parse raises `ValueError`, and a device it parses but
cannot allocate on here (a GPU ordinal past the last, say) raises `BackendUnavailableError`
carrying torch's own message, which differs by machine and build. Both are raised before
any cycle runs, and neither falls back to the CPU.
