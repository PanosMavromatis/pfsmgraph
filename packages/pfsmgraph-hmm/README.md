# pfsmgraph-hmm

Hidden Markov models in the arc-emission formulation, for the
[`pfsmgraph`](https://github.com/PanosMavromatis/pfsmgraph) family: a model's parameters,
its Viterbi decode, one record at a time or many padded together, and Baum-Welch training
over a fixed topology, each with a choice of backend. Topology search is forthcoming.

It depends on `pfsmgraph-dataseq`, for its vocabulary and its records, and on numpy.

```bash
pip install pfsmgraph-hmm
```

## Emission is on the arc

A two-state model over two symbols, and one decoded sequence:

```python
import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceDataset, SymbolTable
from pfsmgraph.hmm import HMMParams, viterbi

vocab = SymbolTable(["a", "b"])

output_p = np.zeros((2, 2, vocab.size))
output_p[0, 0, USER_BASE:] = [0.9, 0.1]
output_p[0, 1, USER_BASE:] = [0.2, 0.8]
output_p[1, 0, USER_BASE:] = [0.5, 0.5]
output_p[1, 1, USER_BASE:] = [0.1, 0.9]

params = HMMParams(
    init_state_p=[0.6, 0.4],
    transition_p=[[0.7, 0.3], [0.4, 0.6]],
    output_p=output_p,
    vocabulary=vocab,
)

ds = SequenceDataset.from_symbols([["a", "a", "b", "b"]], vocab, labels=["s1"])
path = viterbi(params, ds[0])
```

```python
>>> vocab.size
8
>>> path
ViterbiPath(n_symbols=4, total_bits=5.0180, label='s1')
>>> path.states
array([0, 0, 0, 1, 1])
```

`output_p` has three axes: a symbol is emitted while crossing the arc from state `i` to
state `j`, so its probability is `output_p[i, j, symbol]`, and there is no per-state
emission matrix anywhere in this package. It is sized to `vocab.size`, reserved codes
included, so a record's codes index it directly. And four symbols give five states:
`states[t]` is the state occupied before symbol `t` is emitted.

## The decode minimises bits

`total_bits` is the description length of the path, `-log2` of its probability, so it grows
as the path becomes less likely. A sequence every path of which crosses a zero-probability
arc raises `ImpossibleSequenceError`, a `ValueError`, rather than returning a path.

## Training

`baum_welch` re-estimates a model's parameters over a corpus of records until the
description length stops falling, and returns a new model rather than changing the one it
was given. Arcs of probability zero stay zero, so the topology is fixed:

```python
from pfsmgraph.hmm import baum_welch, viterbi_batch

corpus = SequenceDataset.from_symbols(
    [["a", "a", "b", "b", "a", "b"], ["b", "b", "a"], ["a", "a", "a", "b"]],
    vocab,
    labels=["s1", "s2", "s3"],
)
result = baum_welch(params, corpus)
```

```python
>>> result.cycles, result.converged
(50, True)
>>> round(result.description_lengths[0], 4), round(result.description_lengths[-1], 4)
(13.4368, 10.4047)
>>> viterbi_batch(result.params, corpus)
[ViterbiPath(n_symbols=6, total_bits=4.8905, label='s1'), ViterbiPath(n_symbols=3, total_bits=2.1181, label='s2'), ViterbiPath(n_symbols=4, total_bits=3.4267, label='s3')]
```

Each record's counts are kept apart, so no transition is counted across a record boundary.
`viterbi_batch` and `baum_welch` take a `batch_size`, which bounds how many records are
padded into one call and changes no result: `viterbi_batch`'s row `i` is `viterbi` on
record `i`, bit for bit.

## Backends

Every call has four implementations — pure Python/numpy, Cython, Numba CPU-parallel and
Numba CUDA — each tested against the others for exact equality on one machine.
`baum_welch` has a fifth, `torch`, which derives the expected counts as gradients and
agrees with the reference within a measured tolerance rather than exactly. The keyword-only
`backend=` chooses one, and defaults to the pure-Python reference whatever is installed. A
backend that cannot run here raises `BackendUnavailableError` naming what is missing, and
nothing falls back; `backends()` reports what can run:

```python
>>> from pfsmgraph.hmm import backends
>>> [s.name for s in backends("viterbi")]
['python', 'cython', 'cpu_parallel', 'cuda']
>>> [s.name for s in backends("baum_welch")]
['python', 'cython', 'cpu_parallel', 'cuda', 'torch']
>>> viterbi(params, ds[0], backend="python").total_bits == path.total_bits
True
```

The accelerated backends' dependencies are optional extras:

```bash
pip install 'pfsmgraph-hmm[cpu-parallel]'   # numba
pip install 'pfsmgraph-hmm[gpu]'            # numba-cuda, not on macOS
pip install 'pfsmgraph-hmm[torch]'          # torch, for baum_welch
```

`gpu` means numba-cuda and is unrelated to `torch`, whose `baum_welch` backend runs on the
CPU unless `device=` names a torch device such as `"cuda:0"`.

## Documentation

The contracts a caller may rely on — the invariants, why they hold, and the seams between
this distribution and the rest of the family — are in
[`docs/api/hmm/`](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/api/hmm/README.md).
Docstrings are normative for signatures; that directory is normative for contracts. Every
code block in both, this README included, is executed and its output pasted from the run.

The decisions behind them are recorded in
[`docs/design/adr/`](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/README.md).

## License

MIT.
