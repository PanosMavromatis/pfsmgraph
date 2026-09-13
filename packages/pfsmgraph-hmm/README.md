# pfsmgraph-hmm

Hidden Markov models in the arc-emission formulation, for the
[`pfsmgraph`](https://github.com/PanosMavromatis/pfsmgraph) family. Version 0.1.0 carries a
model's parameters and its Viterbi decode; Baum-Welch training and topology search are
forthcoming.

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

## One kernel in 0.1.0

`viterbi` runs a pure-Python/numpy kernel. Cython, Numba CPU-parallel and Numba CUDA
translations of it exist in the repository, each tested against it for equivalence, but
no public call selects them yet, so this release ships no compiled code and declares no
optional extras.

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
