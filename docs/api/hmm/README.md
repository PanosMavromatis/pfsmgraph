# `pfsmgraph.hmm`

Hidden Markov models in the arc-emission formulation, translated from a Lush original.
Version 0.1.0 carries the model's parameters and its decode; training (Baum-Welch) and
topology search are later revisions.

- **Parameters** — [`params.md`](params.md): `HMMParams`.
- **Decode** — [`viterbi.md`](viterbi.md): `viterbi`, `ViterbiPath`,
  `ImpossibleSequenceError`.

It depends on `pfsmgraph-dataseq`, for its vocabulary and its records, and on `numpy`.

## The one example worth reading first

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

Three things in that example are unlike every textbook HMM, and each is deliberate.

`output_p` has **three** axes, not two. A symbol is emitted while crossing the arc from
state `i` to state `j`, so its probability is `output_p[i, j, symbol]`: there is no
per-state emission matrix `B[state, symbol]` anywhere in this package.

It is sized to `vocab.size`, which is 8 for two symbols. The first six codes are the
reserved block `dataseq` fixes, and their fibres are left at zero. A record's codes index
the axis directly, with no offset.

Four symbols give **five** states. `states[t]` is the state occupied *before* symbol `t`
is emitted, so a path always visits one more state than it emits symbols.

## The contracts

These hold for every caller and are not configurable.

1. **Emission is on the arc.** The emission parameter is `output_p[i, j, symbol]`,
   indexed by source state, destination state and symbol. The emission factor depends on
   both endpoints, so it cannot be hoisted out of a recurrence over either one.
   ([ADR 0015](../../design/adr/0015-arc-emission-mealy-formulation.md))
2. **A path over `N` symbols visits `N + 1` states.** `ViterbiPath.states` is one longer
   than `record.codes`, and `ViterbiPath.n_symbols` is `N`.
3. **The decode minimises bits; it does not maximise probability.** `total_bits` is the
   description length of the path, `-log2` of its probability. It grows as the path
   becomes less likely, and an impossible arc costs `+inf`.
4. **The reserved fibres are exactly zero.** `output_p[:, :, :USER_BASE]` must hold zeros,
   and `HMMParams` refuses anything else. So a model never emits `PAD`, `UNK`, `BOS`,
   `EOS`, `GAP` or `MSK`, and a record carrying one is impossible, by arithmetic rather
   than by convention.
   ([ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md))
5. **Impossibility is a type.** A record with no finite-cost path raises
   `ImpossibleSequenceError`, a `ValueError` subclass. It never returns a path with
   infinite `total_bits`.
6. **Ties go to the smallest state index**, at every step and at the final state. Two
   equally cheap predecessors resolve to the lower-numbered one, so a decode is a function
   of its inputs rather than of the order in which a backend happened to compare them.
7. **Parameters are a frozen value.** `HMMParams` copies its arrays and marks them
   read-only; the stationary distribution and the entropies are computed on first access,
   never stored beside the arrays they derive from. `viterbi` is a free function over
   that value, not a method on a model.
   ([ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md))

## The public surface

`pfsmgraph.hmm.__all__` is exactly four names. Anything underscore-prefixed — the
`_params`, `_viterbi` and `_numeric` modules, the backend kernels, and every attribute
beginning with `_` — is private, out of contract, and may change without notice.

```python
>>> import pfsmgraph.hmm
>>> pfsmgraph.hmm.__all__
['HMMParams', 'ImpossibleSequenceError', 'ViterbiPath', 'viterbi']
```

| Name | Kind | Documented in |
|---|---|---|
| `HMMParams` | frozen dataclass | [params.md](params.md) |
| `viterbi` | function | [viterbi.md](viterbi.md) |
| `ViterbiPath` | frozen dataclass | [viterbi.md](viterbi.md) |
| `ImpossibleSequenceError` | exception | [viterbi.md](viterbi.md) |

## Backends and extras

**`viterbi` runs a pure-Python/numpy kernel, and in 0.1.0 there is no way to choose
another.** The repository also holds a Cython, a Numba CPU-parallel and a Numba CUDA
translation of the same kernel, each tested against it for equivalence. None of them is
reachable through a public call: selecting a backend at run time is an open question
([ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md)), and
until it is settled the public decode is the reference kernel only.

**0.1.0 declares no optional extras.** A backend you could not select would install `numba`
or `numba-cuda` for nothing, so the extras arrive with the API that makes them reachable.
When they do, GPU here means `numba-cuda`, unrelated to the `torch` stack `pfsmgraph-dl`
uses.
([ADR 0004](../../design/adr/0004-gpu-backends-and-optional-dependency-strategy.md))

## Related records

- [ADR 0015](../../design/adr/0015-arc-emission-mealy-formulation.md) — arc emission.
- [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md) — the frozen
  parameter value, and why the decode is a free function.
- [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) —
  the reserved block the symbol axis spans.
- [ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md) — the four-phase
  lifecycle the backends follow.
- [`FORMALIZATION.md`](../../design/algorithms/viterbi/FORMALIZATION.md) — the
  recurrence, base cases and tie-breaking rule, independent of any language.
