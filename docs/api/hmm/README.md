# `pfsmgraph.hmm`

Hidden Markov models in the arc-emission formulation, translated from a Lush original.
Version 0.1.0 carried the model's parameters and its decode. 0.2.0 adds a batched decode,
Baum-Welch training over a fixed topology, and a choice of backend for each; topology
search is a later revision.

- **Parameters** — [`params.md`](params.md): `HMMParams`.
- **Decode** — [`viterbi.md`](viterbi.md): `viterbi`, `viterbi_batch`, `ViterbiPath`,
  `ImpossibleSequenceError`.
- **Training** — [`baum_welch.md`](baum_welch.md): `baum_welch`, `BaumWelchResult`.
- **Backends** — [`backends.md`](backends.md): `backend=`, `backends`, `BackendStatus`,
  `BackendUnavailableError`.

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

### Training it

The same model, as a starting point, trained on three records and then used to decode
them together:

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
>>> float(params.transition_p[0, 0])
0.7
```

Training takes the corpus from 13.44 bits to 10.40. It returns a new `HMMParams` in
`result.params` and leaves the model it started from untouched, as the last line shows.

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
   `ImpossibleSequenceError`, a `ValueError` subclass, from `viterbi`, from
   `viterbi_batch` unless `on_impossible="none"` asks for `None` in its place, and from
   `baum_welch` when the starting model gives it no path. No call returns a path with
   infinite `total_bits`.
6. **Ties go to the smallest state index**, at every step and at the final state. Two
   equally cheap predecessors resolve to the lower-numbered one, so a decode is a function
   of its inputs rather than of the order in which a backend happened to compare them.
7. **Parameters are a frozen value.** `HMMParams` copies its arrays and marks them
   read-only; the stationary distribution and the entropies are computed on first access,
   never stored beside the arrays they derive from. `viterbi`, `viterbi_batch` and
   `baum_welch` are free functions over that value, not methods on a model, and training
   returns a new value rather than changing the one it was given.
   ([ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md))
8. **Training keeps the topology.** An arc or an emission of probability 0 carries no
   expected count, so it is still 0 after every cycle. A state the last cycle could not
   re-estimate keeps its previous row and fibres, and `degenerate_states` names it.
9. **Training counts per record.** `records` is a corpus, and each record's counts are
   computed separately and summed in record order before each M-step, so no transition is
   counted across a record boundary. One record holding a whole concatenated corpus is
   the special case, and trains as the Lush original's single stream did.
10. **Batching bounds memory and changes no result.** `batch_size` sets how many records
    are padded into one kernel call. `viterbi_batch`'s row `i` is `viterbi` on record `i`,
    bit for bit, on every backend and at every `batch_size`; `baum_welch`'s result is the
    same at every `batch_size` to the bit on the reference and its compiled phases, and
    within `torch`'s tolerance on `torch`. Padding weighs nothing: `PAD`'s fibres are zero
    by contract 4.
11. **A backend is chosen, never inferred.** `backend=` is keyword-only and defaults to
    `"python"` whatever is installed; a backend that cannot run raises
    `BackendUnavailableError` rather than falling back. The compiled phases are
    bit-identical to the reference on one host. `torch` agrees within
    `N · eps · max(1, x)` per expected count and per description length.
    ([ADR 0021](../../design/adr/0021-runtime-backend-selection.md),
    [ADR 0020](../../design/adr/0020-scaled-probability-domain-forward-backward.md))
12. **`device=` names a torch device, and only `torch` takes one** other than the CPU.
    It is probed before any work and never chosen for you; `backend="cuda"` runs on
    numba-cuda's current device and takes no `device=` at all.

## The public surface

`pfsmgraph.hmm.__all__` is exactly ten names. Anything underscore-prefixed — the
`_params`, `_viterbi`, `_baum_welch`, `_forward_backward`, `_backends`, `_numeric`,
`_search`, `_trials`, `_topology` and `_mdl` modules, the backend kernels, and every
attribute beginning with `_` — is private, out of contract, and may change without notice.

The last four are revision 04's topology search, and their privacy is a decision rather
than an omission: [ADR 0025](../../design/adr/0025-topology-search-not-exported.md) keeps
`__all__` at ten names at 0.3.0, because `__all__` may widen at a later version at no cost
to any consumer and may never narrow. Private here means out of contract, not unreachable —
`_search` imports and runs, and what is withheld is the promise that it will keep working.

```python
>>> import pfsmgraph.hmm
>>> pfsmgraph.hmm.__all__
['BackendStatus', 'BackendUnavailableError', 'BaumWelchResult', 'HMMParams', 'ImpossibleSequenceError', 'ViterbiPath', 'backends', 'baum_welch', 'viterbi', 'viterbi_batch']
```

| Name | Kind | Documented in |
|---|---|---|
| `HMMParams` | frozen dataclass | [params.md](params.md) |
| `viterbi` | function | [viterbi.md](viterbi.md) |
| `viterbi_batch` | function | [viterbi.md](viterbi.md) |
| `ViterbiPath` | frozen dataclass | [viterbi.md](viterbi.md) |
| `ImpossibleSequenceError` | exception | [viterbi.md](viterbi.md) |
| `baum_welch` | function | [baum_welch.md](baum_welch.md) |
| `BaumWelchResult` | frozen dataclass | [baum_welch.md](baum_welch.md) |
| `backends` | function | [backends.md](backends.md) |
| `BackendStatus` | frozen dataclass | [backends.md](backends.md) |
| `BackendUnavailableError` | exception | [backends.md](backends.md) |

## Backends and extras

**`viterbi` and `viterbi_batch` have four implementations, and `backend=` chooses among
them**: `"python"`, `"cython"`, `"cpu_parallel"` and `"cuda"`, one per lifecycle phase,
each tested against the others for exact equality on the same inputs. **`baum_welch` has
five**: the same four, which compute the E-step with the same sums in the same order and
agree to the bit, and `"torch"`, which derives the counts as gradients and agrees within a
measured tolerance. **The default is `"python"`**, and nothing in
the environment changes it. **A backend that cannot run here raises
`BackendUnavailableError`** naming what is missing; nothing falls back. `backends(name)`
reports which can run, and why the others cannot.
([ADR 0021](../../design/adr/0021-runtime-backend-selection.md))

**Every backend's dependencies beyond numpy are optional extras**, `cpu-parallel` for numba,
`gpu` for numba-cuda and `torch` for torch, so the base install stays lean. GPU here means
`numba-cuda`; the `torch` extra needs no device and is unrelated to `gpu`.
([ADR 0004](../../design/adr/0004-gpu-backends-and-optional-dependency-strategy.md))
[backends.md](backends.md) has the detail.

## Related records

- [ADR 0015](../../design/adr/0015-arc-emission-mealy-formulation.md) — arc emission.
- [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md) — the frozen
  parameter value, and why the decode is a free function.
- [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) —
  the reserved block the symbol axis spans.
- [ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md) — the four-phase
  lifecycle the compiled backends follow.
- [ADR 0021](../../design/adr/0021-runtime-backend-selection.md) — choosing a backend at
  run time, and why nothing falls back.
- [ADR 0020](../../design/adr/0020-scaled-probability-domain-forward-backward.md) — the
  forward-backward's numeric contract, and the tolerance `torch` is held to.
- [`viterbi/FORMALIZATION.md`](../../design/algorithms/viterbi/FORMALIZATION.md) — the
  decode's recurrence, base cases and tie-breaking rule, independent of any language, and
  its batched form.
- [`forward_backward/FORMALIZATION.md`](../../design/algorithms/forward_backward/FORMALIZATION.md)
  — the E-step's recurrences and the evaluation order every training backend but `torch`
  reproduces.
