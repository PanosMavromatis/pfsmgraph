# Training

`baum_welch` and the `BaumWelchResult` it returns: Baum-Welch re-estimation of a model's
parameters over a fixed topology. See [README.md](README.md) for the contracts,
[params.md](params.md) for the model it trains, and [backends.md](backends.md) for
`backend=`.

Every example on this page runs against the same starting model and corpus:

```python
import sys

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
baum_welch(params: HMMParams, records, *, backend: BackendName = "python", score_backend: BackendName = "python", batch_size: int | None = None, device: str | None = None, batch_cycles: int = 10, change_bits: float = 0.1, patience: int = 3, max_cycles: int | None = None, min_cycles: int = 0, log: TextIO | None = None) -> BaumWelchResult
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

**`min_cycles` holds the rule off** for a start known to sit near such a point, as a split
state's twins do ([ADR 0022](../../design/adr/0022-state-split-initialisation.md) section 4).
Checks still fall every `batch_cycles` cycles, so the floor takes effect at the first check
at or past it. The unchanged count keeps running beneath the floor, so a run that is
already quiet stops at that check, while a batch that changes still resets the count.
The model above converges in 50 cycles, so a floor of 75 stops it at the check at 80:

```python
>>> floored = baum_welch(params, ds, min_cycles=75)
>>> floored.cycles, floored.converged
(80, True)
```

A smaller `max_cycles` still wins, and returns with `converged` false.

### It reports progress on a stream, if given one

`log` is a text stream to report on; `None`, the default, writes nothing. Given
`sys.stdout`, it writes a header and the starting model's bits, a row at every
convergence check, and a line naming how the run stopped. Each line is flushed as it is
written, so a notebook cell shows a long run while it continues:

```python
>>> logged = baum_welch(params, ds, log=sys.stdout)
  cycle       bits         change  quiet
      0  13.436774
     10  11.072166  -2.364608e+00  0/3
     20  10.438376  -6.337892e-01  0/3
     30  10.414882  -2.349393e-02  1/3
     40  10.407574  -7.308038e-03  2/3
     50  10.404659  -2.915515e-03  3/3
  converged after 50 cycles
```

`change` is the row's bits minus the previous row's, and `quiet` counts unchanged checks
out of `patience`, so a reader sees a run approaching its stop. Bits are fixed-point, in a
column sized from cycle 0, which is the widest it gets since the description length never
rises; `change` is scientific, so a change far below `change_bits` still shows its size.
A `max_cycles` stop between checks writes one last row, with no `quiet` count because no
check was taken:

```python
>>> budget = baum_welch(params, ds, max_cycles=25, log=sys.stdout)
  cycle       bits         change  quiet
      0  13.436774
     10  11.072166  -2.364608e+00  0/3
     20  10.438376  -6.337892e-01  0/3
     25  10.422821  -1.555547e-02
  stopped at max_cycles after 25 cycles
```

The log only formats values already computed, so a run is identical with it and without
it, on every backend. Nothing is kept on the result; every number printed is already in
`description_lengths`. For a file, pass one: `log=open(path, "w")`. This is the Lush
original's `run-converge` made observable, not its training log, which recorded one line
per accepted split or merge and belongs with topology search.

That other log now exists, and this page is deliberately the wrong place to look for it.
The topology search takes a `log=` of its own and writes one row per round, carrying the
model size, the split or merge, both halves of the description length and `d` — the
original's columns. It is not documented here because the search is **private**: nothing
under `pfsmgraph.hmm` exports it, and [the public surface](README.md#the-public-surface)
is still exactly ten names. A search never passes this page's stream down to
`baum_welch` either, so the two logs cannot interleave: a round re-converges dozens of
candidates, and forwarding would bury the row that says what the search did under a
convergence block per rejected move.

## `BaumWelchResult`

```python
BaumWelchResult(params: HMMParams, description_lengths: tuple[float, ...], cycles: int, converged: bool, degenerate_states: tuple[int, ...])
```

A frozen dataclass.

| Field | Meaning |
|---|---|
| `params` | the trained `HMMParams`, a new value; the one passed in is untouched |
| `description_lengths` | the corpus description length in bits after each cycle, `cycles + 1` entries, the starting model's first |
| `cycles` | how many re-estimation cycles ran |
| `converged` | `True` when the stopping rule fired, `False` when `max_cycles` was reached first |
| `degenerate_states` | the states the last cycle could not re-estimate, ascending; `()` when there are none |

`description_lengths[c]` is the corpus description length of the
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

`baum_welch` has five backends, which differ in how the E-step is computed; the M-step and
the stopping rule are shared.

| Name | E-step | Needs |
|---|---|---|
| `"python"` | explicit forward and backward passes, numpy, the reference | nothing |
| `"cython"` | the same passes, compiled; **bit-identical** to `"python"` | a platform wheel, or a source build |
| `"cpu_parallel"` | the same passes in Numba, parallel over each timestep's (record, state) cells; **bit-identical** to `"python"` | the `cpu-parallel` extra |
| `"cuda"` | the same passes in Numba CUDA, one device thread per cell; **bit-identical** to `"python"` | the `gpu` extra and a CUDA device |
| `"torch"` | the forward pass only; the counts are reverse-mode gradients | the `torch` extra |

`"cython"`, `"cpu_parallel"` and `"cuda"` are lifecycle phases of the reference ([ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md)):
they perform the same sums in the same order with no fused multiply-add
([ADR 0020](../../design/adr/0020-scaled-probability-domain-forward-backward.md)), so a
training run on any of them is the reference's run, to the bit, on the same machine: on
`"cpu_parallel"` at any thread count, and on `"cuda"` on the device, where the kernel is
built so that no multiply-add is fused. The example runs the two that every checkout of
this repository can; `"cuda"` is held to the same equality by the test suite wherever a
device is present:

```python
>>> for name in ("cython", "cpu_parallel"):
...     compiled = baum_welch(params, ds, backend=name)
...     print(name, compiled.description_lengths == result.description_lengths and all(
...         bool(np.array_equal(a, b)) for a, b in zip(
...             (compiled.params.init_state_p, compiled.params.transition_p, compiled.params.output_p),
...             (result.params.init_state_p, result.params.transition_p, result.params.output_p))))
cython True
cpu_parallel True
```

`torch` derives the counts rather than writing them out, which is why it and the reference
are worth comparing: they share no code. **They agree within a tolerance, not bit for bit**, because
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

Every lifecycle phase trains, so `backend=` accepts every name `backends("baum_welch")`
lists; which of them can run depends on the machine, as [backends.md](backends.md) shows.

Which phase is *fastest* is a property of the model rather than of the phase: the parallel
ones split a single timestep across its cells, so at small `S` a parallel region or a device
launch costs more than the work inside it. At `S <= 8` over a 1,268-symbol corpus, `cython`
was the fastest phase, `cpu_parallel` took 18-24 ms and `cuda` about 500 ms per corpus
forward pass. Since those four agree to the bit, choosing among them is a speed choice and
nothing else. `torch` is the one whose result can differ, and on the same fixtures a topology
search run on it chose the same moves while costing 350-400 times more on the CPU. Those
figures are dated observations on one host, and they live with their host in
[`docs/benchmarks/hmm-backends.md`](../../benchmarks/hmm-backends.md).

### The convergence check has its own backend

`backend=` picks the E-step. The stopping rule needs one more forward pass -- the corpus
description length at the end of every batch of cycles, which is also the last entry of
`description_lengths` -- and that one is chosen by `score_backend=`, defaulting to
`"python"` ([ADR 0024](../../design/adr/0024-search-compiled-work.md) section 2). It takes
the names `forward_backward` has, and is validated before any work.

Those four phases are bit-identical, so it changes how fast the check runs and nothing it
returns:

```python
>>> checked = baum_welch(params, ds, backend="cython", score_backend="cython")
>>> checked.description_lengths == result.description_lengths and checked.cycles == result.cycles
True
```

`"torch"` is not among them. `forward_backward` has no torch phase, and nothing is
substituted for one ([ADR 0021](../../design/adr/0021-runtime-backend-selection.md)), so
asking for it is an error rather than a silent swap:

```python
>>> baum_welch(params, ds, score_backend="torch")
ValueError: baum_welch's score_backend has no 'torch' backend: that lifecycle phase is not implemented for it. It has ['python', 'cython', 'cpu_parallel', 'cuda']
```

Keeping the two names apart is what lets a torch E-step be checked by a compiled forward
pass:

```python
>>> mixed = baum_welch(params, ds, backend="torch", score_backend="cython")
>>> mixed.cycles == result.cycles
True
```

On a single training run the check is a small part of the cost. It dominates a **topology
search**, which re-converges a candidate model per trial: one measured round took 9.38 s
with the check on numpy and 1.70 s with it on Cython. The search is not public yet; when it
is, the guidance above applies to it unchanged, since it reaches these kernels through the
same two names.

## Errors

Every error is raised before the starting model is re-estimated, so no partial result is
lost. Arguments and codes are checked before any E-step runs; a record with no path of
finite description length under the starting model is found by the first E-step, and
cannot be trained on, since all of its counts are zero:

```python
>>> impossible = SequenceRecord(np.array([USER_BASE, 1]))
>>> baum_welch(params, [impossible])
ImpossibleSequenceError: record 0 has no path of finite description length under the starting model of 2 state(s), so every one of its expected counts is zero and it cannot be trained on
```

Code 1 is `UNK`, whose fibres are zero in every model, so no path emits it. A code outside
the model's symbol axis, a corpus with no symbols, and a stopping rule that cannot stop
(`batch_cycles` or `patience` below 1, `change_bits` not positive) raise `ValueError`.

So do a negative `max_cycles` or `min_cycles`, a `batch_size` below 1 and a device the backend cannot use. `device=` names a torch
device, so only `"torch"` takes one other than the CPU; `"cuda"` runs on numba-cuda's current
device and takes none at all, so `backend="cuda", device="cuda"` raises `ValueError` where a
device exists and `BackendUnavailableError` where none does. A device name must be a string:

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
