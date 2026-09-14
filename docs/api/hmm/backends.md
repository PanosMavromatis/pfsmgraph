# Backends

`backend=`, `backends`, `BackendStatus` and `BackendUnavailableError`: how a caller chooses
which implementation of a dynamic-programming kernel runs, and how it finds out what can run
here. See [README.md](README.md) for the contracts, and [viterbi.md](viterbi.md) for the
decode these examples call.

The decode has up to four implementations, one per lifecycle phase
([ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md)):

| Name | Implementation | Needs |
|---|---|---|
| `"python"` | pure Python and numpy, the reference | nothing |
| `"cython"` | a compiled Cython extension | a platform wheel, or a source build |
| `"cpu_parallel"` | Numba, parallel over the states of one timestep | the `cpu-parallel` extra |
| `"cuda"` | Numba CUDA | the `gpu` extra and a CUDA device |

Training has two, and they are not lifecycle phases of each other
([baum_welch.md](baum_welch.md)):

| Name | Implementation | Needs |
|---|---|---|
| `"python"` | explicit forward and backward passes, numpy, the reference | nothing |
| `"torch"` | the forward pass, with counts as reverse-mode gradients | the `torch` extra |

Every example on this page runs against the same model and record:

```python
import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceDataset, SymbolTable
from pfsmgraph.hmm import BackendStatus, BackendUnavailableError, HMMParams, backends, viterbi

vocab = SymbolTable(["a", "b"])

output_p = np.zeros((2, 2, vocab.size))
output_p[0, 0, USER_BASE:] = [0.9, 0.1]
output_p[0, 1, USER_BASE:] = [0.2, 0.8]
output_p[1, 0, USER_BASE:] = [0.5, 0.5]
output_p[1, 1, USER_BASE:] = [0.1, 0.9]

params = HMMParams([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], output_p, vocab)
record = SequenceDataset.from_symbols([["a", "a", "b", "b"]], vocab, labels=["s1"])[0]
```

**The examples are chosen so their output is the same on every machine.** Which backends
*can run* differs between a laptop and a GPU server, so nothing below prints availability
itself. It is the one question this page cannot answer for you; `backends()` can.

## `backend=`

```python
viterbi(params: HMMParams, record: SequenceRecord, *, backend: BackendName = "python") -> ViterbiPath
```

A keyword-only string naming the implementation to run. **The default is `"python"`**, the
reference, present in every install, and nothing about the environment changes it: installing
an extra or plugging in a GPU never changes what an unqualified call runs.

```python
>>> viterbi(params, record, backend="python")
ViterbiPath(n_symbols=4, total_bits=5.0180, label='s1')
```

**Every decode backend computes the same function.** All four take their logarithms on the host with
numpy and then only add and compare, so on one machine they return identical states and
identical `total_bits`, not merely close ones:

```python
>>> len({tuple(viterbi(params, record, backend=s.name).states) for s in backends("viterbi") if s.available})
1
```

That is agreement on one host. numpy's vectorised `log2` can round differently across CPUs,
so the last bit of `total_bits` is not promised to match between machines.

**The name is checked before any work.** A string that is not a backend raises `ValueError`:

```python
>>> viterbi(params, record, backend="numba")
ValueError: backend must be one of ['python', 'cython', 'cpu_parallel', 'cuda', 'torch'], got 'numba'
```

## `BackendUnavailableError`

A backend that exists but cannot run here raises `BackendUnavailableError`, also before any
work, with a message naming what is missing and what supplies it. **Nothing falls back.** A
call that asked for `"cuda"` on a machine without a device fails; it never quietly runs on the
CPU, because a result whose backend depends on the environment is one nobody can reproduce
([ADR 0021](../../design/adr/0021-runtime-backend-selection.md)).

It subclasses `ImportError`, since that is what it is, so code that already guards an optional
import catches it:

```python
>>> issubclass(BackendUnavailableError, ImportError)
True
```

The messages, which depend on what this machine lacks and so are not executed here:

| Missing | Message ends with |
|---|---|
| the compiled extension | `this install of pfsmgraph-hmm has no compiled extension (a pure wheel, or a build with -Dcompiled=false): install a platform wheel, or build from source with a C compiler` |
| numba | `numba is not installed: pip install 'pfsmgraph-hmm[cpu-parallel]'` |
| numba-cuda | `numba-cuda is not installed: pip install 'pfsmgraph-hmm[gpu]'` |
| a CUDA device | `no CUDA device detected` |

A dependency that is installed but fails to import for another reason, such as a stale build
or an incompatible numpy, reports that import error itself instead, because reinstalling the
extra would not fix it.

A caller who wants a fallback writes one, and has then chosen it:

```python
try:
    path = viterbi(params, record, backend="cuda")
except BackendUnavailableError:
    path = viterbi(params, record)
```

## `backends`

```python
backends(name: str) -> tuple[BackendStatus, ...]
```

Every backend the public call `name` has, **available or not**, in lifecycle-phase order:

```python
>>> [s.name for s in backends("viterbi")]
['python', 'cython', 'cpu_parallel', 'cuda']
>>> [s.name for s in backends("baum_welch")]
['python', 'torch']
>>> backends("viterbi")[0]
BackendStatus(name='python', available=True, reason=None)
```

The order is by phase, not by speed, and nothing ranks the backends: which is fastest depends
on the model's size and on the machine, and a decode on a small model can be hundreds of times
slower on the GPU than on the Cython backend.

An unavailable backend's `reason` is the same text `BackendUnavailableError` would carry, so
asking in advance and asking by calling give one answer. **Branch on `available`, not on
`reason`**: the reason is written for a person and may be reworded.

The first call probes every backend, which can import numba and query for a CUDA device; the
result is cached for the life of the process, and later `backend=` calls read the same cache.
A device that appears while the process runs is not noticed. `import pfsmgraph.hmm` itself
imports no backend.

`name` is a public call. Kernels that no public call exposes yet have no entry:

```python
>>> backends("forward_backward")
ValueError: no public call 'forward_backward' has backends; known: ['baum_welch', 'viterbi']
```

## `BackendStatus`

```python
BackendStatus(name: str, available: bool, reason: str | None = None)
```

A frozen dataclass: the value to pass as `backend=`, whether that call would run here, and
why not when it would not.

## Extras

The compiled backends' dependencies are optional, so the base install stays lean
([ADR 0004](../../design/adr/0004-gpu-backends-and-optional-dependency-strategy.md)):

```bash
pip install 'pfsmgraph-hmm[cpu-parallel]'   # numba
pip install 'pfsmgraph-hmm[gpu]'            # numba-cuda, numpy<2.5
pip install 'pfsmgraph-hmm[torch]'          # torch, for baum_welch
```

They are two extras rather than one because the CPU-parallel backend needs no device.
`gpu` happens to install numba too, since numba-cuda depends on it. "GPU" here means
numba-cuda, unrelated to the `torch` stack `pfsmgraph-dl` uses. `gpu` caps numpy below 2.5
because numba-cuda 0.30.4 fails to import against numpy 2.5, and it installs nothing on macOS,
where numba-cuda publishes no wheel; `backends()` says so. Neither extra names a CUDA toolkit,
since that choice belongs to the machine's driver.

`torch` is a third extra, for `baum_welch`'s `"torch"` backend, which runs in float64 on the
CPU unless `baum_welch`'s `device=` names another torch device. It is unrelated to `gpu`, which
supplies numba-cuda, not torch's CUDA. **A `torch` result is not bit-identical
to the reference**, and `BackendStatus` carries no tolerance: each expected count agrees within
`N · eps · max(1, count)` and each description length within `N · eps · max(1, bits)`, as
[baum_welch.md](baum_welch.md) shows.
