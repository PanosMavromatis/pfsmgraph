# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cpu_parallel.py sha256:7c719a7326b62c8c524ad3c52afd6447c432db8f67140314176d1dc41b7cfe1d
"""Forward-backward and the batched E-step, ADR 0002 phase 4: Numba CUDA.

The decomposition is phase 3's, re-expressed rather than chosen again, and the result is
held to phase 1 **bit for bit** on the device, like every earlier phase. The specification
is ``docs/design/algorithms/forward_backward/FORMALIZATION.md``.

**Each timestep's ``(record, state)`` cells are device threads, and ``t`` is a sequence of
launches on the host.** ``cuda.syncthreads()`` synchronises one block, not a grid, so the
only grid-wide barrier is the end of a launch. A cell reads only row ``t - 1`` (forward) or
``t + 1`` (backward) of its own record and writes only its own entries, as in
``_forward_backward_cpu_parallel.py``; every reduction is a serial loop inside one thread,
folded from the first term in the formalization's order.

**NVVM fuses multiply-add by default, and that decided the kernel's shape.** Measured
2026-09-15 on an NVIDIA L4 with numba-cuda 0.30.4: a kernel computing ``c + a * b`` returned
the exactly rounded ``fma(a, b, c)`` in 20,000 of 20,000 checked cases and differed from the
host's unfused result in 24,081 of 200,000, **without** ``fastmath`` and still at
``opt=False``, so the fusion happens in code generation. ``cuda.jit`` builds its NVVM options
from ``fastmath``, ``opt`` and ``debug`` alone and offers no public ``fma=0``. So **no kernel
here both multiplies and adds a product**: every product is written to device memory by one
launch and summed by a later one, and two kernels cannot be fused into one instruction on
any NVVM. Storing the product and reading it back inside one kernel also measured unfused,
but it relies on the optimiser not forwarding the stored value, and was declined for that.
This settles the question ADR 0020 left open; the formalization's TC-21 is the constructed
case that sees a regression, and it runs here on the device.

One timestep therefore takes these launches, each over cells unless marked:

- forward: ``_forward_terms`` writes ``terms[b, j, i] = alpha[b, t - 1, i] * w[i, j, k]``;
  ``_forward_fold`` folds them over ``i`` into ``column[b, j]``; ``_scale``, one thread per
  record, folds ``column[b]`` over ``j`` into ``scale[b, t]``; ``_normalise`` divides;
- backward: ``_backward_terms`` writes ``terms[b, i, j] = w[i, j, k] * beta[b, t + 1, j]``;
  ``_backward_fold`` folds over ``j`` and divides by ``scale[b, t]``;
- counts: ``_xi`` writes ``terms[b, i, j] = (alpha[b, t, i] * w[i, j, k]) * beta[b, t + 1, j]``,
  products only; ``_add_counts`` adds each into its count cells and forms ``init_counts``.

A division is never contracted with anything, so ``_normalise`` and ``_backward_fold`` divide
the sums they read. **The cost is launches**: about eight per timestep, so this phase pays
off only where ``B·S²`` per step is large. That is a measurement for goal 4's benchmark, not
a defect.

**The host builds** ``w`` **and takes every logarithm**, as every phase does: ``w`` is one
numpy multiply per element, uploaded once, and the description length is ``bits`` of the
downloaded scale factors, summed on the host in phase 1's expression.

**Padding follows the formalization's contract.** ``scale`` is uploaded as ones, so a padded
step's factor is exactly ``1.0`` and only live steps overwrite it; β is seeded from each
record's own column ``n``; a padded position launches threads that return without writing.
α and β past ``n`` are never read and never written. A batch of width 0 crosses no arc and
launches nothing: its seeds are computed on the host.

**Nothing on the device validates or raises**, and here the platform enforces it.

**This module refuses to import without a device**, for the reason ``_viterbi_cuda.py``
gives: ``@cuda.jit`` decorates lazily, so the import would otherwise succeed on a machine
with no GPU and the backend table would report a backend that fails on first call. Raising
:class:`ImportError` is what makes a missing device the loud skip the table's
``needs="CUDA device"`` names; ``PFSMGRAPH_REQUIRE_BACKENDS=cuda`` makes it a failure.

**The grid is small, and numba-cuda warns about that per launch configuration.** The
warning is filtered for that message alone, locally, as ``_viterbi_cuda.py`` filters it; the
class is ``numba.cuda.core.errors.NumbaPerformanceWarning``, which a filter on numba's own
class of that name does not match.
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import cuda
from numba.cuda.core.errors import NumbaPerformanceWarning

from ._numeric import bits

if not cuda.is_available():
    raise ImportError(
        "no CUDA device detected: pfsmgraph.hmm._forward_backward_cuda needs a working "
        "CUDA device and driver, and numba-cuda reports none"
    )

#: Threads per block. A placeholder: no occupancy tuning before the answer is right.
_THREADS_PER_BLOCK = 64


# --- forward -------------------------------------------------------------------------


@cuda.jit
def _seed_alpha(init, alpha):
    """``alpha[b, 0, j] = init[j]``: stores only."""
    cell = cuda.grid(1)
    size = alpha.shape[2]
    if cell >= alpha.shape[0] * size:
        return
    alpha[cell // size, 0, cell % size] = init[cell % size]


@cuda.jit
def _forward_terms(t, w, sym, lengths, alpha, terms):
    """Products only: ``terms[b, j, i] = alpha[b, t - 1, i] * w[i, j, k]``."""
    cell = cuda.grid(1)
    size = alpha.shape[2]
    if cell >= sym.shape[0] * size:
        return
    b = cell // size
    j = cell % size
    if t <= lengths[b]:
        k = sym[b, t - 1]
        for i in range(size):
            terms[b, j, i] = alpha[b, t - 1, i] * w[i, j, k]


@cuda.jit
def _forward_fold(t, lengths, terms, column):
    """Sums only: ``column[b, j]``, ascending over ``i``, from the first term."""
    cell = cuda.grid(1)
    size = column.shape[1]
    if cell >= column.shape[0] * size:
        return
    b = cell // size
    j = cell % size
    if t <= lengths[b]:
        acc = terms[b, j, 0]
        for i in range(1, size):
            acc = acc + terms[b, j, i]
        column[b, j] = acc


@cuda.jit
def _scale(t, lengths, column, scale):
    """One thread per record: ``scale[b, t]``, ascending over ``j``. Padded steps keep 1."""
    b = cuda.grid(1)
    if b >= column.shape[0]:
        return
    if t <= lengths[b]:
        q = column[b, 0]
        for j in range(1, column.shape[1]):
            q = q + column[b, j]
        scale[b, t] = q


@cuda.jit
def _normalise(t, lengths, column, scale, alpha):
    """Division only: ``alpha[b, t, j] = column[b, j] / scale[b, t]``, or 0 over a zero."""
    cell = cuda.grid(1)
    size = column.shape[1]
    if cell >= column.shape[0] * size:
        return
    b = cell // size
    j = cell % size
    if t <= lengths[b]:
        q = scale[b, t]
        if q != 0.0:
            alpha[b, t, j] = column[b, j] / q
        else:
            alpha[b, t, j] = 0.0


# --- backward ------------------------------------------------------------------------


@cuda.jit
def _seed_beta(lengths, scale, beta):
    """One thread per record: ``beta[b, n, i] = 1 / scale[b, n]``, or 0 over a zero."""
    b = cuda.grid(1)
    if b >= beta.shape[0]:
        return
    n = lengths[b]
    q = scale[b, n]
    for i in range(beta.shape[2]):
        if q != 0.0:
            beta[b, n, i] = 1.0 / q
        else:
            beta[b, n, i] = 0.0


@cuda.jit
def _backward_terms(t, w, sym, lengths, beta, terms):
    """Products only: ``terms[b, i, j] = w[i, j, k] * beta[b, t + 1, j]``."""
    cell = cuda.grid(1)
    size = beta.shape[2]
    if cell >= sym.shape[0] * size:
        return
    b = cell // size
    i = cell % size
    if t < lengths[b]:
        k = sym[b, t]
        for j in range(size):
            terms[b, i, j] = w[i, j, k] * beta[b, t + 1, j]


@cuda.jit
def _backward_fold(t, lengths, terms, scale, beta):
    """Sums, then a division: ``beta[b, t, i]``, ascending over ``j``, over ``scale[b, t]``."""
    cell = cuda.grid(1)
    size = beta.shape[2]
    if cell >= beta.shape[0] * size:
        return
    b = cell // size
    i = cell % size
    if t < lengths[b]:
        acc = terms[b, i, 0]
        for j in range(1, size):
            acc = acc + terms[b, i, j]
        q = scale[b, t]
        if q != 0.0:
            beta[b, t, i] = acc / q
        else:
            beta[b, t, i] = 0.0


# --- counts --------------------------------------------------------------------------


@cuda.jit
def _xi(t, w, sym, lengths, alpha, beta, terms):
    """Products only: ``terms[b, i, j] = (alpha[b, t, i] * w[i, j, k]) * beta[b, t + 1, j]``."""
    cell = cuda.grid(1)
    size = alpha.shape[2]
    if cell >= sym.shape[0] * size:
        return
    b = cell // size
    i = cell % size
    if t < lengths[b]:
        k = sym[b, t]
        for j in range(size):
            terms[b, i, j] = (alpha[b, t, i] * w[i, j, k]) * beta[b, t + 1, j]


@cuda.jit
def _add_counts(t, sym, present, lengths, terms, init_counts, transition_counts, emission_counts):
    """Sums only: each ξ into its count cells, and ``init_counts`` from ξ at ``t = 0``."""
    cell = cuda.grid(1)
    size = init_counts.shape[1]
    if cell >= sym.shape[0] * size:
        return
    b = cell // size
    i = cell % size
    if t < lengths[b]:
        symbol = present[sym[b, t]]
        for j in range(size):
            xi = terms[b, i, j]
            transition_counts[b, i, j] = transition_counts[b, i, j] + xi
            emission_counts[b, i, j, symbol] = emission_counts[b, i, j, symbol] + xi
        if t == 0:
            row = terms[b, i, 0]
            for j in range(1, size):
                row = row + terms[b, i, j]
            init_counts[b, i] = row


# --- host ----------------------------------------------------------------------------


def _arc_table(transition_p, output_p, codes):
    """``(present, sym, w)`` exactly as the earlier phases' helper builds them."""
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)
    present, sym = np.unique(codes_c, return_inverse=True)
    w = np.ascontiguousarray(trans_c[:, :, np.newaxis] * out_c[:, :, present])
    sym_c = np.ascontiguousarray(sym.reshape(codes_c.shape), dtype=np.int64)
    return np.ascontiguousarray(present, dtype=np.int64), sym_c, w


def _grids(batch, size):
    """``(cells, records)`` launch configurations: ``B·S`` cells, and ``B`` records."""
    per_block = _THREADS_PER_BLOCK
    return (
        ((batch * size + per_block - 1) // per_block, per_block),
        ((batch + per_block - 1) // per_block, per_block),
    )


def _launch(kernel, grid, *args):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Grid size .* low occupancy", category=NumbaPerformanceWarning
        )
        kernel[grid](*args)


class _Host:
    """A host array standing in for a device one, so both paths download alike."""

    def __init__(self, array):
        self._array = array

    def copy_to_host(self):
        return self._array


def _no_launch(init_c, batch, width, size, n_symbols, with_counts):
    """A batch that crosses no arc: the seeds, computed on the host, and no launch.

    ``alpha[b, 0] = init``, ``scale`` all ones, and ``beta[b, 0] = 1 / 1``, which is what
    the seed launches would write; every count is zero. Nothing reaches the device, which
    also keeps a zero-size arc table from being uploaded.
    """
    alpha = np.empty((batch, width + 1, size), dtype=np.float64)
    alpha[:, 0] = init_c
    beta = np.empty((batch, width + 1, size), dtype=np.float64)
    beta[:, 0] = 1.0 / 1.0
    d = {
        "alpha": _Host(alpha),
        "beta": _Host(beta),
        "scale": _Host(np.ones((batch, width + 1), dtype=np.float64)),
    }
    if with_counts:
        d["init_counts"] = _Host(np.zeros((batch, size), dtype=np.float64))
        d["transition_counts"] = _Host(np.zeros((batch, size, size), dtype=np.float64))
        d["emission_counts"] = _Host(np.zeros((batch, size, size, n_symbols), dtype=np.float64))
    return d


def _run(init_state_p, transition_p, output_p, codes, lengths, with_counts):
    """Upload once, run the launches, and leave every result on the device."""
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    lengths_c = np.ascontiguousarray(lengths, dtype=np.int64)
    present, sym_c, w = _arc_table(transition_p, output_p, codes)
    batch, width = sym_c.shape
    size = init_c.shape[0]
    n_symbols = np.asarray(output_p).shape[2]
    if width == 0 or batch == 0:
        return _no_launch(init_c, batch, width, size, n_symbols, with_counts)
    cells, records = _grids(batch, size)

    d = {
        "w": cuda.to_device(w),
        "sym": cuda.to_device(sym_c),
        "present": cuda.to_device(present),
        "lengths": cuda.to_device(lengths_c),
        "alpha": cuda.device_array((batch, width + 1, size), dtype=np.float64),
        "beta": cuda.device_array((batch, width + 1, size), dtype=np.float64),
        # Ones: scale[b, 0] and every padded step are exactly 1, and only live steps
        # are overwritten.
        "scale": cuda.to_device(np.ones((batch, width + 1), dtype=np.float64)),
        "column": cuda.device_array((batch, size), dtype=np.float64),
        "terms": cuda.device_array((batch, size, size), dtype=np.float64),
    }
    if with_counts:
        d["init_counts"] = cuda.to_device(np.zeros((batch, size), dtype=np.float64))
        d["transition_counts"] = cuda.to_device(np.zeros((batch, size, size), dtype=np.float64))
        d["emission_counts"] = cuda.to_device(np.zeros((batch, size, size, n_symbols), dtype=np.float64))

    _launch(_seed_alpha, cells, cuda.to_device(init_c), d["alpha"])
    for t in range(1, width + 1):
        _launch(_forward_terms, cells, t, d["w"], d["sym"], d["lengths"], d["alpha"], d["terms"])
        _launch(_forward_fold, cells, t, d["lengths"], d["terms"], d["column"])
        _launch(_scale, records, t, d["lengths"], d["column"], d["scale"])
        _launch(_normalise, cells, t, d["lengths"], d["column"], d["scale"], d["alpha"])

    _launch(_seed_beta, records, d["lengths"], d["scale"], d["beta"])
    for t in range(width - 1, -1, -1):
        _launch(_backward_terms, cells, t, d["w"], d["sym"], d["lengths"], d["beta"], d["terms"])
        _launch(_backward_fold, cells, t, d["lengths"], d["terms"], d["scale"], d["beta"])

    if with_counts:
        for t in range(width):
            _launch(_xi, cells, t, d["w"], d["sym"], d["lengths"], d["alpha"], d["beta"], d["terms"])
            _launch(
                _add_counts, cells, t, d["sym"], d["present"], d["lengths"], d["terms"],
                d["init_counts"], d["transition_counts"], d["emission_counts"],
            )
    return d


def _forward_backward(init_state_p, transition_p, output_p, codes):
    """The recurrences: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    The signature and result are ``_forward_backward.py``'s: ``(alpha, beta, scale)``,
    shapes ``(N + 1, S)``, ``(N + 1, S)`` and ``(N + 1,)``, byte-identical to phase 1. A
    batch of one, whose record fills the width, so every row is written.
    """
    codes = np.asarray(codes)
    d = _run(init_state_p, transition_p, output_p, codes[np.newaxis], [codes.shape[0]], False)
    return d["alpha"].copy_to_host()[0], d["beta"].copy_to_host()[0], d["scale"].copy_to_host()[0]


def _e_step_batch(init_state_p, transition_p, output_p, codes, lengths, device=None):
    """Per-record ``(counts, bits)`` for a padded batch: ``baum_welch``'s ``cuda`` kernel.

    The signature and result are ``_forward_backward.py``'s ``_e_step_batch``, every array
    byte-identical to phase 1's. It runs on numba-cuda's current device; ``device`` is part
    of the shared signature, which ``baum_welch`` validates, and nothing here reads it.
    """
    d = _run(init_state_p, transition_p, output_p, codes, lengths, True)
    scale = d["scale"].copy_to_host()
    bits_per_record = np.add.accumulate(bits(scale), axis=1)[:, -1]
    counts = (
        d["init_counts"].copy_to_host(),
        d["transition_counts"].copy_to_host(),
        d["emission_counts"].copy_to_host(),
    )
    return counts, bits_per_record
