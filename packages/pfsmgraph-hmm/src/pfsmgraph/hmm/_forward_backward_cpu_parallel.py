# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cython.pyx sha256:a3dc5d01a67ce825dddcae92bc297999131531df19620713c1bd823126470feb
"""Forward-backward and the batched E-step, ADR 0002 phase 3: Numba CPU-parallel, ``prange``.

A **mechanical** translation of ``_forward_backward_cython.pyx``, which is itself a
mechanical translation of ``_forward_backward.py``. Phase 1 stays the oracle, and this
phase is held to it **bit for bit**, like phase 2: where any of the three could differ
they are made to agree, and a disagreement is a defect in the chain, never a tolerance
to widen. The specification is ``docs/design/algorithms/forward_backward/FORMALIZATION.md``.

**The parallel axis is read from the formalization, not chosen here: each timestep's
``(record, state)`` cells, with ``t`` serial and every reduction serial.** So ``t`` is
the outer loop, in both passes and in the counts, and ``prange`` runs over the ``B·S``
cells of one step, flattened, as the batched Viterbi kernel does:

- forward: cell ``(b, j)`` reads only ``alpha[b, t - 1]`` and ``w`` and writes only
  ``column[b, j]``; then ``Q[b, t]`` per record, a serial sum over ``j``; then
  ``alpha[b, t, j] = column[b, j] / Q[b, t]`` per cell;
- backward: cell ``(b, i)`` reads only ``beta[b, t + 1]`` and writes only ``beta[b, t, i]``;
- counts: cell ``(b, i)`` owns ``transition_counts[b, i]``, ``emission_counts[b, i]`` and
  ``init_counts[b, i]``, each receiving exactly one addition per position.

No two iterations of any ``prange`` touch the same element, so the loops are race-free by
construction rather than by scheduling. That is also phase 4's launch geometry, which is
the reason for it (ADR 0016 scopes this phase to rehearsing the decomposition CUDA
reuses), and the formalization records that whole records would be faster on a CPU.

**No reduction is parallel, and that is the whole difference from a Viterbi phase 3.**
A ``min`` gives the same result in any order; a float sum does not. A ``prange`` that
numba recognised as a sum reduction would combine per-thread partial sums in an order
set by the thread count, moving the last bit of about half of all scale factors. So every
accumulator here is a local of one cell, folded in a ``range`` loop from the first term
exactly as ``np.add.accumulate`` defines it, and ξ is grouped ``(alpha * w) * beta``.
``test_forward_backward_backends.py`` compares on ``tobytes()`` at one thread and at the
most this process has.

**No multiply-add is fused.** ``acc + a * b`` is the shape an FMA instruction fuses, and
numba compiles for the host CPU, which here has FMA. It does not fuse without the
``contract`` fast-math flag, and these kernels never set ``fastmath``: ADR 0020 measured
2000 of 2000 unfused, and TC-21 is the constructed case that would see a regression.

**Division is** ``safe_divide`` **'s**, ``x / q`` when ``q != 0.0`` and ``0.0`` otherwise.
The guard runs first, so numba's Python error model never meets a zero divisor.

**Padding follows the formalization's contract.** A padded step's scale factor is exactly
``1.0``; β is seeded from each record's own ``1 / Q[b, n]``; a padded position adds
nothing to any count. α past ``n`` and β past ``n`` are never read, so they are not
written, and the buffers are allocated with ``np.empty``.

**The per-record kernel is the batch of one.** ``_forward_backward`` runs the batched
recurrences with ``B = 1`` and ``n = N``, so every row it returns is written.

**Nothing here validates and nothing raises**, which nopython mode makes a constraint as
well as a discipline. **No ``cache=True``**, for the reason the Viterbi phase gives: it
interacts badly with ``parallel=True`` while a source is moving.

**The parallelism is thin.** ``t`` is strictly sequential, so a parallel region opens and
closes several times per timestep around ``O(B·S²)`` work, and at small ``S`` and ``B``
this kernel is expected to be slower than phase 2. ADR 0016 scopes phase 3 to parallel
correctness, so that is the phase working as specified.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._numeric import bits


@njit(parallel=True)
def _recurrences(init, w, sym, lengths, alpha, beta, scale, column):
    """α, β and the scale factors of every record, one timestep at a time.

    The caller owns every buffer: ``alpha`` and ``beta`` ``(B, L + 1, S)``, ``scale``
    ``(B, L + 1)``, ``column`` ``(B, S)`` scratch. Writes rows ``0 .. n[b]`` of α and β
    and every column of ``scale``.
    """
    batch = sym.shape[0]
    width = sym.shape[1]
    size = init.shape[0]

    for b in range(batch):
        for j in range(size):
            alpha[b, 0, j] = init[j]
        scale[b, 0] = 1.0

    for t in range(1, width + 1):
        # The parallel loop over the (record, destination) cells of step t.
        for cell in prange(batch * size):
            b = cell // size
            j = cell % size
            if t <= lengths[b]:
                k = sym[b, t - 1]
                # Serial, ascending over the source i, from the first term.
                acc = alpha[b, t - 1, 0] * w[0, j, k]
                for i in range(1, size):
                    acc = acc + alpha[b, t - 1, i] * w[i, j, k]
                column[b, j] = acc

        # Each record's scale factor, a serial ascending sum over j. Parallel across
        # records only: q is a local of one iteration, never a prange reduction.
        for b in prange(batch):
            if t <= lengths[b]:
                q = column[b, 0]
                for j in range(1, size):
                    q = q + column[b, j]
                scale[b, t] = q
            else:
                # A padded step: exactly 1, whose bits add nothing.
                scale[b, t] = 1.0

        for cell in prange(batch * size):
            b = cell // size
            j = cell % size
            if t <= lengths[b]:
                q = scale[b, t]
                alpha[b, t, j] = column[b, j] / q if q != 0.0 else 0.0

    # Each record's β starts from its own column n, not column L.
    for b in prange(batch):
        n = lengths[b]
        q = scale[b, n]
        for i in range(size):
            beta[b, n, i] = 1.0 / q if q != 0.0 else 0.0

    for t in range(width - 1, -1, -1):
        # The parallel loop over the (record, source) cells of step t.
        for cell in prange(batch * size):
            b = cell // size
            i = cell % size
            if t < lengths[b]:
                k = sym[b, t]
                # Serial, ascending over the destination j, from the first term.
                acc = w[i, 0, k] * beta[b, t + 1, 0]
                for j in range(1, size):
                    acc = acc + w[i, j, k] * beta[b, t + 1, j]
                q = scale[b, t]
                beta[b, t, i] = acc / q if q != 0.0 else 0.0


@njit(parallel=True)
def _counts(alpha, beta, w, sym, present, lengths, init_counts, transition_counts, emission_counts):
    """Every record's three count arrays, adding each position's ξ as it is formed.

    The caller zeroes the counts. Positions stay serial and ascending; within one, the
    ``(record, source)`` cells run in parallel, since each owns its own count rows.
    """
    batch = sym.shape[0]
    width = sym.shape[1]
    size = w.shape[0]

    for t in range(width):
        for cell in prange(batch * size):
            b = cell // size
            i = cell % size
            if t < lengths[b]:
                k = sym[b, t]
                symbol = present[k]
                row = 0.0
                for j in range(size):
                    # Grouped as phase 1 groups it: (alpha * w) * beta.
                    xi = (alpha[b, t, i] * w[i, j, k]) * beta[b, t + 1, j]
                    if t == 0:
                        # xi_0 summed over j, ascending, from the first term.
                        row = xi if j == 0 else row + xi
                    transition_counts[b, i, j] = transition_counts[b, i, j] + xi
                    emission_counts[b, i, j, symbol] = emission_counts[b, i, j, symbol] + xi
                if t == 0:
                    init_counts[b, i] = row


def _arc_table(transition_p, output_p, codes):
    """``(present, sym, w)`` exactly as phase 2's helper builds them.

    One numpy multiply per element for ``w``, so its bits are phase 1's; ``sym`` has the
    shape of ``codes``. Contiguous and freshly allocated, so numba sees one array type
    whatever the caller passed, frozen parameters included.
    """
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)
    present, sym = np.unique(codes_c, return_inverse=True)
    w = np.ascontiguousarray(trans_c[:, :, np.newaxis] * out_c[:, :, present])
    sym_c = np.ascontiguousarray(sym.reshape(codes_c.shape), dtype=np.int64)
    return np.ascontiguousarray(present, dtype=np.int64), sym_c, w


def _seed(init_state_p):
    """A writable contiguous float64 copy: one numba type, and the same bits."""
    return np.array(init_state_p, dtype=np.float64, order="C", copy=True)


def _forward_backward(init_state_p, transition_p, output_p, codes):
    """The recurrences: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    The signature and result are ``_forward_backward.py``'s: ``(alpha, beta, scale)``,
    shapes ``(N + 1, S)``, ``(N + 1, S)`` and ``(N + 1,)``, byte-identical to phase 1. A
    batch of one, whose single record fills the width, so every row is written.
    """
    init_c = _seed(init_state_p)
    _, sym_c, w = _arc_table(transition_p, output_p, np.asarray(codes)[np.newaxis])

    n = sym_c.shape[1]
    size = init_c.shape[0]
    lengths = np.array([n], dtype=np.int64)

    alpha = np.empty((1, n + 1, size), dtype=np.float64)
    beta = np.empty((1, n + 1, size), dtype=np.float64)
    scale = np.empty((1, n + 1), dtype=np.float64)
    column = np.empty((1, size), dtype=np.float64)

    _recurrences(init_c, w, sym_c, lengths, alpha, beta, scale, column)

    return alpha[0], beta[0], scale[0]


def _e_step_batch(init_state_p, transition_p, output_p, codes, lengths, device=None):
    """Per-record ``(counts, bits)`` for a padded batch: ``baum_welch``'s ``cpu_parallel`` kernel.

    The signature and result are ``_forward_backward.py``'s ``_e_step_batch``:
    ``((init_counts, transition_counts, emission_counts), bits)``, shapes ``(B, S)``,
    ``(B, S, S)``, ``(B, S, S, A)`` and ``(B,)``, every array byte-identical to phase 1's.
    ``device`` is part of the shared signature; ``baum_welch`` accepts only ``None`` or
    ``"cpu"`` for this backend, and nothing here reads it.

    The description length is phase 1's host expression over a ``(B, L + 1)`` scale array
    whose padded entries are exactly ``1.0``. Counts are per record; the sum over records
    is the caller's.
    """
    init_c = _seed(init_state_p)
    lengths_c = np.array(lengths, dtype=np.int64, order="C", copy=True)
    present, sym_c, w = _arc_table(transition_p, output_p, codes)

    batch, width = sym_c.shape
    size = init_c.shape[0]
    n_symbols = np.asarray(output_p).shape[2]

    alpha = np.empty((batch, width + 1, size), dtype=np.float64)
    beta = np.empty((batch, width + 1, size), dtype=np.float64)
    scale = np.empty((batch, width + 1), dtype=np.float64)
    column = np.empty((batch, size), dtype=np.float64)
    init_counts = np.zeros((batch, size), dtype=np.float64)
    transition_counts = np.zeros((batch, size, size), dtype=np.float64)
    emission_counts = np.zeros((batch, size, size, n_symbols), dtype=np.float64)

    _recurrences(init_c, w, sym_c, lengths_c, alpha, beta, scale, column)
    _counts(alpha, beta, w, sym_c, present, lengths_c, init_counts, transition_counts, emission_counts)

    bits_per_record = np.add.accumulate(bits(scale), axis=1)[:, -1]
    return (init_counts, transition_counts, emission_counts), bits_per_record
