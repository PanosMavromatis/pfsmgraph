# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward.py sha256:e0d40910d14697f3b3da3aeddb1a1f9f72f03cd76c9e5659690b80dbcce1eb72
"""Forward-backward and the batched E-step, ADR 0002 phase 2: compiled, single-threaded Cython.

A **mechanical** translation of ``_forward_backward.py``, held to it **bit for bit**, not
within a tolerance. Every line here traces back to a line there. The phase-1 module stays
on disk and stays the reference; this file never shadows it, which is what lets ADR 0003
compare the two. The specification both implement is
``docs/design/algorithms/forward_backward/FORMALIZATION.md``.

**The evaluation order is the contract, and it is why this file is written the way it
is** [ADR 0020]. Phase 1 reduces with ``np.add.accumulate(...)[-1]``, whose definition is
``r[0] = x[0]``, ``r[k] = r[k-1] + x[k]``. Every sum below is that fold, written out: the
accumulator starts **from the first term, not from 0.0**, and adds the rest in ascending
index. Starting from ``0.0`` would give the same bits for every non-negative term and
differ only in the sign of an all-``-0.0`` sum. Starting from the first term is identical
without the argument. The orders are phase 1's:

- α over the source ``i`` ascending, for each destination ``j``; then ``Q`` over ``j``;
- β over the destination ``j`` ascending, for each source ``i``;
- ``init_counts`` over ``j``; each count cell over positions ``t`` ascending.

**Every product associates as phase 1's does.** Forward and backward terms have two
factors, ``alpha * w`` and ``w * beta``. ξ has three and is ``(alpha * w) * beta``, grouped
as written, since ``alpha * (w * beta)`` differs at the last bit.

**No multiply-add is fused** [ADR 0020 §4]. ``acc + a * b`` is exactly the shape a
contracting compiler turns into one FMA instruction, which rounds differently. The
extension is built with ``-ffp-contract=off`` (``meson.build``). Baseline x86-64 has no FMA
to contract into anyway, but a ``-march=native`` build, aarch64 GCC and Apple clang do.
The formalization's TC-21 is the constructed case that would see it.

**Emission is on the arc** [ADR 0015]. The wrapper builds
``w[i, j, u] = transition_p[i, j] * output_p[i, j, present[u]]`` on the host, with numpy,
exactly as ``_arc_table`` does, and the kernel reads it with both endpoints inside the
reduction. It is phase 1's table, not a hoisted emission factor. The kernel performs only
``+``, ``*`` and ``/``; the only logarithm is :func:`bits` of the scale factors, on the
host, in phase 1's expression.

**Division is** ``safe_divide`` **'s**: ``x / q`` when ``q != 0.0`` and ``0.0`` otherwise,
for ``0/0`` and ``x/0`` alike. An impossible record therefore comes back as zero columns,
a zero β, zero counts and ``+inf`` bits, never ``NaN``.

**Nothing here validates and nothing raises.** That is the backend contract
``_forward_backward.py`` states, for the decode's reason: the CUDA phase implements the
same signatures, and a device function cannot raise.

**The batch is the per-record E-step, one record after another.** Phase 1 steps every
record through each position at once, because that is what numpy vectorises; C finishes a
record before starting the next, reusing one ``(L + 1, S)`` pair of α and β buffers. The
formalization makes loop order free, since nothing combines values across records. What
it makes contract is kept exactly: a padded step's scale factor is ``1.0``, β is seeded
from each record's own column ``n``, and ξ at a padded position contributes nothing, so
no count cell is touched there. α past ``n`` is never read, so it is not written.

.. [ADR 0015] ``docs/design/adr/0015-arc-emission-mealy-formulation.md``
.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

cimport cython
cimport numpy as cnp

import numpy as np

from ._numeric import bits


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _forward(
    const double[::1] init,
    const double[:, :, ::1] w,
    const cnp.int64_t[:, ::1] sym,
    Py_ssize_t b,
    Py_ssize_t n,
    double[:, ::1] alpha,
    double[:, ::1] scale,
    double[::1] column,
) noexcept nogil:
    """α and the scale factors of record ``b`` over its first ``n`` positions.

    Writes ``alpha[0 .. n]`` and ``scale[b, 0 .. n]``. ``column`` is scratch of length
    ``S``. The bound checks are off because every loop is bounded by a shape or by
    ``n <= L``, which ``pad_collate`` guarantees and the kernel does not re-check, and no
    index is ever negative. ``cdivision`` is on because every division is guarded by
    ``q != 0.0`` already: without it Cython adds its own zero check, which cannot fire,
    and which reacquires the GIL to raise ``ZeroDivisionError`` from a ``noexcept``
    function. Float division itself is the same IEEE operation either way.
    """
    cdef Py_ssize_t size = init.shape[0]
    cdef Py_ssize_t t, i, j, k
    cdef double acc, q

    # The seed is not normalised, and scale[0] is set rather than computed.
    for j in range(size):
        alpha[0, j] = init[j]
    scale[b, 0] = 1.0

    for t in range(1, n + 1):
        k = sym[b, t - 1]
        for j in range(size):
            # Ascending over the source i, from the first term.
            acc = alpha[t - 1, 0] * w[0, j, k]
            for i in range(1, size):
                acc = acc + alpha[t - 1, i] * w[i, j, k]
            column[j] = acc
        # Ascending over j, after every column entry exists.
        q = column[0]
        for j in range(1, size):
            q = q + column[j]
        scale[b, t] = q
        for j in range(size):
            alpha[t, j] = column[j] / q if q != 0.0 else 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _backward(
    const double[:, :, ::1] w,
    const cnp.int64_t[:, ::1] sym,
    Py_ssize_t b,
    Py_ssize_t n,
    const double[:, ::1] scale,
    double[:, ::1] beta,
) noexcept nogil:
    """β of record ``b``, from its own ``1 / scale[b, n]`` down to position 0.

    Writes ``beta[0 .. n]``. Scaled by the same factors as α, so ξ needs no division
    by the likelihood.
    """
    cdef Py_ssize_t size = w.shape[0]
    cdef Py_ssize_t t, i, j, k
    cdef double acc, q

    q = scale[b, n]
    for i in range(size):
        beta[n, i] = 1.0 / q if q != 0.0 else 0.0

    for t in range(n - 1, -1, -1):
        k = sym[b, t]
        q = scale[b, t]
        for i in range(size):
            # Ascending over the destination j, from the first term.
            acc = w[i, 0, k] * beta[t + 1, 0]
            for j in range(1, size):
                acc = acc + w[i, j, k] * beta[t + 1, j]
            beta[t, i] = acc / q if q != 0.0 else 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _counts(
    const double[:, ::1] alpha,
    const double[:, ::1] beta,
    const double[:, :, ::1] w,
    const cnp.int64_t[:, ::1] sym,
    const cnp.int64_t[::1] present,
    Py_ssize_t b,
    Py_ssize_t n,
    double[:, ::1] init_counts,
    double[:, :, ::1] transition_counts,
    double[:, :, :, ::1] emission_counts,
) noexcept nogil:
    """Record ``b``'s three count arrays, adding each position's ξ as it is formed.

    The caller zeroes the counts. ξ is never stored: each cell is added into its own
    accumulators, one addition per position in ascending ``t``, which is phase 1's
    ``transition_counts += xi`` taken one element at a time.
    """
    cdef Py_ssize_t size = w.shape[0]
    cdef Py_ssize_t t, i, j, k, symbol
    cdef double xi, row

    for t in range(n):
        k = sym[b, t]
        symbol = present[k]
        for i in range(size):
            row = 0.0
            for j in range(size):
                # Grouped as phase 1 groups it: (alpha * w) * beta.
                xi = (alpha[t, i] * w[i, j, k]) * beta[t + 1, j]
                if t == 0:
                    # init_counts is xi_0 summed over j, ascending, from the first term.
                    row = xi if j == 0 else row + xi
                transition_counts[b, i, j] = transition_counts[b, i, j] + xi
                emission_counts[b, i, j, symbol] = emission_counts[b, i, j, symbol] + xi
            if t == 0:
                init_counts[b, i] = row


def _arc_table(transition_p, output_p, codes):
    """``(present, sym, w)`` as ``_forward_backward.py``'s helper of the same name builds them.

    ``w`` is one numpy multiply per element, so its bits are phase 1's by construction;
    the ``np.unique`` is over the whole array, so ``sym`` has the shape of ``codes``.
    Everything is made contiguous and typed for the memoryviews, which is the one thing
    this adds.
    """
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)
    present, sym = np.unique(codes_c, return_inverse=True)
    w = np.ascontiguousarray(trans_c[:, :, np.newaxis] * out_c[:, :, present])
    # int64, because np.unique's inverse is intp and the memoryview is typed.
    sym_c = np.ascontiguousarray(sym.reshape(codes_c.shape), dtype=np.int64)
    return np.ascontiguousarray(present, dtype=np.int64), sym_c, w


def _forward_backward(init_state_p, transition_p, output_p, codes):
    """The recurrences: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    The signature and result are ``_forward_backward.py``'s: ``(alpha, beta, scale)``,
    shapes ``(N + 1, S)``, ``(N + 1, S)`` and ``(N + 1,)``, byte-identical to phase 1.
    The wrapper builds the arc table on the host, allocates, and calls the kernel with a
    batch of one. It neither validates nor raises.
    """
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    _, sym_c, w = _arc_table(transition_p, output_p, np.asarray(codes)[np.newaxis])

    cdef Py_ssize_t n = sym_c.shape[1]
    cdef Py_ssize_t size = init_c.shape[0]

    alpha = np.empty((n + 1, size), dtype=np.float64)
    beta = np.empty((n + 1, size), dtype=np.float64)
    scale = np.empty((1, n + 1), dtype=np.float64)
    column = np.empty(size, dtype=np.float64)

    _forward(init_c, w, sym_c, 0, n, alpha, scale, column)
    _backward(w, sym_c, 0, n, scale, beta)

    return alpha, beta, scale[0]


def _e_step_batch(init_state_p, transition_p, output_p, codes, lengths, device=None):
    """Per-record ``(counts, bits)`` for a padded batch: ``baum_welch``'s ``cython`` kernel.

    The signature and result are ``_forward_backward.py``'s ``_e_step_batch``:
    ``((init_counts, transition_counts, emission_counts), bits)``, shapes ``(B, S)``,
    ``(B, S, S)``, ``(B, S, S, A)`` and ``(B,)``, every array byte-identical to phase 1's.
    ``device`` is part of the signature every backend shares; ``baum_welch`` accepts only
    ``None`` or ``"cpu"`` for this backend, and nothing here reads it.

    Counts are per record and never summed across the batch: the sum over records is the
    caller's, in record order. The description length is phase 1's host expression over a
    ``(B, L + 1)`` scale array whose padded entries are exactly ``1.0``, so it is phase 1's
    value to the bit, ``-0.0`` included.
    """
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    lengths_c = np.ascontiguousarray(lengths, dtype=np.int64)
    present, sym_c, w = _arc_table(transition_p, output_p, codes)

    cdef Py_ssize_t batch = sym_c.shape[0]
    cdef Py_ssize_t width = sym_c.shape[1]
    cdef Py_ssize_t size = init_c.shape[0]
    cdef Py_ssize_t n_symbols = np.asarray(output_p).shape[2]
    cdef Py_ssize_t b, t, n

    # One α and β buffer pair, reused record to record: rows past a record's n are
    # stale from an earlier record and are never read.
    alpha = np.empty((width + 1, size), dtype=np.float64)
    beta = np.empty((width + 1, size), dtype=np.float64)
    column = np.empty(size, dtype=np.float64)
    scale = np.empty((batch, width + 1), dtype=np.float64)
    init_counts = np.zeros((batch, size), dtype=np.float64)
    transition_counts = np.zeros((batch, size, size), dtype=np.float64)
    emission_counts = np.zeros((batch, size, size, n_symbols), dtype=np.float64)

    # Acquired once, so the loop below runs without touching a Python object.
    cdef const double[::1] init_v = init_c
    cdef const double[:, :, ::1] w_v = w
    cdef const cnp.int64_t[:, ::1] sym_v = sym_c
    cdef const cnp.int64_t[::1] present_v = present
    cdef const cnp.int64_t[::1] lengths_v = lengths_c
    cdef double[:, ::1] alpha_v = alpha
    cdef double[:, ::1] beta_v = beta
    cdef double[::1] column_v = column
    cdef double[:, ::1] scale_v = scale
    cdef double[:, ::1] init_counts_v = init_counts
    cdef double[:, :, ::1] transition_counts_v = transition_counts
    cdef double[:, :, :, ::1] emission_counts_v = emission_counts

    with nogil:
        for b in range(batch):
            n = lengths_v[b]
            _forward(init_v, w_v, sym_v, b, n, alpha_v, scale_v, column_v)
            # A padded step: the scale factor is exactly 1, whose bits add nothing.
            for t in range(n + 1, width + 1):
                scale_v[b, t] = 1.0
            _backward(w_v, sym_v, b, n, scale_v, beta_v)
            _counts(
                alpha_v, beta_v, w_v, sym_v, present_v, b, n,
                init_counts_v, transition_counts_v, emission_counts_v,
            )

    bits_per_record = np.add.accumulate(bits(scale), axis=1)[:, -1]
    return (init_counts, transition_counts, emission_counts), bits_per_record
