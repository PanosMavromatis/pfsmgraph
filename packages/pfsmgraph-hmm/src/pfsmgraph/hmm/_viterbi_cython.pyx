# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi.py sha256:6b26469bd36030a2cc2fcce8c9106deab18cf2851cf934551763d5238db2438a
"""The Viterbi decode, ADR 0002 phase 2: compiled, single-threaded Cython.

A **mechanical** translation of ``_viterbi.py``. Every line here traces back to a
line there, and where the two could differ they are made to agree rather than
merely to be close -- see *Bit-exactness* below. The phase-1 module stays on disk
and stays the reference; this file never shadows it, which is what lets ADR 0003
compare the two.

**The domain is bits, so this is a min-sum.** ``arc = -log2(p)`` grows as ``p``
falls, and the decode minimizes. A port that reaches for ``max`` inverts every
comparison; that is the single most likely defect in this file and the reason the
formalization states the objective as contract rather than description.

**Every logarithm is taken on the host, by** :func:`bits` **-- the function phase 1
uses -- and the compiled kernel performs only ``+`` and ``<``.** That is the
contract, not an optimisation. The wrapper computes ``seed = bits(init_p)`` and
the arc costs ``arc_bits = bits(trans_p[:, :, None] * out_p[:, :, present])`` in
phase 1's operation order (one float64 multiply, then one numpy ``-log2``), and
``_decode`` reads them. Addition and strict comparison are exact IEEE-754
operations, so the candidate values are phase 1's bit patterns by construction.
The tie-break is identical too: ``cand < best`` is strict, so an equal candidate
never displaces an earlier ``i``, which is exactly what ``np.argmin`` does when it
returns the first minimal index. The formalization makes "ties to the SMALLEST i"
contract, because two correct backends would otherwise legitimately disagree.

*Corrected 2026-09-14.* This file used to fuse ``-log2`` into the inner loop
through libc's scalar ``log2``, and claimed the result was bit-identical to phase
1 because the operation order matched. It was not: numpy's vectorised ``log2``
and libm's round one ulp apart in roughly 0.07-0.19% of probability-shaped inputs
on an AVX-512 host, concentrated on high-probability arcs, so ``total_bits``
could differ by an ulp and a near-tie could resolve differently. Matching the
operation order is not enough when the two sides call different functions. The
measurement is in the branch plan for ``fix/hmm-viterbi-log2``.

**Emission is on the arc** [ADR 0015]: the cost is ``arc_bits[i, j, k]``, read
inside the ``i`` loop because it depends on *both* endpoints. The table spans the
symbols present in the record, ``(S, S, U)`` with ``U = np.unique(codes)``, and
the record is re-indexed into it as ``sym``. That is not the ``(S, S, A)``
precompute phase 1 refuses: ``U <= min(N, A)`` bounds it by both the record and
the vocabulary, and the emission factor is still per arc and still read inside
both loops. What moved is only where the logarithm runs.

**An all-``+inf`` row decodes to state 0, matching numpy.** ``best`` starts at
``INFINITY`` and the comparison is strict, so when every candidate is ``+inf``
nothing ever displaces the initial ``best_i = 0``. ``np.argmin`` on an all-``inf``
row also returns 0. This is the impossible-sequence path, and the two backends
have to agree on it because the wrapper's error message names a position.

**Nothing here validates and nothing raises.** That is the backend contract
``_viterbi.py`` states, and it is why phases 2-4 are transliterations: a CUDA
device function cannot raise a Python exception. An impossible sequence comes back
as ``total_bits == inf``; ``viterbi()`` in ``_viterbi.py`` turns it into
``ImpossibleSequenceError``, for every backend, in one place.

.. [ADR 0015] ``docs/design/adr/0015-arc-emission-mealy-formulation.md``
"""

cimport cython
cimport numpy as cnp
from libc.math cimport INFINITY

import numpy as np

from ._numeric import bits


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _decode(
    const double[::1] seed,
    const double[:, :, ::1] arc_bits,
    const cnp.int64_t[::1] sym,
    double[:, ::1] delta,
    cnp.int64_t[:, ::1] psi,
    cnp.int64_t[::1] states,
) noexcept nogil:
    """The recurrence and the backtrace, entirely in C.

    Reads no Python object and allocates nothing, so the whole of it runs under
    ``nogil``. The caller owns every buffer; this writes ``delta``, ``psi`` and
    ``states`` in place.

    ``wraparound(False)`` is safe because no index here is ever negative: the
    backtrace counts down to 0 and stops, and every value read out of ``psi`` is
    a state index the forward pass wrote. ``boundscheck(False)`` is safe because
    every loop is bounded by the shape it indexes. Both were added only after the
    suite was green with them on, which is the order the phase-2 skill requires --
    turning them on first would hide an off-by-one as a wrong answer instead of an
    IndexError.
    """
    cdef Py_ssize_t n = sym.shape[0]
    cdef Py_ssize_t size = seed.shape[0]
    cdef Py_ssize_t t, i, j, k, best_i
    cdef double best, cand

    # Base case. delta[0, j] = bits(init_p[j]) -- the bit domain, which is
    # deviation 1 of the formalization and the whole of the seeding fix. The
    # original seeds a raw probability into a bit accumulator, inverting the
    # preference among start states; bits(0) is +inf here and absorbs, so an
    # impossible start stays impossible instead of becoming the best value. The
    # caller took the logarithm; this only copies it.
    for j in range(size):
        delta[0, j] = seed[j]

    # psi's row 0 is never read: the backtrace stops at psi[1]. It is zeroed by
    # the caller rather than left undefined, matching phase 1.

    for t in range(1, n + 1):
        k = sym[t - 1]
        for j in range(size):
            best = INFINITY
            best_i = 0
            for i in range(size):
                # Only + and < here: the arc cost was taken on the host, by the
                # same bits() phase 1 uses, so the candidate is phase 1's bit
                # pattern rather than a scalar-libm neighbour of it.
                cand = delta[t - 1, i] + arc_bits[i, j, k]
                if cand < best:
                    best = cand
                    best_i = i
            delta[t, j] = best
            psi[t, j] = best_i

    # Backtrace. states[N] is the argmin over the final row, ties to the
    # smallest j; then each predecessor is read straight out of psi.
    best = INFINITY
    best_i = 0
    for j in range(size):
        if delta[n, j] < best:
            best = delta[n, j]
            best_i = j
    states[n] = best_i

    for t in range(n - 1, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]


def _viterbi(init_state_p, transition_p, output_p, codes):
    """The recurrence itself: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    The signature is identical to ``_viterbi.py``'s, because ADR 0003 compares
    backends and a backend that changed the signature could not be compared.
    Returns ``(states, total_bits)`` with ``states`` of shape ``(N + 1,)`` -- a
    path over ``N`` symbols visits ``N + 1`` states, since a symbol is emitted
    while *crossing* an arc.

    This wrapper does the four things the compiled kernel cannot: it takes every
    logarithm, with :func:`bits`; it coerces the results to the layout a typed
    memoryview requires; it allocates the working buffers; and it converts the
    result back to Python. It does **not** validate, and it does not raise --
    that is ``viterbi()``'s job, once, for all backends.
    """
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)

    # The logarithms, on the host, in phase 1's operation order: one float64
    # multiply, then one numpy -log2. This is the contract every backend shares.
    # np.unique on an empty record yields an empty `present`, so the table is
    # (S, S, 0) and the kernel's time loop simply does not run.
    #
    # The memoryviews above are `const` because these arrays need not be
    # writable -- ADR 0017 freezes the parameters, and a table derived from them
    # is read-only in spirit. Dropping `const` compiles and then fails at every
    # call with "buffer source array is read-only" for a frozen input.
    seed = np.ascontiguousarray(bits(init_c))
    present, sym = np.unique(codes_c, return_inverse=True)
    arc_bits = np.ascontiguousarray(bits(trans_c[:, :, None] * out_c[:, :, present]))
    # int64, because np.unique's inverse is intp and the memoryview is typed; a
    # mismatch raises at call time rather than at compile time.
    sym_c = np.ascontiguousarray(sym, dtype=np.int64)

    cdef Py_ssize_t n = sym_c.shape[0]
    cdef Py_ssize_t size = seed.shape[0]

    # int64 for the state arrays, matching _viterbi.py's STATE_DTYPE. The
    # original used a float matrix for psi and round-tripped state indices
    # through it; that defect is not reproduced here either.
    delta_np = np.empty((n + 1, size), dtype=np.float64)
    psi_np = np.zeros((n + 1, size), dtype=np.int64)
    states_np = np.empty(n + 1, dtype=np.int64)

    _decode(seed, arc_bits, sym_c, delta_np, psi_np, states_np)

    return states_np, float(delta_np[n, states_np[n]])
