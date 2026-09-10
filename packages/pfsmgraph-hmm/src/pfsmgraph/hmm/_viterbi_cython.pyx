# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi.py sha256:48666ca8a3126ba141ec1a0ddba8ed1823e502f93d72483181906afd24135742
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

**Emission is on the arc** [ADR 0015]: the factor is ``out_p[i, j, code]``, read
inside the ``i`` loop because it depends on *both* endpoints. Phase 1 forms a
whole ``(S, S)`` array of arc costs per position and says in as many words that
this is not a hoist and must not be refactored into one. Here the same quantity
is computed scalar-wise at the point of use -- which is the fusion phase 1's
comment anticipates ("Phase 2 fuses this into the inner loop"), not a hoist: the
value still depends on ``i`` and ``j`` and is never lifted out of either loop.

**Bit-exactness with phase 1 is intended, not approximate.** The same two
float64 operations happen in the same order -- multiply, then ``-log2``, then add
-- so the candidate values are identical bit patterns, and the tie-break is
identical too: ``cand < best`` is strict, so an equal candidate never displaces an
earlier ``i``, which is exactly what ``np.argmin`` does when it returns the first
minimal index. The formalization makes "ties to the SMALLEST i" contract, because
two correct backends would otherwise legitimately disagree.

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
from libc.math cimport INFINITY, log2

import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _decode(
    const double[::1] init_p,
    const double[:, ::1] trans_p,
    const double[:, :, ::1] out_p,
    const int[::1] codes,
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
    cdef Py_ssize_t n = codes.shape[0]
    cdef Py_ssize_t size = trans_p.shape[0]
    cdef Py_ssize_t t, i, j, best_i
    cdef int code
    cdef double best, cand

    # Base case. delta[0, j] = bits(init_p[j]) -- the bit domain, which is
    # deviation 1 of the formalization and the whole of the seeding fix. The
    # original seeds a raw probability into a bit accumulator, inverting the
    # preference among start states; bits(0) is +inf here and absorbs, so an
    # impossible start stays impossible instead of becoming the best value.
    for j in range(size):
        delta[0, j] = -log2(init_p[j])

    # psi's row 0 is never read: the backtrace stops at psi[1]. It is zeroed by
    # the caller rather than left undefined, matching phase 1.

    for t in range(1, n + 1):
        code = codes[t - 1]
        for j in range(size):
            best = INFINITY
            best_i = 0
            for i in range(size):
                # The arc cost, fused: multiply the two probabilities, then take
                # one logarithm. Phase 1 does exactly this, vectorized -- so the
                # rounding is identical rather than merely equivalent.
                cand = delta[t - 1, i] + -log2(trans_p[i, j] * out_p[i, j, code])
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

    This wrapper does the three things the compiled kernel cannot: it coerces the
    inputs to the layout a typed memoryview requires, it allocates the working
    buffers, and it converts the result back to Python. It does **not** validate,
    and it does not raise -- that is ``viterbi()``'s job, once, for all backends.
    """
    # ascontiguousarray is a no-op on what HMMParams actually holds -- float64
    # and C-contiguous already -- so the common path copies nothing. It is here
    # for the caller who hands in a transposed view, where a `::1` memoryview
    # would otherwise raise instead of working.
    #
    # These stay READ-ONLY: ADR 0017 freezes the parameter arrays, so the
    # memoryviews above must be `const`. Dropping `const` compiles and then fails
    # at every call with "buffer source array is read-only" -- a runtime error
    # from a compile-time-looking mistake.
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    # int32 because that is dataseq's CODE_DTYPE. Coercing here rather than in
    # the caller keeps the boundary in one place.
    codes_c = np.ascontiguousarray(codes, dtype=np.int32)

    cdef Py_ssize_t n = codes_c.shape[0]
    cdef Py_ssize_t size = trans_c.shape[0]

    # int64 for the state arrays, matching _viterbi.py's STATE_DTYPE. The
    # original used a float matrix for psi and round-tripped state indices
    # through it; that defect is not reproduced here either.
    delta_np = np.empty((n + 1, size), dtype=np.float64)
    psi_np = np.zeros((n + 1, size), dtype=np.int64)
    states_np = np.empty(n + 1, dtype=np.int64)

    _decode(init_c, trans_c, out_c, codes_c, delta_np, psi_np, states_np)

    return states_np, float(delta_np[n, states_np[n]])
