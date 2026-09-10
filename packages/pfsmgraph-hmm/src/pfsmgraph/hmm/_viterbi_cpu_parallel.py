# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi_cython.pyx sha256:79da7c8de03cf69be864a08a7d07650ae46eabb1998e13013e17a2a19a67d057
"""The Viterbi decode, ADR 0002 phase 3: Numba CPU-parallel, ``prange``.

A **mechanical** translation of ``_viterbi_cython.pyx``, which is itself a
mechanical translation of ``_viterbi.py``. Phase 1 stays the oracle: where this
file and the ``.pyx`` could differ they are made to agree, and where either
disagrees with the Python that is a defect in the chain rather than a tolerance
to widen.

**The parallel axis is ``j``; the reduction over ``i`` stays serial.** That is
*read* from the formalization's ``Parallel decomposition`` section -- settled
2026-09-10 at this phase -- and not chosen here. For fixed ``t`` the ``S`` values
of ``j`` are independent: each reads only row ``t - 1`` of ``delta`` and writes
only ``delta[t, j]`` and ``psi[t, j]``, so the loop is race-free by construction
rather than by scheduling discipline.

**Parallelising ``i`` instead would be a silent contract break, and is the one
defect this file exists to not have.** ``argmin`` over ``i`` returns the *first*
minimal index -- contract, per the formalization, because two backends breaking
ties differently are both correct and disagree. A parallel reduction combines
partial results in an unspecified order and so has no first. Numba privatises
recognised reduction patterns (``+=``, ``max()``) automatically; **an argmin is
not one of them**, which is exactly the reduction a DP backtrace needs. It stays
in a ``range`` loop, and the strict ``<`` is what keeps first-wins.

**There is no anti-diagonal here, and inventing one would be a race.** This array
is time by state, and ``delta[t, j]`` reads *every* ``delta[t - 1, i]``, so the
anti-diagonal ``{t + j = c}`` contains ``(t, j)`` and ``(t - 1, j + 1)`` with a
direct dependency between them -- not an independent set. ADR 0002's claim that
the wavefront is "the same transformation for every DP kernel in the family" is
withdrawn in ADR 0016's ``Resolved`` section; it holds for ``align``, whose
two-dimensional matrix does have independent anti-diagonals.

**The domain is bits, so this is a min-sum**, and ``-log2(0.0)`` is ``+inf`` in
nopython mode exactly as it is in libc and numpy -- verified rather than assumed,
since a nopython ``math.log2(0.0)`` raising instead would have turned the
impossible-sequence path into an exception inside a kernel that must not raise.
``+inf`` absorbs under addition and sorts where it means, so impossibility falls
out of the arithmetic.

**Bit-exactness with phases 1 and 2 is intended.** The same two float64
operations happen in the same order -- multiply, ``-log2``, add -- so candidate
values are identical bit patterns and the minima are too. Note this is safe here
precisely because the reduction is ``min``, which is exact and order-independent
in a way a float *sum* is not: a forward or posterior recurrence parallelised the
same way would reassociate and need a tolerance. Revision 03 writes one.

**The parallelism is thin, and this kernel may be slower than phase 2.** ``S`` is
5 and 8 in the tracked fixtures and on the order of 50 at the ceiling, so
``prange`` over ``j`` offers at most ``S``-way parallelism over an ``S``-element
reduction, and per-timestep fork/join may exceed the work. ADR 0016 scopes phase
3 to parallel *correctness* -- validating a decomposition under real concurrency
before CUDA, where the same decomposition is reused and a race is far more
expensive to find. A phase-3 kernel that loses on wall-clock is the phase working
as specified, not a defect.

**Nothing here validates and nothing raises**, which in nopython mode stops being
a discipline and becomes a constraint: a ``@njit`` function cannot construct an
arbitrary Python exception or hold a Python object at all. A kernel written to
phase 1's rule already satisfies it. An impossible sequence comes back as
``total_bits == inf``; ``viterbi()`` in ``_viterbi.py`` turns that into
``ImpossibleSequenceError``, once, for every backend.

**No ``cache=True``.** It interacts badly with ``parallel=True`` while the source
is still moving -- the cache key does not track everything that can change, and a
stale entry reproduces the previous kernel's behaviour with no warning. The cost
is one compilation per process, which a test session pays once.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _decode(init_p, trans_p, out_p, codes, delta, psi, states):
    """The recurrence and the backtrace. The caller owns every buffer."""
    n = codes.shape[0]
    size = trans_p.shape[0]

    # Base case, in the bit domain: deviation 1 of the formalization and the
    # whole of the seeding fix. Serial rather than prange -- it is S elements
    # once per call, so a fork/join here would cost more than it saves.
    for j in range(size):
        delta[0, j] = -np.log2(init_p[j])

    # psi's row 0 is never read; the backtrace stops at psi[1]. Zeroed by the
    # caller rather than left undefined, matching phases 1 and 2.

    for t in range(1, n + 1):
        code = codes[t - 1]
        # THE parallel loop. Each iteration writes delta[t, j] and psi[t, j] and
        # nothing else, and reads only row t - 1. Do not lift the `i` loop into
        # this one: see the module docstring.
        for j in prange(size):
            best = np.inf
            best_i = 0
            # Serial, ascending, strict `<`. This is the tie-break.
            for i in range(size):
                # The arc cost, fused: multiply the two probabilities, then one
                # logarithm -- the same order as phases 1 and 2, so the rounding
                # is identical rather than merely equivalent. Emission is on the
                # arc (ADR 0015), so out_p depends on both endpoints and cannot
                # be hoisted out of either loop.
                cand = delta[t - 1, i] + -np.log2(trans_p[i, j] * out_p[i, j, code])
                if cand < best:
                    best = cand
                    best_i = i
            delta[t, j] = best
            psi[t, j] = best_i

    # Backtrace. Strictly serial -- states[t] depends on states[t + 1] -- and
    # that is the whole of it: N steps of pointer-chasing against O(N*S**2) of
    # forward work.
    best = np.inf
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

    The signature is identical to ``_viterbi.py``'s and ``_viterbi_cython.pyx``'s,
    because ADR 0003 compares backends and a backend that changed the signature
    could not be compared. Returns ``(states, total_bits)`` with ``states`` of
    shape ``(N + 1,)`` -- a path over ``N`` symbols visits ``N + 1`` states.

    This wrapper does what the kernel cannot: coerce the inputs to a layout
    nopython mode can type, allocate the working buffers, and convert back to
    Python. It does **not** validate and does not raise.
    """
    # Contiguous and float64, matching phase 2. A no-op on what HMMParams
    # actually holds. Unlike the Cython memoryviews these need no `const`
    # treatment -- numba reads a read-only array without complaint -- so ADR
    # 0017's frozen buffers cost nothing here.
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    codes_c = np.ascontiguousarray(codes, dtype=np.int32)

    n = codes_c.shape[0]
    size = trans_c.shape[0]

    delta = np.empty((n + 1, size), dtype=np.float64)
    psi = np.zeros((n + 1, size), dtype=np.int64)
    states = np.empty(n + 1, dtype=np.int64)

    _decode(init_c, trans_c, out_c, codes_c, delta, psi, states)

    return states, float(delta[n, states[n]])
