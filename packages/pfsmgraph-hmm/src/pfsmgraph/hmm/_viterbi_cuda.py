# dp-compile: derived-from packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi_cpu_parallel.py sha256:7d9613675425d21c901b690300fd54066661469be4fae3b2b63c1f9264be3f9b
"""The Viterbi decode, ADR 0002 phase 4: Numba CUDA.

The decomposition is ``_viterbi_cpu_parallel.py``'s, re-expressed in the CUDA
execution model rather than chosen again: **states within one timestep**, one
device thread per destination state ``j``, with the reduction over ``i`` serial
and ascending inside each thread. Phase 1 stays the oracle.

**The sequential axis is a sequence of launches.** ``delta[t]`` needs all of
``delta[t - 1]``, and CUDA has no grid-wide barrier inside a kernel --
``cuda.syncthreads()`` synchronises one block -- so the loop over ``t`` lives on
the host and each timestep is one kernel launch. The launch boundary is the
barrier. Keeping ``t`` inside the kernel would be correct exactly while ``S``
fits in one block, and wrong above it.

**The logarithms are evaluated on the host, and that is a correctness decision
rather than a speed one.** ``-math.log2`` inside a ``cuda.jit`` kernel lowers to
libdevice, which rounds differently from numpy's: measured 2026-09-13 on an
NVIDIA L4, 1,087,159 of 4,000,008 probability-shaped inputs differ bitwise, each
by one ulp. A transliteration would shift candidate costs by an ulp, let a
near-tie flip an ``argmin``, and make the tie-break contract depend on which
device ran it -- and the constructed uniform model would not notice, because
identical inputs round identically on one device. So the host computes every arc
cost with :func:`bits`, exactly as phase 1 does (multiply the two probabilities,
then one numpy ``-log2``), and the device performs only ``+`` and ``<``. Those
are exact IEEE-754 operations, so the kernel is bit-exact **with phase 1, the
oracle**, by construction rather than by tolerance.

*Corrected 2026-09-13, before this file was registered.* The first version of
this paragraph said phases 1-3 "share one ``log2``" and are bit-identical only
because of it. They do not: phase 1 takes numpy's vectorised ``log2``, phases 2
and 3 call libm's scalar one, and on an AVX-512 host the two differ by one ulp in
about 0.1% of inputs (3,937 of 4,000,000, numpy 2.4.6). So this kernel matched
phase 1 bit for bit and could sit one ulp per arc from phases 2 and 3 in
``total_bits``.

*Resolved 2026-09-14 on* ``fix/hmm-viterbi-log2``. Phases 2 and 3 now take their
logarithms the way this file does -- ``bits`` on the host, over the same
``(S, S, U)`` table -- and perform only ``+`` and ``<``, so all four backends are
bit-exact with one another on a given host. The contract is numpy's ``log2`` as
evaluated through ``bits``; it is not a promise of identical bits across
machines, since numpy's SIMD dispatch differs between them.

**The table spans the symbols present, not the vocabulary.** ``arc_bits`` is
``(S, S, U)`` over ``U = np.unique(codes)``, and the record is re-indexed into
it. ``(S, S, A)`` would grow with the vocabulary -- this family's symbols are
words -- and ``(N, S, S)`` with the record; ``U <= min(N, A)`` bounds it by both.
Phases 2 and 3 now build the same table. It is not the ``(S, S, A)`` precompute
phase 1 refused: that was refused
as ``O(S**2 * A)`` host work hoisted for speed, and it reads like the emission
hoist ADR 0015 forbids. Here the emission factor still depends on both
endpoints -- ``arc_bits[i, j, k]`` -- and is still read inside the inner loop;
what moved is only *where the logarithm runs*.

**The domain is bits, so this is a min-sum.** ``bits(0)`` is ``+inf``, which
absorbs under device addition exactly as on the host, so impossibility falls out
of the arithmetic.

**Nothing on the device validates and nothing raises** -- here the platform
enforces what phase 1 made a rule: a device function holds no Python object and
cannot raise. An impossible sequence returns ``total_bits == inf``, and
``viterbi()`` in ``_viterbi.py`` turns that into ``ImpossibleSequenceError``.

**This module refuses to import without a device.** ``_backends.detect()``
probes a backend by importing its module, and ``@cuda.jit`` decorates lazily, so
with numba-cuda installed and no GPU the import would succeed and the header
would print ``cuda ✓`` over a backend that fails on first call. Raising
:class:`ImportError` here is what turns "no device" into the loud skip that the
registry row's ``optional_on`` names.

**The grid is small by construction, and numba-cuda warns about that once per
launch configuration.** A per-timestep launch over ``S`` states is one or two
blocks, far under numba-cuda's 128-block low-occupancy heuristic. The warning is
filtered locally, for that message alone, rather than by clearing
``config.CUDA_LOW_OCCUPANCY_WARNINGS`` -- a library does not change process-wide
state for its caller. Note the class: numba-cuda raises
``numba.cuda.core.errors.NumbaPerformanceWarning``, which is **not** a subclass
of ``numba.core.errors.NumbaPerformanceWarning``, so a filter on the latter
silently matches nothing.

**No performance tuning yet.** The block size is a placeholder, and one launch
per timestep is the same fixed overhead phase 3 diagnosed: ``t`` is strictly
sequential, so every exact single-record decomposition pays it, and only
revision 03's batch removes it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numba import cuda
from numba.cuda.core.errors import NumbaPerformanceWarning

from ._numeric import bits

if not cuda.is_available():
    raise ImportError(
        "no CUDA device detected: pfsmgraph.hmm._viterbi_cuda needs a working "
        "CUDA device and driver, and numba-cuda reports none"
    )

#: Threads per block. A placeholder until the suite is green -- the skill's rule
#: is no occupancy tuning before the answer is right.
_THREADS_PER_BLOCK = 64


@cuda.jit
def _step(t, arc_bits, sym, delta, psi):
    """One timestep. Each thread owns one destination state ``j``."""
    j = cuda.grid(1)
    size = delta.shape[1]
    # A launch is rounded up to whole blocks, so the last block has threads past
    # the last state. Unchecked, they would write outside delta and psi.
    if j >= size:
        return

    k = sym[t - 1]
    best = math.inf
    best_i = 0
    # Serial, ascending, strict `<`: the tie-break. Only `+` and `<` happen here;
    # every logarithm was taken on the host. Emission is on the arc (ADR 0015),
    # so arc_bits depends on both endpoints and is read inside this loop.
    for i in range(size):
        cand = delta[t - 1, i] + arc_bits[i, j, k]
        if cand < best:
            best = cand
            best_i = i
    delta[t, j] = best
    psi[t, j] = best_i


def _viterbi(init_state_p, transition_p, output_p, codes):
    """The recurrence itself: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    The signature is identical to every other phase's, because ADR 0003 compares
    backends and a backend that changed the signature could not be compared.
    Returns ``(states, total_bits)`` with ``states`` of shape ``(N + 1,)``.

    This wrapper does what the device cannot: take the logarithms, allocate and
    upload the buffers once, drive the timestep launches, and run the backtrace.
    It does **not** validate and does not raise.
    """
    init_c = np.ascontiguousarray(init_state_p, dtype=np.float64)
    trans_c = np.ascontiguousarray(transition_p, dtype=np.float64)
    out_c = np.ascontiguousarray(output_p, dtype=np.float64)
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)

    n = codes_c.shape[0]
    size = trans_c.shape[0]

    # Base case, in the bit domain, on the host: deviation 1 of the
    # formalization and the whole of the seeding fix.
    seed = bits(init_c)

    psi = np.zeros((n + 1, size), dtype=np.int64)
    if n > 0:
        # The arc costs for the symbols this record uses, in phase 1's operation
        # order: one float64 multiply, then one -log2, both by the host.
        present, sym = np.unique(codes_c, return_inverse=True)
        arc_bits = np.ascontiguousarray(
            bits(trans_c[:, :, None] * out_c[:, :, present])
        )

        # Upload once, outside the sequential loop.
        d_arc_bits = cuda.to_device(arc_bits)
        d_sym = cuda.to_device(np.ascontiguousarray(sym, dtype=np.int64))
        d_delta = cuda.device_array((n + 1, size), dtype=np.float64)
        d_delta[0].copy_to_device(seed)
        # psi's row 0 is never read; the backtrace stops at psi[1].
        d_psi = cuda.to_device(psi)

        blocks = (size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Grid size .* low occupancy",
                category=NumbaPerformanceWarning,
            )
            launch = _step[blocks, _THREADS_PER_BLOCK]

        for t in range(1, n + 1):
            launch(t, d_arc_bits, d_sym, d_delta, d_psi)

        last = d_delta[n].copy_to_host()
        psi = d_psi.copy_to_host()
    else:
        last = seed

    # Backtrace, on the host. Strict `<` for the final argmin, matching the
    # recurrence and every other phase: ties go to the smallest state index.
    states = np.empty(n + 1, dtype=np.int64)
    best = math.inf
    best_j = 0
    for j in range(size):
        if last[j] < best:
            best = last[j]
            best_j = j
    states[n] = best_j
    for t in range(n - 1, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states, float(last[states[n]])
