"""Numeric primitives for the HMM, migrated from the Lush original.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``, and nothing outside this distribution may import it. These are
translations of ``.scratch/hmm-lush/Code/Utility/util.lsh``, kept private
because the Utility library they came from is no part of this family's public
surface (master plan, revision ``02-hmm-v0.1.0``).

**These are description lengths, not probabilities.** ``HMMLIB-ACCOUNT.md`` §3
records that the original accumulates ``sum - log2(x)``, so a quantity built
with :func:`bits` is a negative log-base-2 probability -- a description length
in bits, which *grows* as the probability falls. Viterbi over such quantities
is a min-sum, not a max-product, and a port that reaches for ``max`` inverts
every comparison. The original's own naming hides this: its ``data-p`` and
``result-p`` hold bits despite a ``-p`` suffix that means "probability"
everywhere else in that library. Nothing here carries that suffix.

**Two of the original's functions are deliberately absent.** ``safe->--log``
(``util.lsh:439-444``) compared description lengths with ``-1`` standing in for
log-zero; since :func:`bits` yields ``+inf`` there instead, the comparison is
plain ``>`` and a wrapper would only hide that. ``int-delta``
(``util.lsh:417-422``) built the identity term of ``(P^T - I)`` one element at
a time, which is :func:`numpy.eye`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bits", "safe_divide"]


def bits(p):
    """Description length of ``p`` in bits: ``-log2(p)``, and ``+inf`` at zero.

    Replaces ``safe-add--log2`` (``util.lsh:431-437``), which was *binary* --
    ``(sum, x) -> sum - log2(x)`` -- and carried an explicit ``-1`` sentinel
    for log-zero, absorbing on either argument. Under IEEE-754 none of that
    bookkeeping is needed: ``-log2(0)`` is ``+inf``, and ``+inf`` already
    absorbs under addition. So the primitive here is the *unary* transformation
    and accumulation is plain ``+``::

        total = bits(a) + bits(b)
        cand = delta[i - 1, k] + bits(transition_p[k, j] * output_p[k, j, s])

    Only the ``divide by zero`` warning is suppressed, and **only here**. That
    single suppression is why this function exists rather than dissolving into
    its one-line body: otherwise every call site would either emit spurious
    warnings or repeat the ``errstate``, and one of them would eventually
    silence a warning that mattered. ``invalid`` is deliberately *not*
    suppressed, so a negative input still yields ``nan`` loudly -- a negative
    probability is a bug, not a boundary case.

    Scalar input gives a numpy scalar; array input gives an array of the same
    shape.
    """
    with np.errstate(divide="ignore"):
        return -np.log2(np.asarray(p, dtype=np.float64))


def safe_divide(numerator, denominator) -> np.ndarray:
    """Divide, yielding ``0.0`` wherever the denominator is zero.

    Translates ``safe-/`` (``util.lsh:44-49``). The original is scalar and is
    called from inside explicit loops, but in numpy terms every one of its
    fifteen call sites normalizes an array by a scalar or divides two arrays
    elementwise, so this is written array-aware and covers the scalar case by
    broadcasting.

    Zero is returned for **both** ``0/0`` and ``x/0``. That matches the
    original and is not merely nan-avoidance: numpy alone gives ``nan`` and
    ``inf`` respectively, and either would propagate into a parameter array as
    a silent corruption, where a zero is at least visible.

    No call site in 0.1.0 reaches this yet -- all fifteen are in the forward
    pass, the M-step and the topology surgery, which arrive in revisions 03
    and 04.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=np.float64)
    return np.divide(num, den, out=out, where=den != 0.0)
