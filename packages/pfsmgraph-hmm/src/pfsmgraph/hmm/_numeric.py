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
a time, which is :func:`numpy.eye` in :func:`stationary_distribution`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bits", "safe_divide", "stationary_distribution"]


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


def stationary_distribution(transition_p) -> np.ndarray:
    """The chain's stationary distribution, by the original's row-replacement solve.

    Translates the first block of ``update-entropy`` (``hmm.lsh:228-244``, and
    its duplicate at ``hmm-param.lsh:64-82``), which is where the original's
    ``state-p`` comes from. Under ADR 0015 the model is arc-emission, so
    ``transition_p`` is ``(S, S)`` over states alone and this is an ordinary
    Markov-chain stationary solve; the emission array plays no part in it.

    **The row replacement is the algorithm, not a detail of it.** ``pi`` is by
    definition a left eigenvector of ``P`` for eigenvalue 1, so
    ``(P.T - I) @ pi == 0`` -- which makes ``(P.T - I)`` singular *by
    construction*. A port that writes that homogeneous system down and hands it
    to a dense solver therefore fails outright, and writing ``(I - P.T)``
    instead has the same null space and needs the same fix. The original trades
    one redundant equation for the normalization: row 0 of the matrix becomes
    all ones and ``b[0]`` becomes 1, so the first equation reads
    ``sum(pi) == 1``.

    **The LU trio is not translated.** ``LU-solve``, ``LU-decomposition`` and
    ``LU-back-substitution`` (``util.lsh:246-415``) transcribe the Numerical
    Recipes routines, down to NR's 1-indexed convention behind a copy-in /
    copy-out wrapper; :func:`numpy.linalg.solve` replaces all three. That is a
    behaviour change and a deliberate one: ``LU-decomposition`` substitutes
    ``TINY = 1e-20`` for a zero pivot (``util.lsh:321-322``) and returns a
    silently perturbed answer where numpy raises. ``int-delta``, the original's
    Kronecker delta, is :func:`numpy.eye` here -- both of its call sites were
    this identity term, built one element at a time.

    Raises :exc:`ValueError` if the chain is **reducible**. The row replacement
    supplies exactly one equation, so it rescues a one-dimensional null space
    and no more; a chain with two closed communicating classes has a
    two-dimensional stationary space and stays singular after it. That is worth
    naming rather than letting ``LinAlgError: Singular matrix`` through, because
    revision 04 searches topology by state merge and split -- a disconnected
    component is a plausible outcome of the search, not a malformed input -- and
    because ``state_p`` is a cached property under ADR 0017, so the error
    surfaces on an attribute access.

    Row-stochasticity is deliberately *not* checked here: it is a property of
    the model, it belongs to ``HMMParams`` at construction under ADR 0017, and a
    second copy of the check could only disagree with the first. The squareness
    check below is a different kind of thing -- a structural precondition of the
    solve, whose absence would surface as a broadcasting error naming shapes the
    caller never wrote.
    """
    p = np.asarray(transition_p, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(
            f"transition_p must be a square (S, S) matrix, got shape {p.shape}"
        )
    size = p.shape[0]

    # A fresh array, so the caller's matrix is untouched -- and so row 0 is
    # assignable at all: HMMParams holds its arrays with writeable = False
    # (ADR 0017), which would make p.T a read-only view.
    a = p.T - np.eye(size)
    a[0, :] = 1.0
    b = np.zeros(size, dtype=np.float64)
    b[0] = 1.0

    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "transition matrix is reducible: (P.T - I) has a null space of "
            "dimension > 1, so the stationary distribution is not unique and "
            "replacing one row with the normalization cannot determine it"
        ) from err
