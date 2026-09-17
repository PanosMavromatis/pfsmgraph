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

__all__ = [
    "bits",
    "closed_classes",
    "entropy",
    "rand_p_vector",
    "safe_divide",
    "stationary_distribution",
]


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


def _square(values, name: str) -> np.ndarray:
    """``values`` as a float64 ``(S, S)`` array, or ``ValueError`` naming its shape."""
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"{name} must be a square (S, S) matrix, got shape {p.shape}")
    return p


def closed_classes(transition_p) -> np.ndarray:
    """Label each state with its closed communicating class, or ``-1`` if transient.

    New work, not a translation: the original has no such function. It reads
    only which arcs are live (``transition_p > 0``), never their values, so it
    is exact where a numerical rank test is not. A state is **recurrent** when
    every state it can reach can reach it back; the recurrent states split into
    closed classes, numbered from 0 in order of each class's lowest state. Every
    other state is **transient** and labelled ``-1``.

    It exists because the solve's own singularity is not a reliable witness.
    ``[[1, 0, 0], [0, 2/3, 1/3], [0, 1, 0]]`` has two closed classes, ``{0}``
    and ``{1, 2}``, yet ``2/3`` is not representable, so the ``{1, 2}`` block
    misses exact singularity by an ulp and :func:`numpy.linalg.solve` returns
    ``[1, -0, -0]`` without complaint. Revision 04's state merge also weights by
    :func:`stationary_distribution` and needs to know which states carry none.

    Reachability is Warshall's transitive closure over booleans, ``O(S^3)`` with
    ``S`` vectorised steps, which is negligible beside anything that reads a
    corpus.
    """
    live = _square(transition_p, "transition_p") > 0.0
    size = live.shape[0]
    reach = live | np.eye(size, dtype=bool)
    for k in range(size):
        reach |= reach[:, k, None] & reach[k, None, :]
    recurrent = ~np.any(reach & ~reach.T, axis=1)
    labels = np.full(size, -1, dtype=np.int64)
    n_classes = 0
    for i in np.flatnonzero(recurrent):
        if labels[i] < 0:
            # Everything a recurrent state reaches is in its own closed class.
            labels[reach[i]] = n_classes
            n_classes += 1
    return labels


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

    Raises :exc:`ValueError` if the chain is **reducible** into more than one
    closed communicating class. The row replacement supplies exactly one
    equation, so it rescues a one-dimensional null space and no more; a chain
    with two closed classes has a two-dimensional stationary space. **That is
    decided structurally, by :func:`closed_classes`, before any solve**, because
    the solve's singularity is not a reliable witness: a probability that is
    not representable, such as ``2/3``, can lift an exactly singular block off
    singularity by an ulp, and the solve then returns one of the stationary
    distributions without complaint (18 of 254 random reducible chains, measured
    2026-09-17). Naming the failure matters because revision 04 searches
    topology by state merge and split -- a disconnected component is a
    plausible outcome of the search, not a malformed input -- and because
    ``state_p`` is a cached property under ADR 0017, so the error surfaces on an
    attribute access.

    **Transient states get exactly ``0.0``.** They carry no stationary mass, but
    the solve leaves rounding there, up to ``1.8e-14`` in magnitude and some of
    it negative; the entries are overwritten, not renormalised, since the mass
    moved is below any tolerance in this package. The state merge relies on it:
    it weights a pair by stationary mass and refuses a pair with none.

    Row-stochasticity is deliberately *not* checked here: it is a property of
    the model, it belongs to ``HMMParams`` at construction under ADR 0017, and a
    second copy of the check could only disagree with the first. The squareness
    check below is a different kind of thing -- a structural precondition of the
    solve, whose absence would surface as a broadcasting error naming shapes the
    caller never wrote.
    """
    p = _square(transition_p, "transition_p")
    size = p.shape[0]

    labels = closed_classes(p)
    n_classes = int(labels.max()) + 1
    if n_classes > 1:
        members = ", ".join(
            "{" + ", ".join(str(s) for s in np.flatnonzero(labels == c)) + "}"
            for c in range(n_classes)
        )
        raise ValueError(
            f"transition matrix is reducible: it has {n_classes} closed "
            f"communicating classes ({members}), so its stationary distribution "
            "is not unique"
        )

    # A fresh array, so the caller's matrix is untouched -- and so row 0 is
    # assignable at all: HMMParams holds its arrays with writeable = False
    # (ADR 0017), which would make p.T a read-only view.
    a = p.T - np.eye(size)
    a[0, :] = 1.0
    b = np.zeros(size, dtype=np.float64)
    b[0] = 1.0

    try:
        pi = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as err:
        # One closed class makes the replaced system nonsingular in exact
        # arithmetic, so only rounding can reach this.
        raise ValueError(
            "transition matrix has one closed communicating class, but the "
            "row-replaced system is numerically singular"
        ) from err
    pi[labels < 0] = 0.0
    return pi


def entropy(p, axis=-1):
    """Shannon entropy in bits, ``-sum(p * log2(p))``, taking ``0 log 0`` as 0.

    Translates ``calculate-entropy`` (``util.lsh:448-460``), whose loop skips
    zero entries explicitly -- ``(when (<> p-i 0.0) ...)`` -- which is that
    convention spelled as control flow.

    **This deliberately does not reuse :func:`bits`, and the reason is worth
    stating.** Entropy is ``sum(p * bits(p))``, so reuse looks obvious; but
    ``bits(0)`` is ``+inf``, which is *correct* for a description length -- an
    impossible event costs infinitely many bits -- and *wrong* here, because in
    entropy the zero is a weight as well as an argument, and ``0 * inf`` is
    ``nan`` rather than the 0 the convention calls for. The two functions
    disagree about zero because they are asking different questions of it.

    The zeros are handled by substituting 1.0 before the logarithm, since
    ``log2(1) == 0`` makes those terms vanish without ``log2(0)`` ever being
    evaluated. The mask is ``!= 0`` rather than ``> 0`` on purpose: a *negative*
    input then still reaches :func:`numpy.log2` and still yields ``nan``
    loudly, matching :func:`bits`, where a negative probability is a bug rather
    than a boundary case.

    ``axis`` defaults to the last one, so a 1-D distribution gives a scalar and
    an ``(S, A)`` array of per-state symbol distributions gives ``S``
    entropies in one call. That second shape is the one the model wants: the
    original computes each state's symbol distribution by marginalizing the
    Mealy emission over successor states,
    ``p_i(k) = sum_j transition_p[i, j] * output_p[i, j, k]``
    (``hmm.lsh:246-253``), which is ``np.einsum("ij,ijk->ik", ...)``. That
    marginalization is model-shaped and belongs with ``HMMParams``, not here.
    """
    probabilities = np.asarray(p, dtype=np.float64)
    nonzero = np.where(probabilities != 0.0, probabilities, 1.0)
    return -np.sum(probabilities * np.log2(nonzero), axis=axis)


def rand_p_vector(size, noise_width, rng) -> np.ndarray:
    """A near-uniform random probability vector of length ``size``.

    Translates ``rand-p-vector`` (``util.lsh:523-546``): each element is set to
    ``1 + noise_width * U(-1, 1)`` and the vector is then normalized, so
    ``noise_width = 0`` gives the exactly uniform distribution and larger values
    spread it. The original's commented-out alternative at ``531-533`` perturbs
    the *existing* contents instead, and is annotated "it doesn't work very
    well".

    **It assigns rather than perturbs**, which is why this takes a ``size`` and
    returns a new array where the original took an ``idx`` and overwrote it in
    place. The Lush body discards whatever the argument held, so an
    out-parameter would carry no information; the generated C
    (``Code/Utility/C/util.c``) settled that mechanically, as a plain
    ``IDX_PTR(...)[...] = (1+(noise_width*rand))``.

    ``rng`` is a required :class:`numpy.random.Generator`. There is deliberately
    no default and no module-level state: parameters are a frozen value under
    ADR 0017, which is hollow if the value cannot be re-derived, and ADR 0002
    commits to ``prange`` and CUDA phases where a shared generator is a data
    race rather than a style preference. The original drew from Lush's global
    ``(rand 1.0 -1.0)``, which the generated C shows is
    ``((-1) - (1)) * Frand() + (1)`` -- uniform on ``(-1, 1]``, where numpy's
    :meth:`~numpy.random.Generator.uniform` is ``[-1, 1)``. The half-open end
    differs on a set of measure zero and is not worth reproducing.

    ``noise_width`` must lie in ``[0, 1)``. At 1 or above an element can reach
    zero or go negative, and normalizing that still yields a vector summing to 1
    -- a negative "probability" that would flow into a parameter array as
    exactly the silent corruption :func:`safe_divide` returns 0 to avoid. The
    original does not check, but its only call sites pass 0.001 and 0.1.
    """
    # Checked before the comparison rather than left to it: passing the array to
    # be filled is the natural mistake here, since that is what the Lush
    # signature took, and `size < 1` on an array raises numpy's "truth value is
    # ambiguous" -- a ValueError about nothing the caller did.
    if not isinstance(size, (int, np.integer)):
        raise TypeError(
            f"size must be an integer, got {type(size).__name__}; this returns a "
            "new vector rather than filling one, unlike the Lush original"
        )
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if not 0.0 <= noise_width < 1.0:
        raise ValueError(
            f"noise_width must lie in [0, 1), got {noise_width}; at 1 or above "
            "an element can go negative and normalization would hide it"
        )
    values = 1.0 + noise_width * rng.uniform(-1.0, 1.0, size=size)
    return values / values.sum()
