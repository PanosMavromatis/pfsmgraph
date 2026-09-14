"""Forward-backward over one sequence: scaled α and β, and what they are for.

The second dynamic-programming kernel in this package, and the first of revision
03's Baum-Welch. It is [ADR 0002] phase 1, the numpy reference, and it is what
the M-step, the ``hmmlearn`` oracle, the ``torch`` backend and every compiled
phase are checked against.

**It runs in the scaled probability domain, not in log space** [ADR 0020]. So
does the original: ``update-data-p``, ``update-data-dl`` and ``run-add``
(``hmm-trainer.lsh:126-184``, ``346-402``, ``483-652``) all divide each α column
by its own sum and take logarithms of those sums alone. The sums are returned
here as ``scale``. ``Σ bits(scale)`` is the sequence's description length, and
:func:`~pfsmgraph.hmm._numeric.bits` stays the only place a probability becomes
a logarithm, as it is for the decode.

**The evaluation order is contract, not style.** A sum, unlike Viterbi's ``min``,
depends on the order it is taken in. ADR 0020 fixes it as ascending over the
source state and then over the destination, and every later phase has to
reproduce it to stay bit-exact with this module on one host. The reductions
below are therefore ``np.add.accumulate(...)[-1]``, never ``np.sum``, ``@``,
``np.dot`` or ``einsum``. ``accumulate`` returns every partial sum, so
``r[k] = r[k-1] + x[k]`` is part of its definition, and it cannot reorder the
way a reduction that returns only the total is free to. The terms themselves
are elementwise products of two factors, which have no order to fix.

**Emission is on the arc** [ADR 0015]. The table
``w[i, j, u] = transition_p[i, j] * output_p[i, j, present[u]]`` holds one
weight per arc and per symbol present in the record, and every term uses it with
both endpoints. It is the ``(S, S, U)`` table the compiled Viterbi phases already
build, without the logarithm, and it is not a hoisted emission factor.

**What the recurrences are for is expected counts, not ξ.** ``run-add`` keeps ξ
in full as ``P-t``, ``(N + 1)·S²`` floats, only to sum it into the three count
arrays its M-step reads (``hmm-trainer.lsh:584-620``, ``HMMLIB-ACCOUNT.md`` §9).
:func:`_expected_counts` builds each position's ξ, adds it into running totals
and discards it, so memory is ``O(S²·A)`` for the counts plus ``O(N·S)`` for α
and β, never ``O(N·S²)``. The sums over positions follow the same rule as the
recurrences: ascending ``t``, one elementwise addition per position.

.. [ADR 0002] ``docs/design/adr/0002-three-phase-algorithm-lifecycle.md``
.. [ADR 0015] ``docs/design/adr/0015-arc-emission-mealy-formulation.md``
.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

import numpy as np

from ._numeric import bits, safe_divide

__all__: list[str] = []


def _arc_table(transition_p, output_p, codes):
    """``(present, sym, w)``: the symbols present, each position's index into them,
    and ``w[i, j, u] = transition_p[i, j] * output_p[i, j, present[u]]``.

    Shared by the recurrences and the counts so that ξ is formed from the same
    bits the recurrences used. Elementwise, so it has no order to fix.
    """
    present, sym = np.unique(np.asarray(codes), return_inverse=True)
    w = transition_p[:, :, np.newaxis] * output_p[:, :, present]
    return present, sym, w


def _forward_backward(init_state_p, transition_p, output_p, codes):
    """The recurrences: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    Returns ``(alpha, beta, scale)``:

    - ``alpha``, ``(N + 1, S)``. ``alpha[t]`` is the forward variable after
      ``t`` symbols divided by ``scale[1] * ... * scale[t]``, so every row from
      1 on sums to 1 unless the sequence has become impossible. Row 0 is
      ``init_state_p`` exactly as given.
    - ``beta``, ``(N + 1, S)``. ``beta[t]`` is the backward variable divided by
      ``scale[t] * ... * scale[N]``, as ``run-add`` scales it. That choice makes
      ``alpha[t, i] * w[i, j, u] * beta[t + 1, j]`` the pairwise posterior ξ
      with no division by the likelihood, and ``alpha[t] * beta[t] * scale[t]``
      the state posterior γ.
    - ``scale``, ``(N + 1,)``. ``scale[t]`` for ``t >= 1`` is the sum of the
      unscaled column, and ``scale[0]`` is ``1.0``. ``Σ bits(scale)`` is
      ``-log2 P(codes)`` under the parameters as given, including when
      ``init_state_p`` sums to 1 only within ``HMMParams``'s tolerance, since
      row 0 is not renormalised. That is also what exact enumeration over every
      state path computes.

    **Nothing is validated here and nothing raises**, for the reason
    ``_viterbi`` gives: every later phase implements this signature, and a CUDA
    device function cannot raise. An impossible sequence shows up as a zero in
    ``scale``. Scaling goes through ``safe_divide``, so the column that became
    impossible, and every one after it, is all zeros rather than ``0/0 = nan``,
    and ``Σ bits(scale)`` is ``+inf``. ``beta`` is then all zeros as well.
    This follows ``update-data-p`` and ``update-data-dl``, not ``run-add``, whose
    bare ``/`` at ``hmm-trainer.lsh:547``, ``558`` and ``576`` turns the same
    sequence into ``nan`` in every re-estimated parameter.
    """
    codes = np.asarray(codes)
    n = int(codes.shape[0])
    size = int(transition_p.shape[0])

    _, sym, w = _arc_table(transition_p, output_p, codes)

    alpha = np.empty((n + 1, size), dtype=np.float64)
    beta = np.empty((n + 1, size), dtype=np.float64)
    scale = np.empty(n + 1, dtype=np.float64)

    # The seed is not scaled, as in the original, where Q-t[0] is set to 1.0
    # rather than computed.
    alpha[0] = init_state_p
    scale[0] = 1.0

    for t in range(1, n + 1):
        # terms[i, j] = alpha[t-1, i] * w[i, j, u]: elementwise, two factors.
        terms = alpha[t - 1][:, np.newaxis] * w[:, :, sym[t - 1]]
        # Ascending over the source state i, separately for each destination j.
        column = np.add.accumulate(terms, axis=0)[-1]
        # Ascending over j.
        scale[t] = np.add.accumulate(column)[-1]
        alpha[t] = safe_divide(column, scale[t])

    # The last column divided by its own scale factor, then each earlier column
    # by its own: run-add's beta*, which starts at 1 / Q-t[N].
    beta[n] = safe_divide(1.0, scale[n])
    for t in range(n - 1, -1, -1):
        # terms[i, j] = w[i, j, u] * beta[t+1, j]: elementwise, two factors.
        terms = w[:, :, sym[t]] * beta[t + 1][np.newaxis, :]
        # Ascending over the destination state j, separately for each source i.
        row = np.add.accumulate(terms, axis=1)[:, -1]
        beta[t] = safe_divide(row, scale[t])

    return alpha, beta, scale


def _description_length(scale) -> float:
    """``-log2 P(codes)`` in bits: ``Σ bits(scale)``, summed in ascending ``t``.

    The logarithm is taken on the host by :func:`~pfsmgraph.hmm._numeric.bits`,
    as for the decode, and the sum is ``accumulate`` rather than ``np.sum`` for
    the reason the module docstring gives. ``+inf`` exactly when a scale factor
    is zero, which is when the sequence is impossible.
    """
    return float(np.add.accumulate(bits(scale))[-1])


def _state_posteriors(alpha, beta, scale):
    """γ, ``(N + 1, S)``: ``(alpha * beta) * scale``, grouped as written.

    ``gamma[t, i]`` is the probability of occupying state ``i`` before symbol
    ``t`` is emitted, and ``gamma[N]`` after the last. The scale factor restores
    the one ``alpha[t]`` and ``beta[t]`` both divided out. Rows sum to 1 within
    rounding for a possible sequence and are all zeros for an impossible one.
    """
    return (alpha * beta) * scale[:, np.newaxis]


def _expected_counts(alpha, beta, transition_p, output_p, codes):
    """The three count arrays Baum-Welch re-estimates from.

    Returns ``(init_counts, transition_counts, emission_counts)``, shapes
    ``(S,)``, ``(S, S)`` and ``(S, S, A)``: ``run-add``'s ``C-in``, ``C-t`` and
    ``C-t-y``. ``alpha`` and ``beta`` are :func:`_forward_backward`'s.

    At each position ``t < N`` the pairwise posterior
    ``xi[i, j] = (alpha[t, i] * w[i, j, u]) * beta[t + 1, j]`` is formed, added
    into ``transition_counts`` and into ``emission_counts`` at the symbol
    emitted there, and discarded. Because ``beta`` is scaled as ``run-add``
    scales it, that product is already normalised by the likelihood.

    - ``init_counts`` is ξ₀ summed over the destination in ascending order,
      which is how ``run-add`` forms ``C-in``. It equals ``gamma[0]`` within
      rounding, and is all zeros for an empty record, which has no ξ₀.
    - ``emission_counts`` spans the whole symbol axis, as ``output_p`` does, so
      the M-step divides it by ``transition_counts`` with no remapping. Summed
      over symbols it is ``transition_counts``: every arc crossing emits exactly
      one symbol.

    Nothing raises. An impossible sequence has all-zero ``alpha`` columns from
    the dead position on and an all-zero ``beta``, so every count is zero.
    """
    codes = np.asarray(codes)
    n = int(codes.shape[0])
    size = int(transition_p.shape[0])
    n_symbols = int(output_p.shape[2])

    present, sym, w = _arc_table(transition_p, output_p, codes)

    init_counts = np.zeros(size, dtype=np.float64)
    transition_counts = np.zeros((size, size), dtype=np.float64)
    emission_counts = np.zeros((size, size, n_symbols), dtype=np.float64)

    for t in range(n):
        u = sym[t]
        xi = (alpha[t][:, np.newaxis] * w[:, :, u]) * beta[t + 1][np.newaxis, :]
        if t == 0:
            init_counts = np.add.accumulate(xi, axis=1)[:, -1]
        # Elementwise additions, in ascending t: the only order these sums have.
        transition_counts += xi
        emission_counts[:, :, present[u]] += xi

    return init_counts, transition_counts, emission_counts
