"""Forward-backward: scaled α and β over one sequence.

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

.. [ADR 0002] ``docs/design/adr/0002-three-phase-algorithm-lifecycle.md``
.. [ADR 0015] ``docs/design/adr/0015-arc-emission-mealy-formulation.md``
.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

import numpy as np

from ._numeric import safe_divide

__all__: list[str] = []


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

    # (S, S, U) over the symbols this record uses. Elementwise, so order-free.
    present, sym = np.unique(codes, return_inverse=True)
    w = transition_p[:, :, np.newaxis] * output_p[:, :, present]

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
