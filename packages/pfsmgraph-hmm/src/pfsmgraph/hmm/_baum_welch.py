"""Baum-Welch re-estimation over :mod:`._forward_backward`'s expected counts.

The E-step is :func:`~._forward_backward._expected_counts`; this module turns its
three count arrays into new parameters, and will hold the EM loop that alternates
the two. It is kept out of ``_forward_backward.py`` because that module is the
dynamic-programming kernel ADR 0002's phases transliterate, and the M-step has no
recurrence to transliterate: it is two reductions and three elementwise
divisions.

**The M-step is ``run-add``'s stage 5, translated as written**
(``hmm-trainer.lsh:620-651``). Every division is ``safe-/``, here
:func:`~._numeric.safe_divide`, so a zero denominator yields zero rather than
``nan``. Of the three divisions only one can produce something ``HMMParams``
rejects:

- ``init_state_p`` divides by ``Σ init_counts``, which is 1 within rounding for
  any possible record and zero only when every count is zero.
- ``transition_p[j]`` divides by state ``j``'s outgoing count, ``Σ_{t<N} γ_t(j)``,
  which is zero for a state never occupied before the last position. The row
  comes back all zero, and ``HMMParams`` rejects an all-zero row.
- ``output_p[j, k]`` divides by ``transition_counts[j, k]``, which is zero on
  every arc with no posterior mass. But that count is also the new
  ``transition_p[j, k]``'s numerator, so the all-zero fibre only ever lands on a
  dead arc, which ``HMMParams`` exempts.

**What to do about the zero row is the caller's decision, not this function's.**
It returns the outgoing counts beside the parameters so the EM loop can find such
states; the loop restores their previous row and fibres. Any row maximises the EM
objective for a state with no outgoing count, so the choice keeps Baum's
non-decrease guarantee, and the likelihood is bit-identical to the original's
zero row. As with the decode, the numeric function neither validates nor raises.

**The two reductions follow ADR 0020's order.** The original sums the outgoing
count over ``k`` and the initial count over ``j``, both ascending; both are
``np.add.accumulate(...)[-1]`` here for the reason ``_forward_backward.py``'s
docstring gives. The divisions are elementwise and have no order.

.. [ADR 0002] ``docs/design/adr/0002-three-phase-algorithm-lifecycle.md``
.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

import numpy as np

from ._numeric import safe_divide

__all__: list[str] = []


def _m_step(init_counts, transition_counts, emission_counts):
    """Re-estimate the parameters from one E-step's expected counts.

    Takes :func:`~._forward_backward._expected_counts`'s ``(S,)``, ``(S, S)`` and
    ``(S, S, A)`` arrays and returns ``(init_state_p, transition_p, output_p,
    out_counts)``, the last being each state's outgoing count, shape ``(S,)``.

    Nothing is checked and nothing raises. A state whose ``out_counts`` entry is
    zero has an all-zero ``transition_p`` row and all-zero ``output_p`` fibres,
    and the whole result is zeros when every count is: what either means is for
    the caller to decide.
    """
    init_counts = np.asarray(init_counts, dtype=np.float64)
    transition_counts = np.asarray(transition_counts, dtype=np.float64)
    emission_counts = np.asarray(emission_counts, dtype=np.float64)

    init_total = np.add.accumulate(init_counts)[-1]
    out_counts = np.add.accumulate(transition_counts, axis=1)[:, -1]

    init_state_p = safe_divide(init_counts, init_total)
    transition_p = safe_divide(transition_counts, out_counts[:, np.newaxis])
    output_p = safe_divide(emission_counts, transition_counts[:, :, np.newaxis])
    return init_state_p, transition_p, output_p, out_counts
