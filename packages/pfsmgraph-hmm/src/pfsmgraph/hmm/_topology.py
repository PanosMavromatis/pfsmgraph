"""Topology moves for the revision 04 search: state split, and later state merge.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``. The public surface for topology search arrives with the search loop,
which is also where a move is re-converged and scored; this module only builds the
candidate (master plan, revision ``04-hmm-v0.3.0``).

Each move is a free function over :class:`~._params.HMMParams` that returns a new
value built from fresh arrays, as ``baum_welch`` does after each M-step (ADR 0017,
and ``exp/hmm-param-representation``, which measured building a candidate at about
0.005% of the cost of scoring it).

**The split is not a transliteration of the original**, and
ADR 0022 (``docs/design/adr/0022-state-split-initialisation.md``) is
authoritative for how it differs. The Lush ``split-state``
(``.scratch/hmm-lush/Code/HMMlib/hmm-param.lsh:142-215``) seeds uninvolved states'
initial probabilities from ``state_p`` (``:172``), a defect this fixes; and it
tells the twins apart by redrawing every fibre leaving them, which discards what
the split state had learned. Here the twins are told apart on the arcs *into*
them instead, and every learned parameter is kept.
"""

from __future__ import annotations

import numpy as np

from pfsmgraph.dataseq import USER_BASE

from ._numeric import rand_p_vector
from ._params import HMMParams

#: Half-width of each predecessor's perturbation of its inbound halving: the arc
#: into the split state gets ``(1/2 + w*u) * T[j, s]`` with ``u ~ U(-1/2, 1/2)``.
#: ADR 0022 section 2.
_INBOUND_WIDTH = 0.01

#: Weight of the random fibre mixed into each live outbound fibre of both twins.
#: ADR 0022 section 3.
_SEED_WEIGHT = 0.01

#: Noise width of the random fibre that seed mixes in, as ``rand_p_vector`` takes it.
_SEED_NOISE = 0.1


def _split_state(params: HMMParams, state: int, *, rng: np.random.Generator) -> HMMParams:
    """Split ``state`` into itself and a twin appended at index ``S``.

    :param params: the incumbent model, with ``S`` states.
    :param state: the state ``s`` to split, in ``range(S)``.
    :param rng: the generator every random draw comes from. Required, with no
        default and no module state, as :func:`~._numeric.rand_p_vector` takes it.
    :returns: an ``S + 1``-state :class:`~._params.HMMParams` over the same
        vocabulary.

    Writing ``p = S`` for the twin and ``T``, ``O`` for the incumbent's transition
    and output arrays:

    - **Initial probabilities.** ``init'[s] = init'[p] = init[s] / 2``, and every
      other state keeps ``init[i]``. (The original read ``state_p[i]`` here.)
    - **Arcs into the split state.** Each predecessor ``j``, including ``s``
      itself, splits its arc as ``T'[j, s] = (1/2 + w*u_j) * T[j, s]`` and
      ``T'[j, p] = T[j, s] - T'[j, s]``, with one independent ``u_j`` per
      predecessor. The fraction lies in ``[0.495, 0.505]``, so by Sterbenz's lemma
      the subtraction is exact and the two halves sum back to ``T[j, s]``
      bit for bit; every predecessor row keeps its sum and every zero arc stays
      zero. The fibre on ``j -> p`` is a copy of ``j -> s``.
    - **Arcs out of the twins.** ``p``'s transition row and fibres copy ``s``'s,
      taken after the inbound split, so the self-loop's split carries over to
      ``p -> s`` and ``p -> p``.
    - **The seed.** Every fibre leaving either twin on a live arc becomes
      ``(1 - 0.01) * learned + 0.01 * rand_p_vector(n_symbols - USER_BASE, 0.1)``
      over the user symbols. The reserved block stays exactly zero, and fibres on
      dead arcs are copied unchanged.

    Before the seed the result is an exact reparameterisation: every sequence has
    the probability the incumbent gave it. The per-predecessor ``u_j`` let EM
    separate the twins by which predecessor led into them; the seed separates them
    where they have only one distinct predecessor row, which includes every split
    of a one-state model.

    **The draw order is contract** (ADR 0022, **Resolved**), and the number of
    draws depends on ``S`` alone. First ``u = rng.uniform(-0.5, 0.5, size=S)``,
    one per predecessor whether or not its arc is live; then one ``rand_p_vector``
    per destination ``j = 0..S`` for twin ``s``, then for twin ``p``, drawn for
    every arc and discarded on dead ones. So a split consumes
    ``S + 2 * (S + 1) * (n_symbols - USER_BASE)`` uniforms, and a change in which
    arcs are live never shifts a later draw.

    Nothing here re-converges or scores the candidate; ADR 0022 section 4 obliges
    whatever does to give it a minimum EM budget.
    """
    if not isinstance(state, (int, np.integer)) or isinstance(state, bool):
        raise TypeError(f"state must be an integer state index, got {type(state).__name__}")
    size = params.n_states
    if not 0 <= state < size:
        raise ValueError(f"state must lie in [0, {size}), got {state}")
    s, p = int(state), size
    n_user = params.n_symbols - USER_BASE

    # Every draw happens here, in the contract's order, before any array is built.
    u = rng.uniform(-0.5, 0.5, size=size)
    seeds = np.empty((2, size + 1, n_user))
    for twin in range(2):
        for j in range(size + 1):
            seeds[twin, j] = rand_p_vector(n_user, _SEED_NOISE, rng)

    init = np.empty(size + 1)
    init[:size] = params.init_state_p
    init[s] = init[p] = 0.5 * params.init_state_p[s]

    transition = np.zeros((size + 1, size + 1))
    transition[:size, :size] = params.transition_p
    inbound = params.transition_p[:, s]
    transition[:size, s] = (0.5 + _INBOUND_WIDTH * u) * inbound
    transition[:size, p] = inbound - transition[:size, s]
    transition[p] = transition[s]

    output = np.zeros((size + 1, size + 1, params.n_symbols))
    output[:size, :size] = params.output_p
    output[:size, p] = params.output_p[:, s]
    output[p] = output[s]

    live = transition[s] > 0.0
    for twin, t in enumerate((s, p)):
        output[t, live, USER_BASE:] = (1.0 - _SEED_WEIGHT) * output[t, live, USER_BASE:] + (
            _SEED_WEIGHT * seeds[twin, live]
        )

    return HMMParams(init, transition, output, params.vocabulary)
