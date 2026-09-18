"""Topology moves for the revision 04 search: state split and state merge.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``, and nothing here becomes public at 0.3.0: ADR 0025
(``docs/design/adr/0025-topology-search-not-exported.md``) keeps the whole topology
search private, so 0.3.0 ships it with no supported entry point. Where a move is
re-converged and scored is :mod:`._trials`; this module only builds the candidate
(master plan, revision ``04-hmm-v0.3.0``).

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

**The merge follows the Lush ``merge-states`` (``hmm-param.lsh:218-369``) closely,
and differs in three places.** It carries ``init_state_p`` where ``:262`` read
``state_p`` (ADR 0022 section 1); it weights by stationary mass that is exactly
zero on transient states, where the original weighted by a solve's rounding; and
it refuses a pair of two transient states, from which the original built an
all-zero transition row.
"""

from __future__ import annotations

import numpy as np

from pfsmgraph.dataseq import USER_BASE

from ._numeric import rand_p_vector, safe_divide
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


def _merge_states(params: HMMParams, first: int, second: int) -> HMMParams:
    """Merge two states into the lower-indexed one, removing the higher.

    :param params: the incumbent model, with ``S`` states and a unique stationary
        distribution.
    :param first: one state of the pair, in ``range(S)``.
    :param second: the other, distinct from ``first``; the order does not matter.
    :returns: an ``S - 1``-state :class:`~._params.HMMParams` over the same
        vocabulary.
    :raises ValueError: if the pair is not two distinct states of the model; if the
        incumbent is reducible (from :attr:`~._params.HMMParams.state_p`); or if
        both states are transient, since then no stationary mass weights their
        outbound arcs.

    Writing ``a < b`` for the pair, ``T``, ``O`` for the incumbent's transition and
    output arrays, and ``p1, p2`` for ``state_p[a], state_p[b]`` renormalised to
    sum to one, states above ``b`` move down one index and every array keeps its
    other entries:

    - **Initial probabilities.** ``init'[a] = init[a] + init[b]``, and every other
      state keeps its own. (The original's ``:262`` read ``state_p`` here; ADR 0022
      section 1.)
    - **Arcs into the merged state.** ``T'[j, a] = T[j, a] + T[j, b]``, and the fibre
      is their transition-weighted average, so each predecessor row keeps its sum.
    - **Arcs out of it.** ``T'[a, j] = p1*T[a, j] + p2*T[b, j]``, with the fibre
      weighted by ``p1*T[a, j]`` and ``p2*T[b, j]``.
    - **The self-loop.** The four arcs among the pair collapse:
      ``T'[a, a] = p1*(T[a, a] + T[a, b]) + p2*(T[b, a] + T[b, b])``, with the fibre
      averaging all four.

    Every fibre average divides by its new transition with
    :func:`~._numeric.safe_divide`, so a dead arc gets an all-zero fibre. Nothing
    here is drawn at random, and the merge is symmetric in its arguments.

    **Weights are stationary mass, not occupancy.** A transient state carries
    exactly zero (:func:`~._numeric.stationary_distribution`), so merged with a
    recurrent one it contributes nothing to the outbound arcs, whatever it learned.
    That is the original's behaviour, and right for a cyclic stream. Two transient
    states have no weight at all: the original's ``safe-/`` then built an all-zero
    row, which ``HMMParams`` rejects, so that case is refused here. A merge cannot
    make an incumbent with one closed class reducible: merging two recurrent states
    contracts one class, and a transient state's outbound arcs get weight zero while
    its inbound arcs now lead into the class.

    Nothing here re-converges or scores the candidate.
    """
    for name, value in (("first", first), ("second", second)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer state index, got {type(value).__name__}")
    size = params.n_states
    for name, value in (("first", first), ("second", second)):
        if not 0 <= value < size:
            raise ValueError(f"{name} must lie in [0, {size}), got {value}")
    if first == second:
        raise ValueError(f"cannot merge state {first} with itself")
    a, b = sorted((int(first), int(second)))

    weight = params.state_p
    total = weight[a] + weight[b]
    if total == 0.0:
        raise ValueError(
            f"cannot merge states {a} and {b}: both are transient, so neither carries "
            "stationary mass to weight their outbound arcs by"
        )
    p1, p2 = weight[a] / total, weight[b] / total

    T, O = params.transition_p, params.output_p
    keep = np.flatnonzero(np.arange(size) != b)

    init = params.init_state_p[keep]
    init[a] = params.init_state_p[a] + params.init_state_p[b]

    transition = T[np.ix_(keep, keep)]
    output = O[np.ix_(keep, keep)]

    # Arcs into the merged state, for every row including a; (a, a) is overwritten below.
    transition[:, a] = T[keep, a] + T[keep, b]
    output[:, a] = safe_divide(
        T[keep, a, None] * O[keep, a] + T[keep, b, None] * O[keep, b], transition[:, a, None]
    )

    # Arcs out of the merged state, weighted by stationary mass; (a, a) again below.
    transition[a] = p1 * T[a, keep] + p2 * T[b, keep]
    output[a] = safe_divide(
        p1 * T[a, keep, None] * O[a, keep] + p2 * T[b, keep, None] * O[b, keep],
        transition[a, :, None],
    )

    # The four arcs among the pair collapse into the merged self-loop.
    transition[a, a] = p1 * (T[a, a] + T[a, b]) + p2 * (T[b, a] + T[b, b])
    output[a, a] = safe_divide(
        p1 * T[a, a] * O[a, a]
        + p1 * T[a, b] * O[a, b]
        + p2 * T[b, a] * O[b, a]
        + p2 * T[b, b] * O[b, b],
        transition[a, a],
    )

    return HMMParams(init, transition, output, params.vocabulary)
