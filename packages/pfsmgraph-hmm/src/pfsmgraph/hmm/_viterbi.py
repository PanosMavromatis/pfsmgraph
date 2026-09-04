"""The Viterbi decode: the most probable state path over one sequence.

This is the project's first dynamic-programming kernel, and therefore the first
occupant of [ADR 0002]'s four-phase lifecycle. It is phase 1 -- pure
Python/numpy, written for correctness and for being read, with performance
deferred to the Cython phase rather than pursued here.

**It is a min-sum, not a max-product.** ``HMMLIB-ACCOUNT.md`` section 3 records
that the original accumulates ``sum - log2(x)``, so the quantity being minimized
is a *description length in bits*, which grows as the probability falls. A port
that reaches for ``max`` inverts every comparison, and the original's own naming
hides it: its ``data-p`` and ``result-p`` hold bits despite a ``-p`` suffix
meaning "probability" everywhere else in that library. Nothing here carries that
suffix, and :func:`~pfsmgraph.hmm._numeric.bits` is the only place the domain
change happens.

**Emission is on the arc** [ADR 0015], so the emission factor
``output_p[i, j, symbol]`` depends on *both* endpoints and cannot be hoisted out
of the inner loop the way ``B[state, symbol]`` can in the state-emission
formulation every textbook uses. The loop below therefore forms a full
``(S, S)`` array of arc costs at every position. That is not a hoist and must
not be refactored into one.

**The seeding defect is fixed, not reproduced.** The original seeds
``delta[0][j]`` with the raw ``init-state-p[j]`` -- a probability -- into an
accumulator that holds bits everywhere else (``hmm-trainer.lsh:216-218``,
``HMMLIB-ACCOUNT.md`` section 7, marked *provenance unknown*). Since smaller is
better in this domain, that inverts the preference among start states, and an
exactly-zero ``init_p`` seeds ``0.0``, the *best* possible value, where it should
be impossible. Here ``delta[0] = bits(init_state_p)``, which makes the degenerate
case correct by arithmetic: ``bits(0)`` is ``+inf`` and absorbs under addition.
Measured against the original's own saved decodes, the correction changes exactly
one position in 3807; see ``docs/agents/core.md``.

.. [ADR 0002] ``docs/design/adr/0002-three-phase-algorithm-lifecycle.md``
.. [ADR 0015] ``docs/design/adr/0015-arc-emission-mealy-formulation.md``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pfsmgraph.dataseq import SequenceRecord

from ._numeric import bits
from ._params import HMMParams

__all__ = ["ImpossibleSequenceError", "ViterbiPath", "viterbi"]

#: The dtype of a decoded state path. ``int64`` rather than the original's float
#: matrix: ``HMMLIB-ACCOUNT.md`` section 7's second defect is ``psi`` declared
#: ``float-matrix`` and round-tripping state indices through it, exact only below
#: 2**24 states. Decided not-reproduced by the master plan; this is where that
#: decision lives.
STATE_DTYPE = np.int64


def _viterbi(init_state_p, transition_p, output_p, codes):
    """The recurrence itself: ``(S,)``, ``(S, S)``, ``(S, S, A)``, ``(N,)`` in.

    Returns ``(states, total_bits)`` where ``states`` is ``(N + 1,)`` -- a path
    over ``N`` symbols visits ``N + 1`` states, because each symbol is emitted
    while *crossing* an arc -- and ``total_bits`` is the description length of
    that path.

    **Nothing is validated here and nothing raises**, which is a property of the
    backend contract rather than an oversight. Every later lifecycle phase
    implements this same signature, and a CUDA device function cannot raise a
    Python exception; keeping the kernel purely numeric is what lets phases 2-4
    be transliterations rather than redesigns. An impossible sequence therefore
    comes back as ``total_bits == inf`` with a meaningless path, and
    :func:`viterbi` is what turns that into an error. Callers outside this module
    get the checked wrapper.

    ``total_bits`` is ``+inf`` exactly when no path of positive probability
    exists, since ``bits`` is finite precisely on the positive reals.
    """
    n = int(np.asarray(codes).shape[0])
    size = int(transition_p.shape[0])

    delta = np.empty((n + 1, size), dtype=np.float64)
    psi = np.zeros((n + 1, size), dtype=STATE_DTYPE)

    # Row 0 is the seed, and psi's row 0 is never read -- the backtrace stops at
    # psi[1]. The original's author noted the same of theirs, at
    # hmm-trainer.lsh:219: "(psi 0 state-j) is never used, and is in fact
    # undefined". Ours is defined, because np.zeros is cheaper than explaining an
    # uninitialized row.
    delta[0] = bits(init_state_p)

    for position in range(1, n + 1):
        # (S, S): the cost of crossing i -> j while emitting this symbol. Formed
        # per position rather than precomputed over the whole symbol axis: an
        # (S, S, A) table is O(S**2 * A) and would need the vocabulary to stay
        # small, and it reads like the hoisted emission factor ADR 0015 forbids
        # even though it is not one. Phase 2 fuses this into the inner loop.
        arc_bits = bits(transition_p * output_p[:, :, codes[position - 1]])
        candidates = delta[position - 1][:, np.newaxis] + arc_bits
        # argmin returns the *first* minimal index, which is the tie-break the
        # original has: its update is guarded by a strict "cand is better than
        # current" (safe->--log), so an equal candidate never displaces an
        # earlier one. The tracked models never exercise this -- 0 exact ties in
        # 3804 positions, since learned float parameters do not collide -- so it
        # is pinned by a constructed test instead. It is not decoration: an
        # exactly uniform model ties at every position, and that is what
        # rand_p_vector(size, noise_width=0) returns.
        psi[position] = np.argmin(candidates, axis=0)
        delta[position] = candidates.min(axis=0)

    states = np.empty(n + 1, dtype=STATE_DTYPE)
    states[n] = int(np.argmin(delta[n]))
    for position in range(n - 1, -1, -1):
        states[position] = psi[position + 1, states[position + 1]]

    return states, float(delta[n, states[n]])


class ImpossibleSequenceError(ValueError):
    """No path over this record has finite description length under this model.

    Raised when every state is unreachable, which happens whenever the record
    crosses an arc of probability zero -- an unseen bigram, or a reserved code,
    since ``HMMParams`` requires the reserved fibres of ``output_p`` to be
    exactly zero. ``encode(..., on_unknown="unk")`` is the documented
    ``dataseq`` path that produces the second case.

    A subclass of ``ValueError`` so that ``except ValueError`` keeps working,
    and a distinct type because revision 04's topology search will decode many
    sequences against many candidate topologies, where an impossible sequence is
    an ordinary *search outcome* rather than a malformed input. Discriminating
    the two on an error message would not survive the first rewording.
    """


@dataclass(frozen=True, eq=False)
class ViterbiPath:
    """One decoded path: the states, its cost in bits, and the record's label.

    :param states: ``(N + 1,)`` of :data:`STATE_DTYPE`, read-only. **One longer
        than the symbol array**, because emission is on the arc (ADR 0015): a
        path over ``N`` symbols visits ``N + 1`` states, and ``states[t]`` is the
        state occupied *before* symbol ``t`` is emitted. This is the same
        geometry ``dataseq``'s ``seq-state`` carries and the same one
        ``save-viterbi-path`` printed, with its leading ``-`` row.
    :param total_bits: the description length of this path -- the quantity the
        decode minimized. Always finite; an infinite one raises
        :class:`ImpossibleSequenceError` instead of reaching a caller.
    :param label: the decoded record's ``label``, carried through unchanged. The
        only string that crosses this boundary: ``states`` holds state indices,
        which are not vocabulary codes and have no symbol to decode to.

    **Per-position entropies are deliberately absent.** The original's
    ``path-entropy`` slot is ``state_entropies[path_states[i]]`` element by
    element, which is ``params.state_entropies[path.states]`` here -- one
    fancy-index away, and derived. ADR 0017 stores no derived quantity for the
    same reason: a stored copy can disagree with what it was derived from, and
    this one would be a copy of a cached property of a frozen value.
    """

    states: np.ndarray
    total_bits: float
    label: str | None = None

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=STATE_DTYPE)
        if states.ndim != 1:
            raise ValueError(
                f"states must be 1-D, got {states.ndim}-D with shape {states.shape}"
            )
        # Copy before freezing, as everywhere else in this family: asarray may
        # have returned the caller's own array.
        states = states.copy()
        states.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "total_bits", float(self.total_bits))

    @property
    def n_symbols(self) -> int:
        """``N``. One *fewer* than the number of states visited."""
        return int(self.states.shape[0]) - 1

    def __repr__(self) -> str:
        label = "" if self.label is None else f", label={self.label!r}"
        return (
            f"ViterbiPath(n_symbols={self.n_symbols}, "
            f"total_bits={self.total_bits:.4f}{label})"
        )


def _dead_symbol(init_state_p, transition_p, output_p, codes) -> int:
    """The index of the symbol whose emission left no state reachable.

    A diagnostic, run only on the failure path, and boolean rather than numeric:
    a path has finite cost exactly when every arc it crosses has positive
    probability, since ``bits`` is finite precisely on the positive reals. So
    the support of the min-sum is this sweep, and the two cannot disagree.

    **There is deliberately no "the seed itself was dead" case.** It exists in
    the mathematics -- an all-zero ``init_state_p`` makes every path impossible
    before a symbol is read -- and is unrepresentable in a constructed model,
    because ``HMMParams`` requires ``init_state_p`` to sum to 1. Carrying a
    branch for it would be carrying a branch that cannot fire.
    """
    reachable = np.asarray(init_state_p) > 0.0
    live_arc = np.asarray(transition_p) > 0.0
    for position, code in enumerate(codes):
        emits = output_p[:, :, code] > 0.0
        reachable = (reachable[:, np.newaxis] & live_arc & emits).any(axis=0)
        if not reachable.any():
            return position
    raise AssertionError(  # pragma: no cover -- see the docstring's support argument
        "every symbol is reachable, so the decode cannot have been impossible"
    )


def viterbi(params: HMMParams, record: SequenceRecord) -> ViterbiPath:
    """Decode the most probable state path over ``record`` under ``params``.

    A free function over a frozen parameter value and one record, not a method
    on a trainer. ADR 0017 settles that: the Lush decode reads no forward
    variable, so ``update-viterbi-path``'s placement on ``hmm-trainer`` was an
    artefact of where the corpus lived rather than a dependency.

    :param params: the model. Only its three arrays and ``n_symbols`` are read.
    :param record: one ``dataseq`` ``SequenceRecord``. A record never holds
        padding, so there is no mask to consult; ``pad_collate`` batches are out
        of scope until revision 03.
    :raises ImpossibleSequenceError: if no path has finite description length.
    :raises ValueError: if a code falls outside the model's symbol axis.

    ``record.codes`` indexes ``output_p``'s third axis directly, with no offset
    anywhere -- that is what goal 1's ``A == vocab.size`` decision bought, and it
    is why ADR 0002's later phases have no index arithmetic to port.
    """
    codes = record.codes
    if codes.size:
        lowest, highest = int(codes.min()), int(codes.max())
        if lowest < 0 or highest >= params.n_symbols:
            raise ValueError(
                f"record holds code(s) outside the model's symbol axis "
                f"[0, {params.n_symbols}): observed [{lowest}, {highest}]. The "
                f"usual cause is a record encoded against a different vocabulary "
                f"than the one output_p was sized against"
            )

    states, total_bits = _viterbi(
        params.init_state_p, params.transition_p, params.output_p, codes
    )

    if np.isinf(total_bits):
        index = _dead_symbol(
            params.init_state_p, params.transition_p, params.output_p, codes
        )
        raise ImpossibleSequenceError(
            f"no path over these {record.length} symbols has finite description "
            f"length under a model of {params.n_states} state(s): every state "
            f"became unreachable emitting symbol {index} of the record, code "
            f"{int(codes[index])}, which reaches path position {index + 1}. No "
            f"arc carrying that code leaves a reachable state with positive "
            f"probability, and bits(0) is +inf, which absorbs under addition"
        )

    return ViterbiPath(states=states, total_bits=total_bits, label=record.label)
