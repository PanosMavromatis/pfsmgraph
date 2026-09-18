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
from pfsmgraph.dataseq import SequenceRecord, pad_collate

from ._backends import BackendName, _resolve
from ._numeric import bits
from ._params import HMMParams

__all__ = ["ImpossibleSequenceError", "ViterbiPath", "viterbi", "viterbi_batch"]

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


def _viterbi_batch(init_state_p, transition_p, output_p, codes, lengths):
    """The recurrence over a padded batch: ``codes`` ``(B, L)``, ``lengths`` ``(B,)``.

    Returns ``(states, total_bits)``, shapes ``(B, L + 1)`` and ``(B,)``. Row ``b`` is
    :func:`_viterbi` on record ``b`` alone, **bit for bit**: ``states[b, : n_b + 1]``
    is its path, zeros follow it, and ``total_bits[b]`` is its total. A min-sum
    performs only ``bits``, ``+`` and ``<``, elementwise, and none of them regroups
    across the batch axis, so batching cannot move a bit the way regrouping a sum
    does in ``_e_step_batch``:

    - at a padded step δ is carried unchanged and ψ is not written, so
      ``delta[b, L]`` equals ``delta[b, n_b]`` and one final column serves every row;
    - each backtrace starts at the record's own ``n_b``, so no padded ψ is read;
    - ``argmin`` over the source axis keeps the first-wins tie-break in every row.

    **The carry is load-bearing.** ``PAD``'s fibre is zero in every valid model, so a
    padded step taken would make ``delta[b, L]`` infinite and report every record
    shorter than the batch impossible. Liveness is derived from ``lengths`` rather
    than taken as a mask, so the two cannot disagree. As in :func:`_viterbi`,
    nothing is validated and nothing raises: an impossible row comes back with
    ``total_bits[b] == inf``.
    """
    codes = np.asarray(codes)
    lengths = np.asarray(lengths, dtype=np.int64)
    batch, width = codes.shape
    size = int(transition_p.shape[0])

    delta = np.empty((batch, width + 1, size), dtype=np.float64)
    psi = np.zeros((batch, width + 1, size), dtype=STATE_DTYPE)
    delta[:, 0] = bits(init_state_p)

    for position in range(1, width + 1):
        delta[:, position] = delta[:, position - 1]
        rows = np.flatnonzero(lengths >= position)
        # (R, S, S): each live record's arc costs, the per-record (S, S) with a
        # leading axis. Still formed per position, for the reason _viterbi gives.
        emit = np.moveaxis(output_p[:, :, codes[rows, position - 1]], 2, 0)
        candidates = delta[rows, position - 1][:, :, np.newaxis] + bits(transition_p * emit)
        psi[rows, position] = np.argmin(candidates, axis=1)
        delta[rows, position] = candidates.min(axis=1)

    states = np.zeros((batch, width + 1), dtype=STATE_DTYPE)
    last = np.argmin(delta[:, width], axis=1)
    total_bits = delta[np.arange(batch), width, last]
    for row in range(batch):
        n = int(lengths[row])
        states[row, n] = last[row]
        for position in range(n - 1, -1, -1):
            states[row, position] = psi[row, position + 1, states[row, position + 1]]
    return states, total_bits


class ImpossibleSequenceError(ValueError):
    """No path over this record has finite description length under this model.

    Raised when every path over the record crosses at least one arc of
    probability zero, so that after some symbol no state is reachable. A reserved
    code does that to every path at once, since ``HMMParams`` requires the
    reserved fibres of ``output_p`` to be exactly zero, and
    ``encode(..., on_unknown="unk")`` is the documented ``dataseq`` path that puts
    one into a record. An unseen bigram does it only when no route avoids the
    dead arc.

    A subclass of ``ValueError`` so that ``except ValueError`` keeps working,
    and a distinct type because revision 04's topology search decodes many
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
        decode minimized. Finite in every path :func:`viterbi` returns, since an
        infinite one raises :class:`ImpossibleSequenceError` instead of reaching a
        caller; this constructor does not check it.
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


def _range_error(params, codes) -> str | None:
    """Why ``codes`` do not fit ``params``' symbol axis, or ``None`` when they do."""
    if codes.size:
        lowest, highest = int(codes.min()), int(codes.max())
        if lowest < 0 or highest >= params.n_symbols:
            return (
                f"record holds code(s) outside the model's symbol axis "
                f"[0, {params.n_symbols}): observed [{lowest}, {highest}]. The "
                f"usual cause is a record encoded against a different vocabulary "
                f"than the one output_p was sized against"
            )
    return None


def _impossible_message(params, record) -> str:
    """What :class:`ImpossibleSequenceError` says about ``record``: its dead symbol."""
    codes = record.codes
    index = _dead_symbol(params.init_state_p, params.transition_p, params.output_p, codes)
    return (
        f"no path over these {record.length} symbols has finite description "
        f"length under a model of {params.n_states} state(s): every state "
        f"became unreachable emitting symbol {index} of the record, code "
        f"{int(codes[index])}, which reaches path position {index + 1}. No "
        f"arc carrying that code leaves a reachable state with positive "
        f"probability, and bits(0) is +inf, which absorbs under addition"
    )


def viterbi(
    params: HMMParams, record: SequenceRecord, *, backend: BackendName = "python"
) -> ViterbiPath:
    """Decode the most probable state path over ``record`` under ``params``.

    A free function over a frozen parameter value and one record, not a method
    on a trainer. ADR 0017 settles that: the Lush decode reads no forward
    variable, so ``update-viterbi-path``'s placement on ``hmm-trainer`` was an
    artefact of where the corpus lived rather than a dependency.

    :param params: the model. Only its three arrays, ``n_symbols`` and
        ``n_states`` are read, the last for an error message.
    :param record: one ``dataseq`` ``SequenceRecord``. A record never holds
        padding, so there is no mask to consult. To decode many records at once,
        padded together, use :func:`viterbi_batch`.
    :param backend: which ADR 0002 lifecycle phase runs the decode -- ``"python"``
        (the reference, and the default), ``"cython"``, ``"cpu_parallel"`` or
        ``"cuda"``. All four are bit-exact with one another on a given host.
        Validated before any work (ADR 0021); see :func:`~pfsmgraph.hmm.backends`.
    :raises ValueError: if ``backend`` is not a backend name.
    :raises BackendUnavailableError: if ``backend`` cannot run in this
        environment. Nothing falls back.
    :raises ImpossibleSequenceError: if no path has finite description length.
    :raises ValueError: if a code falls outside the model's symbol axis. This is
        a range check only: a record encoded against a different vocabulary whose
        codes all fall in range decodes without error, against the wrong symbols.

    ``record.codes`` indexes ``output_p``'s third axis directly, with no offset
    anywhere -- that is what sizing that axis to ``vocabulary.size`` bought (see
    :class:`HMMParams`), and it is why ADR 0002's later phases have no index
    arithmetic to port.
    """
    kernel = _resolve("viterbi", backend)
    codes = record.codes
    message = _range_error(params, codes)
    if message is not None:
        raise ValueError(message)

    states, total_bits = kernel(
        params.init_state_p, params.transition_p, params.output_p, codes
    )

    if np.isinf(total_bits):
        raise ImpossibleSequenceError(_impossible_message(params, record))

    return ViterbiPath(states=states, total_bits=total_bits, label=record.label)


#: The values :func:`viterbi_batch`'s ``on_impossible`` accepts.
ON_IMPOSSIBLE = ("raise", "none")


def viterbi_batch(
    params: HMMParams,
    records,
    *,
    backend: BackendName = "python",
    batch_size: int | None = None,
    on_impossible: str = "raise",
) -> list[ViterbiPath | None]:
    """Decode every record in ``records`` under ``params``, one path per record.

    The result is :func:`viterbi` on each record in turn, in record order, **bit for
    bit** and at every ``batch_size``: the kernel decodes ``pad_collate``'s padded
    batch as one array rather than record by record, and a min-sum performs only
    ``+`` and ``<``, so batching moves no bit.

    :param params: the model, as for :func:`viterbi`.
    :param records: any iterable of ``dataseq`` ``SequenceRecord``, a
        ``SequenceDataset`` included. An empty record decodes to one state, as it
        does alone; no records decode to ``[]``, without a kernel call.
    :param backend: which ADR 0002 lifecycle phase runs the decode. Validated before
        any work (ADR 0021); see :func:`~pfsmgraph.hmm.backends`.
    :param batch_size: how many records each kernel call pads together; ``None``
        passes them all at once. It bounds memory, which grows as
        ``batch_size · L · S`` for the longest record ``L`` in a batch, and changes
        no result.
    :param on_impossible: ``"raise"``, the default, raises
        :class:`ImpossibleSequenceError` for the first record in record order that
        has no path of finite description length; ``"none"`` puts ``None`` in that
        record's place and decodes the rest. A search that decodes against many
        candidate models, where an impossible record is an ordinary outcome, wants
        the second.
    :raises ValueError: if ``backend`` is not a backend name or names a phase this
        call has not reached, if ``batch_size`` is below 1, if ``on_impossible`` is
        not one of :data:`ON_IMPOSSIBLE`, or if any record holds a code outside the
        model's symbol axis, naming the record's index. All are raised before any
        record is decoded.
    :raises BackendUnavailableError: if ``backend`` cannot run in this environment.
        Nothing falls back.
    :raises ImpossibleSequenceError: under ``on_impossible="raise"``, naming the
        record's index and the symbol at which every state became unreachable.
    """
    kernel = _resolve("viterbi_batch", backend)
    if on_impossible not in ON_IMPOSSIBLE:
        raise ValueError(
            f"on_impossible must be one of {list(ON_IMPOSSIBLE)}, got {on_impossible!r}"
        )
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1 or None, got {batch_size}")
    records = list(records)
    for index, record in enumerate(records):
        message = _range_error(params, record.codes)
        if message is not None:
            raise ValueError(f"record {index}: {message}")

    step = batch_size or max(len(records), 1)
    paths: list[ViterbiPath | None] = []
    for start in range(0, len(records), step):
        chunk = records[start : start + step]
        batch = pad_collate(chunk)
        states, total_bits = kernel(
            params.init_state_p,
            params.transition_p,
            params.output_p,
            batch["codes"],
            batch["lengths"],
        )
        for offset, record in enumerate(chunk):
            if np.isinf(total_bits[offset]):
                if on_impossible == "raise":
                    raise ImpossibleSequenceError(
                        f"record {start + offset}: {_impossible_message(params, record)}"
                    )
                paths.append(None)
            else:
                paths.append(
                    ViterbiPath(
                        states=states[offset, : record.length + 1],
                        total_bits=total_bits[offset],
                        label=record.label,
                    )
                )
    return paths
