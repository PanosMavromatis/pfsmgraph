"""Baum-Welch re-estimation over :mod:`._forward_backward`'s expected counts.

The E-step is :func:`~._forward_backward._expected_counts`; this module turns its
three count arrays into new parameters, and :func:`_em` alternates the two until
``run-converge``'s stopping rule fires. It is kept out of ``_forward_backward.py``
because that module is the dynamic-programming kernel ADR 0002's phases
transliterate, and the M-step has no recurrence to transliterate: it is two
reductions and three elementwise divisions.

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

**The loop trains on records, not on one stream.** Counts are summed over every
record before each M-step, which is multi-sequence Baum-Welch; the original's
flat concatenated corpus is the special case of a single record, and trains
identically. The stopping rule is the original's and so is its weakness: it
watches only the size of each batch's change, so a start close enough to a
symmetric saddle stops *on* the saddle, and an exactly uniform start is one.

**The data description length at a precision ``d`` is the same forward pass over
rounded parameters**, which is what the original's ``update-data-dl`` is. Rounding
is what makes precision cost bits, so the value means nothing apart from ``d``;
choosing ``d`` belongs with the model description length in revision 04.

.. [ADR 0002] ``docs/design/adr/0002-three-phase-algorithm-lifecycle.md``
.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._forward_backward import (
    _description_length,
    _expected_counts,
    _forward_backward,
)
from ._numeric import safe_divide
from ._params import HMMParams
from ._viterbi import ImpossibleSequenceError

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


#: ``run-converge``'s constants (``hmm-trainer.lsh:661-674``), with ``*factor*``
#: at its value of 1.0 (``hmm-trainer.lsh:10``).
BATCH_CYCLES = 10
CHANGE_BITS = 0.1
PATIENCE = 3


@dataclass(frozen=True)
class _EMResult:
    """What :func:`_em` returns. Private, as the loop is.

    ``description_lengths[c]`` is the corpus description length, in bits, of the
    model after ``c`` cycles, so it has ``cycles + 1`` entries and its first is
    the starting model's. ``degenerate_states`` are the states whose previous
    row and fibres the final cycle restored.
    """

    params: HMMParams
    description_lengths: tuple[float, ...]
    cycles: int
    converged: bool
    degenerate_states: tuple[int, ...]


def _check_codes(params, records):
    for index, record in enumerate(records):
        codes = record.codes
        if codes.size:
            lowest, highest = int(codes.min()), int(codes.max())
            if lowest < 0 or highest >= params.n_symbols:
                raise ValueError(
                    f"record {index} holds code(s) outside the model's symbol axis "
                    f"[0, {params.n_symbols}): observed [{lowest}, {highest}]. The "
                    f"usual cause is a record encoded against a different vocabulary "
                    f"than the one output_p was sized against"
                )


def _corpus_step(params, records):
    """One E-step over every record: summed counts and summed description length.

    Records are visited in order and their counts added elementwise, so the sum
    over records has the same fixed ascending order as the sums inside one.
    """
    size, n_symbols = params.n_states, params.n_symbols
    init_counts = np.zeros(size, dtype=np.float64)
    transition_counts = np.zeros((size, size), dtype=np.float64)
    emission_counts = np.zeros((size, size, n_symbols), dtype=np.float64)
    bits_per_record = np.zeros(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        alpha, beta, scale = _forward_backward(
            params.init_state_p, params.transition_p, params.output_p, record.codes
        )
        bits_per_record[index] = _description_length(scale)
        counts = _expected_counts(
            alpha, beta, params.transition_p, params.output_p, record.codes
        )
        init_counts += counts[0]
        transition_counts += counts[1]
        emission_counts += counts[2]
    total = float(np.add.accumulate(bits_per_record)[-1]) if len(records) else 0.0
    return (init_counts, transition_counts, emission_counts), bits_per_record, total


def _corpus_description_length(init_state_p, transition_p, output_p, records):
    """Bits over every record, from arrays rather than an ``HMMParams``.

    Arrays because :func:`_data_description_length` passes rounded parameters,
    which need not be a valid model: a row can round to all zeros.
    """
    bits_per_record = np.zeros(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        _, _, scale = _forward_backward(
            init_state_p, transition_p, output_p, record.codes
        )
        bits_per_record[index] = _description_length(scale)
    return float(np.add.accumulate(bits_per_record)[-1]) if len(records) else 0.0


def _quantize(p, d):
    """Round each probability to a multiple of ``1 / d`` and renormalise the last axis.

    ``update-approx-init-state-p``, ``-transition-p`` and ``-output-p``
    (``hmm-trainer.lsh:269-341``) in one function: ``round-using``
    (``util.lsh:59-62``) is ``int(x * d + 0.5) / d``, which for the non-negative
    values here is ``floor``, so an exact half rounds up. Each vector is then
    divided by its own sum, taken ascending, through ``safe_divide``.

    The result is arrays, never an ``HMMParams``, and nothing is checked: a vector
    whose every entry is below ``1 / (2 d)`` rounds to all zeros and stays zero,
    which is how coarse precision can make a record impossible. ``d`` need not be
    an integer, as in the original.
    """
    p = np.asarray(p, dtype=np.float64)
    rounded = np.floor(p * d + 0.5) / d
    total = np.add.accumulate(rounded, axis=-1)[..., -1:]
    return safe_divide(rounded, total)


def _data_description_length(params, records, d):
    """The data description length at precision ``d``: ``update-data-dl``.

    The original runs its forward pass again over the rounded ``-r`` matrices
    (``hmm-trainer.lsh:346-402``); here the same corpus description length the EM
    loop uses is called on :func:`_quantize`'s arrays, so there is one forward
    pass, not two. It is the data half of revision 04's two-part score, where
    ``d`` is chosen against the model half; this function takes ``d`` as given.

    ``+inf`` when rounding leaves some record with no path, which the original
    reports as ``1e100`` through its ``-1`` sentinel (``update-total-dl``). The
    original's final ``bits`` of the last column's sum is 1 within rounding and is
    omitted, as it is in :func:`_em`. Raises ``ValueError`` for a ``d`` that is
    not positive and finite, and for a code outside the symbol axis.
    """
    if not (np.isfinite(d) and d > 0):
        raise ValueError(f"d must be positive and finite, got {d}")
    records = list(records)
    _check_codes(params, records)
    return _corpus_description_length(
        _quantize(params.init_state_p, d),
        _quantize(params.transition_p, d),
        _quantize(params.output_p, d),
        records,
    )


def _em(
    params,
    records,
    *,
    batch_cycles=BATCH_CYCLES,
    change_bits=CHANGE_BITS,
    patience=PATIENCE,
    max_cycles=None,
):
    """Baum-Welch from ``params`` over ``records`` until ``run-converge`` stops.

    ``run-converge`` runs ``batch_cycles`` cycles, recomputes the description
    length, and counts the batch as unchanged when it moved by less than
    ``change_bits``; ``patience`` unchanged batches in a row stop the loop, and a
    changed batch resets the count. The defaults are the original's. The loop
    terminates without ``max_cycles``, since the description length never rises
    and is bounded below by zero, so its changes cannot stay at or above
    ``change_bits`` forever; ``max_cycles`` is for callers that want a budget,
    and reaching it returns with ``converged`` false.

    ``records`` is any sequence of ``SequenceRecord``, a ``SequenceDataset``
    included. The counts are summed over records before the M-step, so a single
    record holding a whole concatenated corpus trains exactly as the original's
    one flat stream does.

    A state with no outgoing count after an M-step keeps its previous row and
    fibres (see this module's docstring). Raises ``ValueError`` for a code outside
    the symbol axis, for a corpus with no symbols in it, and for a stopping rule
    that cannot stop; :class:`ImpossibleSequenceError` when a record has no path
    under the starting model, since nothing can then be re-estimated from it.
    """
    records = list(records)
    if batch_cycles < 1 or patience < 1:
        raise ValueError(
            f"batch_cycles and patience must be at least 1, got {batch_cycles} "
            f"and {patience}"
        )
    if not change_bits > 0:
        raise ValueError(
            f"change_bits must be positive, got {change_bits}: a change is never "
            f"below zero bits, so the loop could never stop"
        )
    if max_cycles is not None and max_cycles < 0:
        raise ValueError(f"max_cycles must be non-negative, got {max_cycles}")
    _check_codes(params, records)
    if not any(record.length for record in records):
        raise ValueError("the corpus holds no symbols, so there is nothing to count")

    history = []
    degenerate = ()
    unchanged = 0
    old_bits = None
    cycles = 0
    while unchanged < patience:
        for _ in range(batch_cycles):
            if max_cycles is not None and cycles >= max_cycles:
                return _finish(params, history, records, cycles, False, degenerate)
            counts, bits_per_record, total = _corpus_step(params, records)
            if cycles == 0:
                _raise_if_impossible(params, records, bits_per_record)
                old_bits = total
            history.append(total)
            params, degenerate = _re_estimate(params, counts)
            cycles += 1
        new_bits = _corpus_description_length(
            params.init_state_p, params.transition_p, params.output_p, records
        )
        if abs(new_bits - old_bits) < change_bits:
            unchanged += 1
        else:
            unchanged = 0
        old_bits = new_bits
    history.append(new_bits)
    return _EMResult(params, tuple(history), cycles, True, degenerate)


def _finish(params, history, records, cycles, converged, degenerate):
    history.append(
        _corpus_description_length(
            params.init_state_p, params.transition_p, params.output_p, records
        )
    )
    return _EMResult(params, tuple(history), cycles, converged, degenerate)


def _raise_if_impossible(params, records, bits_per_record):
    for index, bits in enumerate(bits_per_record):
        if np.isinf(bits):
            raise ImpossibleSequenceError(
                f"record {index} has no path of finite description length under "
                f"the starting model of {params.n_states} state(s), so every one "
                f"of its expected counts is zero and it cannot be trained on"
            )


def _re_estimate(params, counts):
    init_state_p, transition_p, output_p, out_counts = _m_step(*counts)
    degenerate = np.flatnonzero(out_counts == 0.0)
    transition_p[degenerate] = params.transition_p[degenerate]
    output_p[degenerate] = params.output_p[degenerate]
    new = HMMParams(init_state_p, transition_p, output_p, params.vocabulary)
    return new, tuple(int(state) for state in degenerate)
