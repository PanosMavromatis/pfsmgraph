"""Scored trials of topology moves for the revision 04 search, and their rankings.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``, and ADR 0025
(``docs/design/adr/0025-topology-search-not-exported.md``) keeps it that way at
0.3.0: ``TrialResult``'s ``d`` and ``data_bits`` would assert the two-part
description length that :func:`~._mdl._total_description_length` returns a bare
``float`` precisely to avoid asserting.

A trial builds one candidate with :mod:`._topology`, re-converges it
with :func:`~._baum_welch.baum_welch`, chooses its ``d`` and scores it:
``try-split`` and ``try-merge``. ``suggest-split``, ``suggest-merge`` and
``suggest-move`` run every trial of a round and rank them. Deciding whether the
winner is accepted belongs to the search loop (master plan, revision
``04-hmm-v0.3.0``).

**``try-split`` (``hmm-trainer.lsh:749-770``) is followed step for step, and departs
in three places.** The candidate is ADR 0022's split, not ``split-state``'s. EM may
not stop before ``min_cycles``, since a split starts at the incumbent's likelihood
and the original's rule stops it before the twins separate (ADR 0022 section 4).
And the unrounded data description length is reported beside the total (section 5),
since the total moves with ``d``'s rounding by as much as the differences being
judged. What the original also did per trial and this does not -- a Viterbi path,
the trials file, the GUI's guards -- is recorded in the branch plan for
``feat/hmm-scored-primitives``, goal 1.

**``try-merge`` (``:858-881``) is the same eleven lines with a different surgery**, so
both trials share :func:`_re_converge_and_score`. The candidate is
:func:`~._topology._merge_states`, whose three departures from ``merge-states`` its
docstring records, and its weights stay stationary mass rather than E-step occupancy:
on every tracked model, trained flat or per record, the two agree to 0.003 and no state
is transient (the branch plan, goal 3). A merge takes no minimum EM budget by default,
since unlike a split it does not start at the incumbent's likelihood.

**The ``suggest-*`` methods (``:952-1073``) keep only a strict-``<`` argmin; these
return every candidate, ranked.** Candidates are enumerated in the original's order --
states ascending then trials, pairs lexicographically, merges before splits -- and
sorted *stably* by total, so the first entry is the original's winner with its ties
broken the original's way, and the rest are the runners-up. Totals are compared, never
subtracted, so two ``+inf`` candidates tie rather than producing ``nan``. Three
departures follow from ranking rather than reducing. An impossible candidate, whether
``baum_welch`` refuses it or its total rounds to ``+inf``, is listed last with no
result, where the original's ``1e100`` initial best made it unselectable and an
all-impossible round crashed. A pair of two transient states is not a candidate and
is not listed, since :func:`~._topology._merge_states` has no weights for it. And each
split trial draws from its own generator, keyed by the round's ``SeedSequence`` and
the trial's ``(state, trial)`` identity (ADR 0022, Resolved), where the original drew
every split from one global stream. No verdict against the incumbent is given.

**The score is obtained by calling** :func:`~._mdl._total_description_length`, never
by assembling it, so a later change of criterion (PRD section 8) substitutes one
function. **``d`` comes from** :func:`~._mdl._scan_d`, **not the original's Brent
search** (ADR 0023 section 5): a bounded scan for the global integer minimum, bounded
by the trial's own unrounded data length, and run afresh on every trial.
:func:`~._mdl._suggest_d` stays as the checked reproduction of ``suggest-d``.
``backend`` runs EM and ``score_backend`` the forward passes inside EM's convergence
check and the scan; they are separate because ``forward_backward`` has no ``torch``
phase, and neither is ever substituted for the other (ADR 0021).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from ._baum_welch import baum_welch
from ._mdl import _scan_d
from ._numeric import closed_classes
from ._params import HMMParams
from ._topology import _merge_states, _split_state
from ._viterbi import ImpossibleSequenceError

#: ``*min-split-trials*`` and ``*split-trials-per-state-p*`` as the original shipped them
#: (``hmm-trainer.lsh:17-18``); its comment suggests 20 or 50 for the second "if we
#: want to be thorough".
MIN_SPLIT_TRIALS = 2
SPLIT_TRIALS_PER_STATE_P = 0.0

__all__: list[str] = []


@dataclass(frozen=True)
class TrialResult:
    """One re-converged, scored candidate.

    ``total_bits`` is the two-part score at ``d``; ``data_bits`` is the corpus
    description length of the unrounded ``params``, which ``d`` does not touch.
    ``cycles`` and ``converged`` are the re-convergence's, as
    :class:`~._baum_welch.BaumWelchResult` reports them.
    """

    params: HMMParams
    d: float
    total_bits: float
    data_bits: float
    cycles: int
    converged: bool


def _try_split(
    params: HMMParams,
    records,
    state: int,
    *,
    rng: np.random.Generator,
    min_cycles: int,
    backend: str = "python",
    score_backend: str = "python",
    batch_size: int | None = None,
) -> TrialResult:
    """Split ``state``, re-converge the candidate, choose its ``d`` and score it.

    :param params: the incumbent model, with ``S`` states.
    :param records: the corpus, any sequence of ``SequenceRecord``.
    :param state: the state to split, in ``range(S)``.
    :param rng: the trial's own generator, which only the split draws from.
        Building one per (round, state, trial) is the caller's (ADR 0022, Resolved).
    :param min_cycles: the fewest EM cycles before the stopping rule may fire.
        Required, because ADR 0022 leaves its size to the search loop.
    :param backend: passed to :func:`~._baum_welch.baum_welch`.
    :param score_backend: the ``forward_backward`` phase for EM's convergence check and
        for scoring each ``d``.
    :param batch_size: passed to :func:`~._baum_welch.baum_welch`.
    :returns: the ``S + 1``-state candidate, re-converged and scored.
    :raises ImpossibleSequenceError: when a record has no path under the incumbent,
        which the split preserves exactly before its seed.
    """
    records = list(records)
    return _re_converge_and_score(
        _split_state(params, state, rng=rng), records, min_cycles, backend, score_backend,
        batch_size,
    )


def _try_merge(
    params: HMMParams,
    records,
    first: int,
    second: int,
    *,
    min_cycles: int = 0,
    backend: str = "python",
    score_backend: str = "python",
    batch_size: int | None = None,
) -> TrialResult:
    """Merge ``first`` and ``second``, re-converge the candidate, choose its ``d`` and score it.

    :param params: the incumbent model, with ``S`` states.
    :param records: the corpus, any sequence of ``SequenceRecord``.
    :param first: one state of the pair; the order of the two does not matter.
    :param second: the other state, distinct from ``first``.
    :param min_cycles: the fewest EM cycles before the stopping rule may fire; ``0``,
        the default, is the original's rule.
    :param backend: passed to :func:`~._baum_welch.baum_welch`.
    :param score_backend: the ``forward_backward`` phase for EM's convergence check and
        for scoring each ``d``.
    :param batch_size: passed to :func:`~._baum_welch.baum_welch`.
    :returns: the ``S - 1``-state candidate, re-converged and scored.
    :raises ValueError: from :func:`~._topology._merge_states`, including for a pair of
        two transient states, which ``suggest-merge`` filters out before trying them.
    :raises ImpossibleSequenceError: when a record has no path under the candidate.
    """
    records = list(records)
    return _re_converge_and_score(
        _merge_states(params, first, second), records, min_cycles, backend, score_backend,
        batch_size,
    )


def _re_converge_and_score(candidate, records, min_cycles, backend, score_backend, batch_size):
    """What both trials do after their surgery: ``run-converge``, choosing ``d``, the score."""
    converged = baum_welch(
        candidate, records, backend=backend, score_backend=score_backend,
        batch_size=batch_size, min_cycles=min_cycles,
    )
    # baum_welch's last entry is its check on score_backend, bit-identical to the
    # reference forward pass, and so the corpus description length of the returned
    # parameters.
    data_bits = converged.description_lengths[-1]
    d, total_bits = _scan_d(converged.params, records, data_bits, backend=score_backend)
    return TrialResult(
        params=converged.params,
        d=d,
        total_bits=total_bits,
        data_bits=data_bits,
        cycles=converged.cycles,
        converged=converged.converged,
    )


@dataclass(frozen=True)
class Move:
    """One candidate of a ``suggest-*`` round.

    ``states`` is ``(state,)`` for a split and ``(a, b)`` with ``a < b`` for a merge;
    ``trial`` numbers a state's split trials from 0 and is 0 for a merge. ``result`` is
    ``None`` when ``baum_welch`` refused the candidate as impossible, and
    ``total_bits`` is then ``+inf``; otherwise it is ``result.total_bits``.
    """

    kind: Literal["merge", "split"]
    states: tuple[int, ...]
    trial: int
    result: TrialResult | None
    total_bits: float


def _suggest_split(
    params: HMMParams,
    records,
    *,
    seed: np.random.SeedSequence,
    min_cycles: int,
    min_split_trials: int = MIN_SPLIT_TRIALS,
    split_trials_per_state_p: float = SPLIT_TRIALS_PER_STATE_P,
    backend: str = "python",
    score_backend: str = "python",
    batch_size: int | None = None,
) -> list[Move]:
    """Every split trial of one round, ranked: ``suggest-split``.

    State ``s`` gets ``min_split_trials + int(split_trials_per_state_p * state_p[s])``
    trials, and trial ``t`` of it draws from
    ``default_rng(SeedSequence(seed.entropy, spawn_key=seed.spawn_key + (s, t)))``.

    :param seed: the round's seed; each trial's generator is derived from it and the
        trial's identity alone, so no trial depends on which others ran.
    :param min_cycles: passed to every :func:`_try_split`.
    :returns: the moves, stably sorted by ``total_bits``.
    """
    records = list(records)
    _check_round(seed, min_split_trials, split_trials_per_state_p)
    return _ranked(
        _split_moves(
            params, records, seed, min_cycles, min_split_trials,
            split_trials_per_state_p, backend, score_backend, batch_size,
        )
    )


def _suggest_merge(
    params: HMMParams,
    records,
    *,
    min_cycles: int = 0,
    backend: str = "python",
    score_backend: str = "python",
    batch_size: int | None = None,
) -> list[Move]:
    """Every merge of one round, ranked: ``suggest-merge``.

    Every pair ``a < b`` is tried except a pair of two transient states, which is left
    out. :returns: the moves, stably sorted by ``total_bits``.
    """
    records = list(records)
    return _ranked(
        _merge_moves(params, records, min_cycles, backend, score_backend, batch_size)
    )


def _suggest_move(
    params: HMMParams,
    records,
    *,
    seed: np.random.SeedSequence,
    split_min_cycles: int,
    merge_min_cycles: int = 0,
    min_split_trials: int = MIN_SPLIT_TRIALS,
    split_trials_per_state_p: float = SPLIT_TRIALS_PER_STATE_P,
    backend: str = "python",
    score_backend: str = "python",
    batch_size: int | None = None,
) -> list[Move]:
    """Every merge and every split trial of one round, ranked together: ``suggest-move``.

    Merges are enumerated before splits and the sort is stable, so a merge wins a tie
    with a split, as it does under the original's shared strict ``<``.
    """
    records = list(records)
    _check_round(seed, min_split_trials, split_trials_per_state_p)
    return _ranked(
        _merge_moves(params, records, merge_min_cycles, backend, score_backend, batch_size)
        + _split_moves(
            params, records, seed, split_min_cycles, min_split_trials,
            split_trials_per_state_p, backend, score_backend, batch_size,
        )
    )


def _split_moves(
    params, records, seed, min_cycles, min_split_trials, split_trials_per_state_p,
    backend, score_backend, batch_size,
):
    moves = []
    state_p = params.state_p
    for state in range(params.n_states):
        for trial in range(min_split_trials + int(split_trials_per_state_p * state_p[state])):
            rng = np.random.default_rng(
                np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key + (state, trial))
            )
            moves.append(
                _attempt(
                    "split", (state,), trial,
                    lambda: _try_split(
                        params, records, state, rng=rng, min_cycles=min_cycles,
                        backend=backend, score_backend=score_backend, batch_size=batch_size,
                    ),
                )
            )
    return moves


def _merge_moves(params, records, min_cycles, backend, score_backend, batch_size):
    transient = closed_classes(params.transition_p) < 0
    return [
        _attempt(
            "merge", (a, b), 0,
            lambda: _try_merge(
                params, records, a, b, min_cycles=min_cycles,
                backend=backend, score_backend=score_backend, batch_size=batch_size,
            ),
        )
        for a, b in combinations(range(params.n_states), 2)
        if not (transient[a] and transient[b])
    ]


def _attempt(kind, states, trial, run):
    """Run one trial; an impossible candidate becomes a move with no result at ``+inf``."""
    try:
        result = run()
    except ImpossibleSequenceError:
        return Move(kind, states, trial, None, np.inf)
    return Move(kind, states, trial, result, result.total_bits)


def _ranked(moves):
    """Stable by ``total_bits``: ties keep enumeration order, and totals are only compared."""
    return sorted(moves, key=lambda move: move.total_bits)


def _check_round(seed, min_split_trials, split_trials_per_state_p):
    if not isinstance(seed, np.random.SeedSequence):
        raise TypeError(f"seed must be a numpy.random.SeedSequence, got {type(seed).__name__}")
    if isinstance(min_split_trials, bool) or not isinstance(min_split_trials, (int, np.integer)):
        raise TypeError(f"min_split_trials must be an integer, got {type(min_split_trials).__name__}")
    if min_split_trials < 0:
        raise ValueError(f"min_split_trials must be non-negative, got {min_split_trials}")
    if not (np.isfinite(split_trials_per_state_p) and split_trials_per_state_p >= 0):
        raise ValueError(
            f"split_trials_per_state_p must be finite and non-negative, got {split_trials_per_state_p}"
        )
