"""Scored trials of a topology move for the revision 04 search: ``try-split``.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``. A trial builds one candidate with :mod:`._topology`, re-converges it
with :func:`~._baum_welch.baum_welch`, chooses its ``d`` and scores it. Ranking
trials against each other, and deciding whether a winner is accepted, belong to the
``suggest-*`` port and the search loop (master plan, revision ``04-hmm-v0.3.0``).

**``try-split`` (``hmm-trainer.lsh:749-770``) is followed step for step, and departs
in three places.** The candidate is ADR 0022's split, not ``split-state``'s. EM may
not stop before ``min_cycles``, since a split starts at the incumbent's likelihood
and the original's rule stops it before the twins separate (ADR 0022 section 4).
And the unrounded data description length is reported beside the total (section 5),
since the total moves with ``d``'s rounding by as much as the differences being
judged. What the original also did per trial and this does not -- a Viterbi path,
the trials file, the GUI's guards -- is recorded in the branch plan for
``feat/hmm-scored-primitives``, goal 1.

**The score is obtained by calling** :func:`~._mdl._total_description_length`, never
by assembling it, so a later change of criterion (PRD section 8) substitutes one
function. ``d`` comes from :func:`~._mdl._suggest_d`, the original's Brent search
from 3821 on every trial, not warm-started from the incumbent's ``d``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._baum_welch import baum_welch
from ._mdl import _suggest_d, _total_description_length
from ._params import HMMParams
from ._topology import _split_state

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
    :param batch_size: passed to :func:`~._baum_welch.baum_welch`.
    :returns: the ``S + 1``-state candidate, re-converged and scored.
    :raises ImpossibleSequenceError: when a record has no path under the incumbent,
        which the split preserves exactly before its seed.
    """
    records = list(records)
    candidate = _split_state(params, state, rng=rng)
    converged = baum_welch(
        candidate, records, backend=backend, batch_size=batch_size, min_cycles=min_cycles
    )
    d = _suggest_d(converged.params, records)
    return TrialResult(
        params=converged.params,
        d=d,
        total_bits=_total_description_length(converged.params, records, d),
        # baum_welch's last entry is the reference forward pass on every backend,
        # which is the corpus description length of the returned parameters.
        data_bits=converged.description_lengths[-1],
        cycles=converged.cycles,
        converged=converged.converged,
    )
