"""The topology search: a walk over state merges and splits that keeps the best model.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``. [ADR 0023](../../../../../docs/design/adr/0023-topology-search-loop.md) is
authoritative for everything below.

**This is a port, and the original is not the GUI.** ``Training/hmm-train-new-nw`` and
``hmm-train-load-nw`` build a trainer and run ::

    (repeat N
      (==> trainer suggest-move)
      (==> trainer keep-model))

which the view's ``Continue for`` button runs too. The trainer's constructor converges
and scores the starting model first (``hmm-trainer.lsh:58-100``); ``suggest-move``
(``:952-1073``) tries every merge and every split trial of the incumbent and keeps the
best; ``keep-model`` (``:677-691``) makes it the incumbent whatever its total, and the
loop ends after ``N`` moves. :func:`_search` does the same, round for round, through
:func:`~._trials._suggest_move`.

**What it does that the original left to a person is a departure, and is named as
one.** The original saved every kept model and let a person choose among them; this
keeps the best model visited by strict ``<`` and returns it (ADR 0023 section 3). The
original stopped only at ``N``; this also stops after ``patience`` rounds without a new
best, and when a round has no possible move, where the original set the model to size
0 (section 4). Seeds are keyed by role rather than drawn from one stream (section 2).
And the trials it calls choose ``d`` by a bounded scan rather than Brent's method, and
give a split 200 EM cycles by default (sections 5 and 6). A reviewer checking this
against the original should expect exactly those differences and no others.

**No rollback exists, and none is needed.** ``HMMParams`` is a frozen value (ADR
0017), so a rejected candidate is simply not kept and the incumbent is a reference,
where the original's ``reset-model`` copied a working copy back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pfsmgraph.dataseq import USER_BASE, Vocabulary

from ._baum_welch import baum_welch
from ._mdl import _scan_d
from ._numeric import rand_p_vector
from ._params import HMMParams
from ._trials import MIN_SPLIT_TRIALS, SPLIT_TRIALS_PER_STATE_P, Move, TrialResult, _suggest_move

__all__: list[str] = []

#: ``init-random``'s noise width for a new model (``hmm.lsh:89``).
START_NOISE_WIDTH = 0.001

#: The split's minimum EM budget (ADR 0022 section 4), sized by ADR 0023 section 6.
SPLIT_MIN_CYCLES = 200

#: The ``spawn_key`` components that give each draw of a search its role.
_START_KEY, _ROUND_KEY = 0, 1


@dataclass(frozen=True)
class Round:
    """One round of a search: its index, the move it took, and whether that set a new best."""

    index: int
    move: Move
    improved: bool


@dataclass(frozen=True)
class SearchResult:
    """What :func:`_search` returns.

    ``start`` is the starting model, converged and scored. ``best`` is the lowest-total
    model visited, ``start`` included, and is the object from ``start`` or from a
    round's ``move.result`` rather than a copy. ``rounds`` holds one entry per round
    that took a move, in order; the view's history pane showed the same. ``stop``
    names the rule that ended the search.
    """

    start: TrialResult
    best: TrialResult
    rounds: tuple[Round, ...]
    stop: Literal["max_rounds", "patience", "dead_end"]


def _one_state_model(vocabulary, rng) -> HMMParams:
    """The model ``hmm-train-new-nw`` starts from: one state, drawn by ``init-random``.

    Draws in ``init-random``'s order (``hmm.lsh:217-225``): the initial distribution,
    then the single transition row, then its one emission fibre over the user
    symbols, each by :func:`~._numeric.rand_p_vector` at noise width 0.001. The first
    two have one element and are exactly ``[1.0]``, but still draw. The ADR 0011
    reserved fibres are zero, which ``HMMParams`` requires.

    With one state the draws cannot change what EM converges to: every position is in
    that state with posterior 1, so the counts do not depend on the parameters, and the
    first M-step lands on the corpus's symbol frequencies from any start.
    """
    init_state_p = rand_p_vector(1, START_NOISE_WIDTH, rng)
    transition_p = rand_p_vector(1, START_NOISE_WIDTH, rng).reshape(1, 1)
    output_p = np.zeros((1, 1, vocabulary.size), dtype=np.float64)
    output_p[0, 0, USER_BASE:] = rand_p_vector(vocabulary.size - USER_BASE, START_NOISE_WIDTH, rng)
    return HMMParams(init_state_p, transition_p, output_p, vocabulary)


def _search(
    records,
    *,
    start: HMMParams | Vocabulary,
    seed: np.random.SeedSequence,
    max_rounds: int,
    patience: int,
    split_min_cycles: int = SPLIT_MIN_CYCLES,
    merge_min_cycles: int = 0,
    min_split_trials: int = MIN_SPLIT_TRIALS,
    split_trials_per_state_p: float = SPLIT_TRIALS_PER_STATE_P,
    backend: str = "python",
    score_backend: str = "python",
    batch_size: int | None = None,
) -> SearchResult:
    """Walk the topology from ``start`` by merges and splits, and return the best model visited.

    The start is converged by :func:`~._baum_welch.baum_welch` under the default rule
    and scored at the ``d`` :func:`~._mdl._scan_d` chooses. Round ``r`` then ranks every
    move of the incumbent with :func:`~._trials._suggest_move` and takes the first with
    a finite total, which becomes the incumbent exactly as its trial left it, whether or
    not its total is lower. It is also the new best when
    ``move.total_bits < best.total_bits``; a tie is not an improvement.

    The search stops at the first of:

    - ``"dead_end"``: a round with no move of finite total, which records no round;
    - ``"patience"``: a round that leaves ``patience + 1`` consecutive rounds without a
      new best, so ``patience=0`` stops at the first round that does not improve, which
      is greedy improvement exactly;
    - ``"max_rounds"``: ``max_rounds`` rounds have run.

    :param records: the corpus, any sequence of ``SequenceRecord``.
    :param start: an ``HMMParams`` to resume from, as ``hmm-train-load-nw`` does, or a
        ``Vocabulary`` to build the one-state model over, as ``hmm-train-new-nw`` does.
    :param seed: the search's seed. The one-state start draws from
        ``spawn_key = seed.spawn_key + (0,)``, and round ``r`` passes
        ``SeedSequence(seed.entropy, spawn_key=seed.spawn_key + (1, r))`` to
        ``_suggest_move``, so split trial ``(s, t)`` of round ``r`` draws from
        ``seed.spawn_key + (1, r, s, t)``. Every trial is rebuildable from its identity.
    :param max_rounds: the most rounds to run, at least 0; required, so that termination
        never depends on the data. ``0`` scores the start alone.
    :param patience: how many consecutive rounds without a new best are tolerated, at
        least 0. Required: nothing has measured a good value.
    :param split_min_cycles: the EM floor of every split trial (ADR 0023 section 6).
    :param merge_min_cycles: the EM floor of every merge trial; ``0`` is the original's rule.
    :param min_split_trials: split trials per state, as ``*min-split-trials*``.
    :param split_trials_per_state_p: further split trials per unit of ``state_p``.
    :param backend: the ``baum_welch`` backend for every EM run.
    :param score_backend: the ``forward_backward`` backend for every score.
    :param batch_size: passed to :func:`~._baum_welch.baum_welch`.
    :raises TypeError: for a ``start`` that is neither, a ``seed`` that is not a
        ``SeedSequence``, or a round count that is not an integer.
    :raises ValueError: for a negative ``max_rounds`` or ``patience``.
    :raises ImpossibleSequenceError: when a record has no path under the start.
    """
    records = list(records)
    _check_search(start, seed, max_rounds, patience)
    if isinstance(start, HMMParams):
        params = start
    else:
        start_rng = np.random.default_rng(
            np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key + (_START_KEY,))
        )
        params = _one_state_model(start, start_rng)
    converged = baum_welch(params, records, backend=backend, batch_size=batch_size)
    data_bits = converged.description_lengths[-1]
    d, total_bits = _scan_d(converged.params, records, data_bits, backend=score_backend)
    start_result = TrialResult(
        params=converged.params,
        d=d,
        total_bits=total_bits,
        data_bits=data_bits,
        cycles=converged.cycles,
        converged=converged.converged,
    )

    incumbent = best = start_result
    rounds: list[Round] = []
    without_improvement = 0
    for index in range(max_rounds):
        moves = _suggest_move(
            incumbent.params,
            records,
            seed=np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key + (_ROUND_KEY, index)),
            split_min_cycles=split_min_cycles,
            merge_min_cycles=merge_min_cycles,
            min_split_trials=min_split_trials,
            split_trials_per_state_p=split_trials_per_state_p,
            backend=backend,
            score_backend=score_backend,
            batch_size=batch_size,
        )
        # Ranked, so the first finite total is the round's winner; totals are only
        # compared, and an infinite one is never taken.
        winner = next((move for move in moves if move.total_bits < np.inf), None)
        if winner is None:
            return SearchResult(start_result, best, tuple(rounds), "dead_end")
        improved = winner.total_bits < best.total_bits
        rounds.append(Round(index, winner, improved))
        incumbent = winner.result
        if improved:
            best, without_improvement = incumbent, 0
        else:
            without_improvement += 1
            if without_improvement > patience:
                return SearchResult(start_result, best, tuple(rounds), "patience")
    return SearchResult(start_result, best, tuple(rounds), "max_rounds")


def _check_search(start, seed, max_rounds, patience):
    if not isinstance(start, HMMParams) and not isinstance(start, Vocabulary):
        raise TypeError(
            f"start must be an HMMParams or a Vocabulary, got {type(start).__name__}"
        )
    if not isinstance(seed, np.random.SeedSequence):
        raise TypeError(f"seed must be a numpy.random.SeedSequence, got {type(seed).__name__}")
    for name, value in (("max_rounds", max_rounds), ("patience", patience)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
