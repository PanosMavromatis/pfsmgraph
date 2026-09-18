"""The topology search: a walk over state merges and splits that keeps the best model.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``, and ADR 0025
(``docs/design/adr/0025-topology-search-not-exported.md``) keeps it that way at
0.3.0 -- the version ships this capability with no supported entry point. ADR 0023
(``docs/design/adr/0023-topology-search-loop.md``) is authoritative for everything
below.

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
from typing import Literal, TextIO

import numpy as np

from pfsmgraph.dataseq import USER_BASE, Vocabulary

from ._baum_welch import _write, baum_welch
from ._mdl import _model_description_length, _scan_d
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

#: The log's columns: label, field width, and how a value sits in it. One row per round,
#: so a round's outcome and what it chose among are read together rather than on two
#: lines of different shapes. Header and rows are both built from this, which is what
#: keeps them aligned: a width lives in one place. A width of 0 is the free last column.
#:
#: The first seven are ``training-log-line``'s (``hmm-trainer.lsh:462-474``) in its
#: order, less ``test-data-dl``, which the original never assigned -- all seven logged
#: lines read ``0``. The last three have no counterpart there and could not have: the
#: original's user ran one trial at a time by hand, so no round of it ever had
#: candidates to rank or a runner-up to lose to.
_COLUMNS = (
    ("Round", 5, ">"),  # the walk's step, 0 for the start; see _log_move_row
    ("Size", 4, ">"),
    ("Move", 9, "^"),
    ("Data DL", 9, "<"),
    ("Model DL", 9, "<"),
    ("Total DL", 9, "<"),
    ("d", 5, "<"),
    ("Candidates", 10, ">"),
    ("Runner-Up", 9, ">"),
    ("Comments", 0, "<"),
)


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
    log: TextIO | None = None,
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
    :param score_backend: the ``forward_backward`` backend for every forward pass: EM's
        convergence checks and every score.
    :param batch_size: passed to :func:`~._baum_welch.baum_welch`.
    :param log: a text stream to report progress on, such as ``sys.stdout``; ``None``
        reports nothing. One header line, one row for the starting model, **one row per
        round**, and a closing sentence naming the rule that stopped the search. A round
        that takes a move reports it, which is ``keep-model``'s rule -- it logged ``when
        topology-dirty``, and every move here changes the topology -- together with how
        many candidates it ranked and what the runner-up cost, neither of which the
        original could report.

        **The nested convergence log is silenced, deliberately.**
        :func:`~._baum_welch.baum_welch` takes a ``log=`` of its own and this never
        passes one down. A round re-converges every candidate under a 200-cycle floor,
        so forwarding the stream would emit a header, a row per convergence check and a
        stop line for each of dozens of *rejected* candidates, burying the one row that
        records what the search did. Nothing is withdrawn by this: ``baum_welch``'s
        ``log=`` is public, and a caller wanting an EM trace calls it directly.
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
    converged = baum_welch(
        params, records, backend=backend, score_backend=score_backend, batch_size=batch_size
    )
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

    if log is not None:
        _log_header(log)
        _log_start_row(log, start_result)

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
        # compared, and an infinite one is never taken. Its *position* is what is kept,
        # because the element after it is the runner-up the log reports and locating the
        # winner by ``==`` would compare two ``Move``s -- frozen dataclasses holding an
        # ``HMMParams``, whose arrays raise rather than answer.
        winner_at = next((at for at, move in enumerate(moves) if move.total_bits < np.inf), None)
        if winner_at is None:
            if log is not None:
                _log_dead_end(log, index, len(moves))
                _log_stop(log, "dead_end", rounds, best)
            return SearchResult(start_result, best, tuple(rounds), "dead_end")
        winner = moves[winner_at]
        improved = winner.total_bits < best.total_bits
        rounds.append(Round(index, winner, improved))
        incumbent = winner.result
        if improved:
            best, without_improvement = incumbent, 0
        else:
            without_improvement += 1
        # The row is written before the stop is taken, so a patience stop reports the
        # round that triggered it rather than dropping it.
        if log is not None:
            _log_move_row(log, index, moves, winner_at, without_improvement, patience)
        if not improved and without_improvement > patience:
            if log is not None:
                _log_stop(log, "patience", rounds, best)
            return SearchResult(start_result, best, tuple(rounds), "patience")
    if log is not None:
        _log_stop(log, "max_rounds", rounds, best)
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


def _marker(move: Move) -> str:
    """``training-log-line``'s move column (``hmm-trainer.lsh:464-468``).

    ``n ^`` for a split of state ``n``, ``i v j`` for a merge of ``i`` and ``j``. The
    starting model's row uses ``-``, the third form. The original padded each to
    thirteen characters; here :data:`_COLUMNS` centres them in nine, which holds three
    digits a side as the original's ``%3d v %-3d`` did.
    """
    if move.kind == "split":
        return f"{move.states[0]} ^"
    first, second = move.states
    return f"{first} v {second}"


def _halves(result: TrialResult) -> tuple[float, float]:
    """The data and model halves of ``result.total_bits``, both at its ``d``.

    **Not** ``result.data_bits`` and the remainder. That field is the corpus length of
    the *unrounded* parameters, which ``d`` never touched, so the difference would carry
    the whole quantization gap into the model column -- 1.5 to 29 bits on the three
    tracked models, against a model half that is 58 bits on the first logged line.
    :func:`~._mdl._model_description_length` reads no corpus at all, so taking the model
    half from it and the data half as ``total - model`` costs no second forward pass and
    agrees with the printed total by construction.
    """
    model_bits = _model_description_length(result.params, result.d)
    return result.total_bits - model_bits, model_bits


def _log_line(log: TextIO, cells) -> None:
    """Write one row, laying every cell out by :data:`_COLUMNS`.

    Trailing blanks are stripped, so a row that fills no late column ends where its
    content does rather than in invisible whitespace.
    """
    laid = (f"{cell:{align}{width}}" if width else cell
            for (_, width, align), cell in zip(_COLUMNS, cells))
    _write(log, ("  " + "  ".join(laid)).rstrip())


def _log_header(log: TextIO) -> None:
    """Name the columns, which the original's file did not.

    It wrote to a file read beside the model that produced it; this writes to a stream
    someone is watching, where ten unlabelled columns are a puzzle. The header is the
    one line here that the format oracles do not constrain.
    """
    _log_line(log, [label for label, _, _ in _COLUMNS])


def _log_start_row(log: TextIO, result: TrialResult) -> None:
    """The starting model's row: scored, but chosen among nothing.

    ``Round`` reads ``0`` -- the walk's step 0, before any move -- and ``Move`` the
    original's ``-``. The three columns the original has no counterpart for are left
    empty, since a start ranks nothing.
    """
    data_bits, model_bits = _halves(result)
    _log_line(log, [
        "0", f"{result.params.n_states}", "-",
        f"{data_bits:g}", f"{model_bits:g}", f"{result.total_bits:g}", f"{result.d:g}",
        "", "", "",
    ])


def _log_move_row(log, index, moves, winner_at, quiet, patience) -> None:
    """One round's row: the move it took, and what it took that move against.

    **The last three columns have no counterpart in the original and could not have
    had one.** Its user ran one trial at a time by hand, so ``keep-model`` recorded an
    accepted move with nothing to compare it against; a search scores every candidate
    of every round and would otherwise discard all of it.

    The runner-up's margin is the part worth keeping. ADR 0023's Open section records
    that near the optimum a candidate's total is a step function of where EM stopped,
    each integer step of ``d`` moving the model half by 58-71 bits at nine states, so
    two candidates closer than that are not separated by the criterion at all. A reader
    who can see the margin can see which regime the round was decided in.

    **The ``Round`` column is ``index + 1``, and that is deliberate.** It counts the
    walk's steps, of which step 0 is the starting model, so a reader follows one
    sequence down the column. ``SearchResult.rounds[r].index`` stays ``r``, because that
    number is also the round's *seed* key -- round ``r`` draws from
    ``spawn_key + (1, r)`` (ADR 0023 section 2) -- and renumbering it would silently
    re-seed every search. So the column and the field differ by one on purpose; the
    column is a row number, the field is a round number.
    """
    winner = moves[winner_at]
    result = winner.result
    data_bits, model_bits = _halves(result)
    runner_up = moves[winner_at + 1] if winner_at + 1 < len(moves) else None
    if runner_up is None or not np.isfinite(runner_up.total_bits):
        margin = "-"
    else:
        margin = f"{runner_up.total_bits - winner.total_bits:+.4g}"
    _log_line(log, [
        f"{index + 1}", f"{result.params.n_states}", _marker(winner),
        f"{data_bits:g}", f"{model_bits:g}", f"{result.total_bits:g}", f"{result.d:g}",
        f"{len(moves)}", margin, "" if quiet == 0 else f"no new best ({quiet}/{patience})",
    ])


def _log_dead_end(log: TextIO, index: int, candidates: int) -> None:
    """A round that ranked candidates and found none possible: no move, so no model columns.

    It still takes a row, and a step number with it, because the round happened and cost
    what a round costs. ``Round`` is ``index + 1`` for the reason :func:`_log_move_row`
    gives.
    """
    _log_line(log, [f"{index + 1}", "", "", "", "", "", "", f"{candidates}", "", "no move possible"])


def _log_stop(log: TextIO, stop: str, rounds, best: TrialResult) -> None:
    """The closing sentence, deliberately not a row.

    Breaking the table is the signal: a reader scanning rows sees at once that the walk
    ended. It names the best model because that need not be the last row -- the walk
    keeps each round's winner whatever its total, so the final row is the incumbent and
    the best may be several rounds above it.
    """
    _write(log, (
        f"  stopped: {stop} after {len(rounds)} round(s); best is "
        f"{best.total_bits:g} bits at {best.params.n_states} state(s)"
    ))
