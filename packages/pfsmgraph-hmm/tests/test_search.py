"""Tests for ``_search``: the topology search walk (ADR 0023).

**Every test here reaches a private function**, labelled rather than hidden as ADR 0003
asks: nothing in ``_search`` is exported yet.

The walk has no oracle for its bits, since ADR 0022 changed the split and ADR 0023 the
choice of ``d``, so most of this file tests properties and constructed cases:

- **The walk's rules, against a scripted round.** ``_suggest_move`` is replaced by a
  script of ranked moves, so acceptance, patience, the dead end and the round cap are
  checked exactly and in isolation from EM. Randomised scripts check the properties
  that must hold for every sequence of totals: the best never rises, a round that does
  not improve leaves the best as the same object, the walk always follows its winner,
  and no search runs past ``max_rounds``.
- **The draw contract.** Every generator is keyed from the search's seed by role, so
  the start and each round are rebuilt from an independent ``SeedSequence`` and
  compared.
- **The start.** The one-state model's draws, and that its converged parameters do not
  depend on them.
- **One real search**, which is also the one oracle there is: its first three rounds
  choose the same moves as the Lush training log of ``m001_0005_005``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, baum_welch
from pfsmgraph.hmm import _search as search_module
from pfsmgraph.hmm._mdl import _scan_d
from pfsmgraph.hmm._numeric import rand_p_vector
from pfsmgraph.hmm._search import (
    SPLIT_MIN_CYCLES,
    START_NOISE_WIDTH,
    Round,
    SearchResult,
    _one_state_model,
    _search,
)
from pfsmgraph.hmm._trials import Move, TrialResult

from _lush_fixtures import FIXTURES, load_corpus_record, load_params

SEED = 20260917
N_USER = 3
VOCABULARY = SymbolTable([f"s{i}" for i in range(N_USER)])


def _corpus(seed=SEED):
    rng = np.random.default_rng(seed)
    return [SequenceRecord(rng.integers(USER_BASE, USER_BASE + N_USER, n)) for n in (40, 25)]


# ---------------------------------------------------------------------------
# The walk's rules, against a scripted round.


class _Script:
    """Stands in for ``_suggest_move``: returns scripted ranked moves and records each call.

    ``rounds[r]`` is a list of totals for round ``r``; ``None`` is a move with no result.
    Each move with a result carries a fresh sentinel as its params, so identity shows
    which model the walk moved to.
    """

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    def __call__(self, params, records, *, seed, **keywords):
        index = len(self.calls)
        self.calls.append((params, seed, keywords))
        moves = []
        for trial, total in enumerate(self.rounds[index]):
            if total is None:
                moves.append(Move("split", (0,), trial, None, np.inf))
            else:
                result = TrialResult(object(), 1.0, total, total, 0, True)
                moves.append(Move("split", (0,), trial, result, total))
        return sorted(moves, key=lambda move: move.total_bits)


@pytest.fixture(scope="module")
def start_total():
    records = _corpus()
    return _search(records, start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=0, patience=0)


def _run(monkeypatch, rounds, *, max_rounds, patience):
    script = _Script(rounds)
    monkeypatch.setattr(search_module, "_suggest_move", script)
    # Cython for speed alone: its scores are bit-identical to numpy's (ADR 0023 section 7).
    result = _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED),
        max_rounds=max_rounds, patience=patience, backend="cython", score_backend="cython",
    )
    return result, script


def test_max_rounds_zero_scores_the_start_alone(start_total):
    assert start_total.rounds == ()
    assert start_total.stop == "max_rounds"
    assert start_total.best is start_total.start
    assert start_total.start.params.n_states == 1


def test_the_walk_follows_each_winner_even_when_it_is_worse(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, script = _run(monkeypatch, [[s - 10], [s - 5], [s - 20]], max_rounds=3, patience=5)
    winners = [round_.move.result for round_ in result.rounds]
    # Round r + 1 is proposed from round r's winner, whatever its total was.
    assert [call[0] for call in script.calls[1:]] == [winner.params for winner in winners[:2]]
    assert script.calls[0][0] is result.start.params
    assert [round_.improved for round_ in result.rounds] == [True, False, True]
    assert result.best is winners[2]
    assert result.stop == "max_rounds"


def test_the_winner_is_the_first_move_with_a_finite_total(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[None, None, s - 1, s - 2]], max_rounds=1, patience=0)
    (round_,) = result.rounds
    assert round_.move.total_bits == s - 2
    assert round_.move.result is result.best


def test_a_result_scoring_inf_is_not_a_winner(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[np.inf, None, s + 3]], max_rounds=1, patience=0)
    assert result.rounds[0].move.total_bits == s + 3


@pytest.mark.parametrize("round_", [[], [None], [None, np.inf], [np.inf]])
def test_a_round_with_no_finite_move_is_a_dead_end_and_records_no_round(monkeypatch, start_total, round_):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[s - 1], round_], max_rounds=5, patience=5)
    assert result.stop == "dead_end"
    assert len(result.rounds) == 1
    assert result.best is result.rounds[0].move.result


def test_a_tie_with_the_best_is_not_an_improvement(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[s]], max_rounds=1, patience=0)
    assert result.rounds[0].improved is False
    assert result.best is result.start
    assert result.stop == "patience"


def test_patience_zero_is_greedy_and_stops_at_the_first_round_that_does_not_improve(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, script = _run(monkeypatch, [[s - 1], [s - 2], [s + 1], [s - 9]], max_rounds=10, patience=0)
    assert len(script.calls) == 3
    assert result.stop == "patience"
    assert result.best.total_bits == s - 2


@pytest.mark.parametrize("patience", [1, 2, 3])
def test_patience_tolerates_that_many_rounds_without_a_new_best(monkeypatch, start_total, patience):
    s = start_total.start.total_bits
    rounds = [[s - 1]] + [[s + k] for k in range(1, 10)]
    result, script = _run(monkeypatch, rounds, max_rounds=10, patience=patience)
    assert len(script.calls) == 1 + patience + 1
    assert result.stop == "patience"


def test_a_new_best_resets_patience(monkeypatch, start_total):
    s = start_total.start.total_bits
    rounds = [[s + 1], [s - 1], [s + 1], [s - 2], [s + 1], [s + 1]]
    result, script = _run(monkeypatch, rounds, max_rounds=6, patience=1)
    assert len(script.calls) == 6
    assert result.stop == "patience"
    assert result.best.total_bits == s - 2


def test_randomised_walks_keep_every_invariant(monkeypatch, start_total):
    s = start_total.start.total_bits
    rng = np.random.default_rng(SEED)
    for _ in range(300):
        max_rounds = int(rng.integers(0, 12))
        patience = int(rng.integers(0, 4))
        rounds = []
        for _ in range(max_rounds):
            size = int(rng.integers(0, 4))
            totals = [
                None if rng.random() < 0.15 else float(s + rng.integers(-6, 7)) for _ in range(size)
            ]
            rounds.append(totals)
        result, script = _run(monkeypatch, rounds, max_rounds=max_rounds, patience=patience)

        assert isinstance(result, SearchResult)
        assert len(result.rounds) <= max_rounds
        assert [r.index for r in result.rounds] == list(range(len(result.rounds)))
        best = result.start
        without = 0
        for round_ in result.rounds:
            assert isinstance(round_, Round)
            ranked = sorted(t for t in rounds[round_.index] if t is not None)
            assert round_.move.total_bits == ranked[0]
            assert round_.improved == (round_.move.total_bits < best.total_bits)
            if round_.improved:
                best, without = round_.move.result, 0
            else:
                without += 1
            assert without <= patience or round_ is result.rounds[-1]
        # The best never rises, and a round that did not improve left it the same object.
        assert result.best is best
        assert result.best.total_bits <= result.start.total_bits
        if result.stop == "dead_end":
            assert not any(t is not None for t in rounds[len(result.rounds)])
        elif result.stop == "patience":
            assert without == patience + 1
        else:
            assert len(result.rounds) == max_rounds


def test_every_option_reaches_every_round(monkeypatch, start_total):
    s = start_total.start.total_bits
    script = _Script([[s - 1], [s - 2]])
    monkeypatch.setattr(search_module, "_suggest_move", script)
    _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=2, patience=0,
        split_min_cycles=7, merge_min_cycles=3, min_split_trials=5, split_trials_per_state_p=1.5,
        backend="python", score_backend="python", batch_size=4,
    )
    expected = dict(
        split_min_cycles=7, merge_min_cycles=3, min_split_trials=5, split_trials_per_state_p=1.5,
        backend="python", score_backend="python", batch_size=4,
    )
    assert [call[2] for call in script.calls] == [expected, expected]


def test_the_default_split_floor_is_adr_0023s(monkeypatch, start_total):
    s = start_total.start.total_bits
    script = _Script([[s - 1]])
    monkeypatch.setattr(search_module, "_suggest_move", script)
    _search(_corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=1, patience=0)
    assert SPLIT_MIN_CYCLES == 200
    assert script.calls[0][2]["split_min_cycles"] == 200
    assert script.calls[0][2]["merge_min_cycles"] == 0


# ---------------------------------------------------------------------------
# The draw contract.


@pytest.mark.parametrize("spawn_key", [(), (4,), (2, 9)])
def test_round_r_is_seeded_from_the_role_key_under_the_searchs_own(monkeypatch, start_total, spawn_key):
    s = start_total.start.total_bits
    script = _Script([[s - 1], [s + 1], [s + 2]])
    monkeypatch.setattr(search_module, "_suggest_move", script)
    seed = np.random.SeedSequence(SEED, spawn_key=spawn_key)
    _search(_corpus(), start=VOCABULARY, seed=seed, max_rounds=3, patience=5)
    for index, (_, round_seed, _) in enumerate(script.calls):
        assert round_seed.entropy == SEED
        assert round_seed.spawn_key == spawn_key + (1, index)


def test_the_one_state_start_draws_from_the_start_key_in_init_random_order():
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,)))
    params = _one_state_model(VOCABULARY, np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,))))

    init = rand_p_vector(1, START_NOISE_WIDTH, rng)
    transition = rand_p_vector(1, START_NOISE_WIDTH, rng)
    fibre = rand_p_vector(N_USER, START_NOISE_WIDTH, rng)
    assert params.init_state_p.tobytes() == init.tobytes()
    assert params.transition_p.tobytes() == transition.reshape(1, 1).tobytes()
    assert params.output_p[0, 0, USER_BASE:].tobytes() == fibre.tobytes()
    assert not params.output_p[0, 0, :USER_BASE].any()
    assert params.vocabulary is VOCABULARY
    # The one-element vectors still draw: skipping them would shift the fibre.
    skipped = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,)))
    assert rand_p_vector(N_USER, START_NOISE_WIDTH, skipped).tobytes() != fibre.tobytes()


def test_a_search_is_reproducible_from_its_seed():
    records = _corpus()
    runs = [
        _search(
            records, start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=2, patience=2,
            backend="cython", score_backend="cython",
        )
        for _ in range(2)
    ]
    for a, b in zip(runs[0].rounds, runs[1].rounds):
        assert (a.move.kind, a.move.states, a.move.trial, a.move.total_bits) == (
            b.move.kind, b.move.states, b.move.trial, b.move.total_bits,
        )
        assert a.move.result.params.transition_p.tobytes() == b.move.result.params.transition_p.tobytes()
    assert runs[0].best.total_bits == runs[1].best.total_bits


# ---------------------------------------------------------------------------
# The start.


def test_the_one_state_start_converges_to_the_same_bits_from_any_draw():
    """With one state the E-step counts do not depend on the parameters (ADR 0023)."""
    records = _corpus()
    starts = [
        _search(records, start=VOCABULARY, seed=np.random.SeedSequence(entropy), max_rounds=0, patience=0).start
        for entropy in (1, 2, 3)
    ]
    first = starts[0].params
    assert not np.array_equal(
        _one_state_model(VOCABULARY, np.random.default_rng(1)).output_p,
        _one_state_model(VOCABULARY, np.random.default_rng(2)).output_p,
    )
    for other in starts[1:]:
        assert other.params.output_p.tobytes() == first.output_p.tobytes()
        assert other.total_bits == starts[0].total_bits


def test_a_given_start_is_converged_under_the_default_rule_and_scored_by_the_scan():
    rng = np.random.default_rng(SEED)
    output = np.zeros((2, 2, USER_BASE + N_USER))
    output[:, :, USER_BASE:] = rng.dirichlet(np.ones(N_USER), size=(2, 2))
    params = HMMParams(rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(2), size=2), output, VOCABULARY)
    records = _corpus()

    result = _search(records, start=params, seed=np.random.SeedSequence(SEED), max_rounds=0, patience=0)
    expected = baum_welch(params, records)
    assert result.start.params.output_p.tobytes() == expected.params.output_p.tobytes()
    assert result.start.cycles == expected.cycles
    assert result.start.data_bits == expected.description_lengths[-1]
    assert (result.start.d, result.start.total_bits) == _scan_d(
        expected.params, records, expected.description_lengths[-1]
    )


@pytest.mark.parametrize(
    ("keywords", "error", "match"),
    [
        (dict(start="a vocabulary"), TypeError, "start"),
        (dict(seed=7), TypeError, "SeedSequence"),
        (dict(max_rounds=2.0), TypeError, "max_rounds"),
        (dict(patience=True), TypeError, "patience"),
        (dict(max_rounds=-1), ValueError, "max_rounds"),
        (dict(patience=-1), ValueError, "patience"),
    ],
)
def test_the_search_rejects_bad_arguments_before_any_work(keywords, error, match):
    arguments = dict(start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=1, patience=0)
    arguments.update(keywords)
    with pytest.raises(error, match=match):
        _search(_corpus(), **arguments)


def test_max_rounds_and_patience_are_required():
    with pytest.raises(TypeError):
        _search(_corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), patience=0)
    with pytest.raises(TypeError):
        _search(_corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=0)


# ---------------------------------------------------------------------------
# One real search.


def test_the_first_rounds_choose_the_moves_the_lush_training_log_records():
    """The only oracle for a trajectory, and an oracle for moves rather than bits.

    `m001_0005_005`'s log (`_training_log`) shows the original growing the one-state
    model by splitting state 0, then 0, then 2, to totals printed at `%g` as 2433.99,
    2131.1 and 1774.03. ADR 0022 changed the split's draws and ADR 0023 the choice of
    `d`, so the candidates' bits cannot match; the winning split in each round beats its
    runners-up by more than those changes move a total, so the moves do, and the totals
    land within a bit of the log.
    """
    records = [load_corpus_record()]
    vocabulary = load_params(FIXTURES / "m001_0001_001.hmm").vocabulary
    result = _search(
        records, start=vocabulary, seed=np.random.SeedSequence(SEED), max_rounds=3, patience=0,
        backend="cython", score_backend="cython",
    )
    assert [(r.move.kind, r.move.states) for r in result.rounds] == [
        ("split", (0,)), ("split", (0,)), ("split", (2,)),
    ]
    logged = [2433.99, 2131.1, 1774.03]
    for round_, total in zip(result.rounds, logged):
        assert abs(round_.move.total_bits - total) < 1.0
        assert round_.improved
    assert result.start.d == 13.0
    assert result.best is result.rounds[-1].move.result
    assert result.best.params.n_states == 4
