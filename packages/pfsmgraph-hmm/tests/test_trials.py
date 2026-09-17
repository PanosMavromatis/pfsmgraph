"""Tests for ``_trials``: the scored trials and the ``suggest-*`` rankings.

The trial is a composition -- ADR 0022's split, ``baum_welch`` under a minimum
budget, ``_suggest_d``, ``_total_description_length`` -- each tested in its own
module, so these tests pin the composition rather than re-testing the parts: that
it is the composition bit for bit, in that order, with the floor and the backend
passed through; that the score is *called*, not assembled; and that the unrounded
data description length ADR 0022 section 5 asks for is the unrounded one.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, baum_welch
from pfsmgraph.hmm import _trials
from pfsmgraph.hmm._mdl import (
    _corpus_description_length,
    _data_description_length,
    _suggest_d,
    _total_description_length,
)
from pfsmgraph.hmm._topology import _merge_states, _split_state
from pfsmgraph.hmm._trials import (
    Move,
    TrialResult,
    _ranked,
    _suggest_merge,
    _suggest_move,
    _suggest_split,
    _try_merge,
    _try_split,
)

SEED = 20260917
# Above the 180 cycles this case converges in without a floor, so the floor binds.
MIN_CYCLES = 205
STATE = 1


def _random_params(rng, size, n_user):
    output = np.zeros((size, size, USER_BASE + n_user))
    output[:, :, USER_BASE:] = rng.dirichlet(np.ones(n_user), size=(size, size))
    return HMMParams(
        rng.dirichlet(np.ones(size)),
        rng.dirichlet(np.ones(size), size=size),
        output,
        SymbolTable([f"s{i}" for i in range(n_user)]),
    )


def _random_corpus(rng, n_user, lengths):
    return [
        SequenceRecord(rng.integers(USER_BASE, USER_BASE + n_user, length))
        for length in lengths
    ]


def _arrays(params):
    return params.init_state_p, params.transition_p, params.output_p


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(SEED)
    params = _random_params(rng, 2, 3)
    records = _random_corpus(rng, 3, (60, 40))
    before = tuple(array.tobytes() for array in _arrays(params))
    trial = _try_split(
        params, records, STATE, rng=np.random.default_rng(SEED + 1), min_cycles=MIN_CYCLES
    )
    return params, records, before, trial


def test_the_trial_is_split_then_baum_welch_under_the_floor_bit_for_bit(case):
    params, records, _, trial = case
    candidate = _split_state(params, STATE, rng=np.random.default_rng(SEED + 1))
    expected = baum_welch(candidate, records, min_cycles=MIN_CYCLES)

    for got, want in zip(_arrays(trial.params), _arrays(expected.params)):
        assert got.tobytes() == want.tobytes()
    assert trial.cycles == expected.cycles and trial.converged == expected.converged


def test_the_score_is_the_total_at_the_chosen_d(case):
    _, records, _, trial = case
    assert isinstance(trial, TrialResult)
    assert trial.d == _suggest_d(trial.params, records)
    assert trial.total_bits == _total_description_length(trial.params, records, trial.d)
    assert type(trial.total_bits) is float


def test_the_score_is_called_rather_than_assembled(case, monkeypatch):
    params, records, _, _ = case
    monkeypatch.setattr(_trials, "_total_description_length", lambda *_: 1234.5)
    trial = _try_split(
        params, records, STATE, rng=np.random.default_rng(SEED + 1), min_cycles=MIN_CYCLES
    )
    assert trial.total_bits == 1234.5


def test_data_bits_is_the_unrounded_corpus_length(case):
    # ADR 0022 section 5: the value d does not touch. The rounded data half at the
    # chosen d is a different number, so data_bits is not quietly that one.
    _, records, _, trial = case
    unrounded = _corpus_description_length(*_arrays(trial.params), records)
    assert trial.data_bits == unrounded
    assert trial.data_bits != _data_description_length(trial.params, records, trial.d)


def test_the_floor_binds_holds_and_the_run_converges(case):
    params, records, _, trial = case
    candidate = _split_state(params, STATE, rng=np.random.default_rng(SEED + 1))
    assert baum_welch(candidate, records).cycles < MIN_CYCLES
    assert trial.cycles >= MIN_CYCLES and trial.converged


def test_the_candidate_has_one_more_state_and_the_incumbent_is_untouched(case):
    params, _, before, trial = case
    assert trial.params.n_states == params.n_states + 1
    assert tuple(array.tobytes() for array in _arrays(params)) == before


@pytest.mark.parametrize("missing", ["rng", "min_cycles"])
def test_the_generator_and_the_floor_have_no_default(case, missing):
    params, records, _, _ = case
    keywords = {"rng": np.random.default_rng(SEED), "min_cycles": MIN_CYCLES}
    del keywords[missing]
    with pytest.raises(TypeError, match=missing):
        _try_split(params, records, STATE, **keywords)


def test_backend_batch_size_and_floor_reach_baum_welch(case, monkeypatch):
    params, records, _, _ = case
    seen = {}

    def spy(candidate, records, **keywords):
        seen.update(keywords)
        return baum_welch(candidate, records, min_cycles=keywords["min_cycles"])

    monkeypatch.setattr(_trials, "baum_welch", spy)
    _try_split(
        params,
        records,
        STATE,
        rng=np.random.default_rng(SEED),
        min_cycles=7,
        backend="python",
        batch_size=1,
    )
    assert seen == {"backend": "python", "batch_size": 1, "min_cycles": 7}


# --- try-merge -------------------------------------------------------------------

PAIR = (0, 2)


@pytest.fixture(scope="module")
def merge_case():
    rng = np.random.default_rng(SEED + 2)
    params = _random_params(rng, 3, 3)
    records = _random_corpus(rng, 3, (60, 40))
    before = tuple(array.tobytes() for array in _arrays(params))
    return params, records, before, _try_merge(params, records, *PAIR)


def test_the_merge_trial_is_merge_then_baum_welch_bit_for_bit(merge_case):
    params, records, _, trial = merge_case
    expected = baum_welch(_merge_states(params, *PAIR), records)

    for got, want in zip(_arrays(trial.params), _arrays(expected.params)):
        assert got.tobytes() == want.tobytes()
    assert trial.cycles == expected.cycles and trial.converged == expected.converged


def test_the_merge_score_is_the_total_at_the_chosen_d_and_is_called(merge_case, monkeypatch):
    params, records, _, trial = merge_case
    assert trial.d == _suggest_d(trial.params, records)
    assert trial.total_bits == _total_description_length(trial.params, records, trial.d)

    monkeypatch.setattr(_trials, "_total_description_length", lambda *_: 1234.5)
    assert _try_merge(params, records, *PAIR).total_bits == 1234.5


def test_the_merge_data_bits_is_the_unrounded_corpus_length(merge_case):
    _, records, _, trial = merge_case
    assert trial.data_bits == _corpus_description_length(*_arrays(trial.params), records)
    assert trial.data_bits != _data_description_length(trial.params, records, trial.d)


def test_the_merge_candidate_has_one_fewer_state_and_the_incumbent_is_untouched(merge_case):
    params, _, before, trial = merge_case
    assert trial.params.n_states == params.n_states - 1
    assert tuple(array.tobytes() for array in _arrays(params)) == before


def test_the_order_of_the_pair_does_not_matter(merge_case):
    # So ranking can enumerate unordered pairs, as suggest-merge does.
    params, records, _, trial = merge_case
    swapped = _try_merge(params, records, *reversed(PAIR))
    for got, want in zip(_arrays(swapped.params), _arrays(trial.params)):
        assert got.tobytes() == want.tobytes()
    assert (swapped.d, swapped.total_bits, swapped.data_bits) == (
        trial.d,
        trial.total_bits,
        trial.data_bits,
    )


def test_a_pair_of_two_transient_states_is_refused(merge_case):
    # States 0 and 1 drain into the absorbing state 2, so both carry no
    # stationary mass; suggest-merge filters such a pair before trying it.
    _, records, _, _ = merge_case
    output = np.zeros((3, 3, USER_BASE + 3))
    output[:, :, USER_BASE:] = 1 / 3
    params = HMMParams(
        [1.0, 0.0, 0.0],
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        output,
        SymbolTable(["s0", "s1", "s2"]),
    )
    with pytest.raises(ValueError, match="both are transient"):
        _try_merge(params, records, 0, 1)


def test_the_merge_floor_defaults_to_the_original_rule_and_is_passed_through(
    merge_case, monkeypatch
):
    params, records, _, _ = merge_case
    seen = []

    def spy(candidate, records, **keywords):
        seen.append(keywords)
        return baum_welch(candidate, records, min_cycles=keywords["min_cycles"])

    monkeypatch.setattr(_trials, "baum_welch", spy)
    _try_merge(params, records, *PAIR)
    _try_merge(params, records, *PAIR, min_cycles=7, backend="python", batch_size=1)
    assert seen == [
        {"backend": "python", "batch_size": None, "min_cycles": 0},
        {"backend": "python", "batch_size": 1, "min_cycles": 7},
    ]


# --- suggest-*: the ranking contract ----------------------------------------------


def _move(kind, states, total, trial=0):
    return Move(kind, states, trial, None, total)


def test_ranking_is_a_stable_sort_on_the_total():
    # Ties keep enumeration order, so a merge enumerated first beats a split with the
    # same total, and two impossible candidates tie without anything being subtracted.
    moves = [
        _move("merge", (0, 1), 5.0),
        _move("merge", (0, 2), np.inf),
        _move("merge", (1, 2), 3.0),
        _move("split", (0,), 5.0),
        _move("split", (1,), np.inf),
        _move("split", (1,), 3.0, trial=1),
    ]
    with np.errstate(all="raise"):
        ranked = _ranked(moves)
    assert ranked == [moves[2], moves[5], moves[0], moves[3], moves[1], moves[4]]


def test_an_all_impossible_round_is_an_ordinary_ranking():
    # The original's best-* slots stayed () and idx-copy failed.
    moves = [_move("merge", (0, 1), np.inf), _move("split", (0,), np.inf)]
    assert _ranked(moves) == moves


def _fake(total):
    return TrialResult(params=None, d=1.0, total_bits=total, data_bits=total, cycles=0, converged=True)


def _spy_split(monkeypatch, totals=None):
    calls = []

    def spy(params, records, state, *, rng, min_cycles, backend, batch_size):
        draw = rng.random()
        calls.append((state, draw, min_cycles, backend, batch_size))
        return _fake(totals(state, len(calls)) if totals else draw)

    monkeypatch.setattr(_trials, "_try_split", spy)
    return calls


def _spy_merge(monkeypatch, totals=None):
    calls = []

    def spy(params, records, first, second, *, min_cycles, backend, batch_size):
        calls.append(((first, second), min_cycles, backend, batch_size))
        return _fake(totals(first, second) if totals else float(first + second))

    monkeypatch.setattr(_trials, "_try_merge", spy)
    return calls


def test_suggest_split_runs_each_state_in_order_with_the_shipped_trial_counts(case, monkeypatch):
    params, records, _, _ = case
    calls = _spy_split(monkeypatch)
    seed = np.random.SeedSequence(7)
    moves = _suggest_split(params, records, seed=seed, min_cycles=11, batch_size=2)

    assert [c[0] for c in calls] == [0, 0, 1, 1]
    assert {c[2:] for c in calls} == {(11, "python", 2)}
    assert sorted((m.states, m.trial) for m in moves) == [((0,), 0), ((0,), 1), ((1,), 0), ((1,), 1)]
    assert [m.total_bits for m in moves] == sorted(c[1] for c in calls)


def test_split_trials_scale_with_state_p(case, monkeypatch):
    params, records, _, _ = case
    calls = _spy_split(monkeypatch)
    _suggest_split(
        params, records, seed=np.random.SeedSequence(7), min_cycles=0,
        min_split_trials=1, split_trials_per_state_p=10.0,
    )
    expected = [1 + int(10.0 * p) for p in params.state_p]
    assert [sum(c[0] == s for c in calls) for s in range(params.n_states)] == expected


def test_each_split_trial_draws_from_its_own_identity_keyed_generator(case, monkeypatch):
    # Trial (s, t) must not depend on how many trials ran before it: the same round
    # with more trials per state gives trial (s, 0) the same draw.
    params, records, _, _ = case
    seed = np.random.SeedSequence(7, spawn_key=(3,))
    few = _spy_split(monkeypatch)
    _suggest_split(params, records, seed=seed, min_cycles=0, min_split_trials=1)
    many = _spy_split(monkeypatch)
    _suggest_split(params, records, seed=seed, min_cycles=0, min_split_trials=3)

    for state in range(params.n_states):
        for trial in range(3):
            key = np.random.SeedSequence(7, spawn_key=(3, state, trial))
            assert many[state * 3 + trial][1] == np.random.default_rng(key).random()
        assert few[state][1] == many[state * 3][1]
    assert len({c[1] for c in many}) == len(many)


def test_suggest_merge_tries_pairs_in_order_and_leaves_out_two_transient_states(monkeypatch):
    # States 0 and 1 drain into the closed class {2, 3}; (0, 1) is not a candidate.
    output = np.zeros((4, 4, USER_BASE + 2))
    output[:, :, USER_BASE:] = 0.5
    params = HMMParams(
        [1.0, 0.0, 0.0, 0.0],
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]],
        output,
        SymbolTable(["a", "b"]),
    )
    calls = _spy_merge(monkeypatch)
    moves = _suggest_merge(params, [], min_cycles=4)

    assert [c[0] for c in calls] == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert {c[1:] for c in calls} == {(4, "python", None)}
    assert [m.states for m in moves] == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_suggest_move_ranks_merges_before_splits_on_a_tie(case, monkeypatch):
    params, records, _, _ = case
    _spy_merge(monkeypatch, totals=lambda a, b: 10.0)
    _spy_split(monkeypatch, totals=lambda state, n: 10.0 if state == 0 else 9.0)
    moves = _suggest_move(params, records, seed=np.random.SeedSequence(1), split_min_cycles=0)

    assert [(m.kind, m.states, m.trial) for m in moves] == [
        ("split", (1,), 0),
        ("split", (1,), 1),
        ("merge", (0, 1), 0),
        ("split", (0,), 0),
        ("split", (0,), 1),
    ]


def test_a_one_state_move_round_has_no_merges(monkeypatch):
    rng = np.random.default_rng(SEED + 5)
    params = _random_params(rng, 1, 3)
    merges = _spy_merge(monkeypatch)
    _spy_split(monkeypatch)
    moves = _suggest_move(params, [], seed=np.random.SeedSequence(1), split_min_cycles=0)
    assert merges == [] and [m.kind for m in moves] == ["split", "split"]


@pytest.mark.parametrize(
    "keywords, error",
    [
        ({"seed": 7}, TypeError),
        ({"seed": np.random.SeedSequence(7), "min_split_trials": -1}, ValueError),
        ({"seed": np.random.SeedSequence(7), "min_split_trials": 2.0}, TypeError),
        ({"seed": np.random.SeedSequence(7), "split_trials_per_state_p": -1.0}, ValueError),
        ({"seed": np.random.SeedSequence(7), "split_trials_per_state_p": np.nan}, ValueError),
    ],
)
def test_a_malformed_round_is_refused_before_any_trial(case, monkeypatch, keywords, error):
    params, records, _, _ = case
    calls = _spy_split(monkeypatch)
    with pytest.raises(error):
        _suggest_split(params, records, min_cycles=0, **keywords)
    assert calls == []


# --- suggest-*: end to end ---------------------------------------------------------


def _transient_start():
    """State 0 is a transient start whose only arc, into 1, is the only one emitting `a`.

    Merging 0 with anything gives it stationary weight 0, so the `a` arc dies and a
    record starting with `a` becomes impossible; merging 1 and 2 keeps every path.
    """
    output = np.zeros((3, 3, USER_BASE + 2))
    output[0, 1, USER_BASE:] = [1.0, 0.0]
    output[1:, 1:, USER_BASE:] = [0.0, 1.0]
    params = HMMParams(
        [1.0, 0.0, 0.0],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]],
        output,
        SymbolTable(["a", "b"]),
    )
    a, b = USER_BASE, USER_BASE + 1
    return params, [SequenceRecord(np.array([a, b, b, b])), SequenceRecord(np.array([a, b]))]


def test_an_impossible_merge_is_ranked_last_with_no_result():
    params, records = _transient_start()
    moves = _suggest_merge(params, records)

    assert [m.states for m in moves] == [(1, 2), (0, 1), (0, 2)]
    assert moves[0].result is not None and np.isfinite(moves[0].total_bits)
    assert all(m.result is None and m.total_bits == np.inf for m in moves[1:])


def test_a_ranked_merge_is_the_trial_itself_bit_for_bit(merge_case):
    params, records, _, _ = merge_case
    moves = _suggest_merge(params, records)

    assert sorted(m.states for m in moves) == [(0, 1), (0, 2), (1, 2)]
    totals = [m.total_bits for m in moves]
    assert totals == sorted(totals)
    pair = next(m for m in moves if m.states == PAIR)
    assert pair.total_bits == _try_merge(params, records, *PAIR).total_bits
    assert pair.result.total_bits == pair.total_bits


def test_a_ranked_split_is_rebuilt_from_its_identity_alone(case):
    params, records, _, _ = case
    seed = np.random.SeedSequence(SEED, spawn_key=(0,))
    moves = _suggest_split(params, records, seed=seed, min_cycles=0, min_split_trials=1)
    move = next(m for m in moves if m.states == (STATE,))

    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0, STATE, 0)))
    alone = _try_split(params, records, STATE, rng=rng, min_cycles=0)
    for got, want in zip(_arrays(move.result.params), _arrays(alone.params)):
        assert got.tobytes() == want.tobytes()
    assert move.total_bits == alone.total_bits
