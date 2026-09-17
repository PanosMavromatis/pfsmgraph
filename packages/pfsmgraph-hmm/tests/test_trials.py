"""Tests for ``_trials._try_split``, the scored split trial.

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
from pfsmgraph.hmm._topology import _split_state
from pfsmgraph.hmm._trials import TrialResult, _try_split

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
