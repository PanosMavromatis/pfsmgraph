"""The state split, against the invariants ADR 0022 makes contract.

**Every test here reaches a private function**, labelled rather than hidden as
ADR 0003 asks: ``_topology`` is private to ``pfsmgraph.hmm`` and nothing in it is
exported, so there is no public call to reach it through.

There is no oracle in the original to compare against, and deliberately so: the
split departs from the Lush ``split-state`` on both of the points ADR 0022
records, so a differential test would pin the defect and the redraw this
replaces. The tests are properties instead, each exact where the arithmetic
allows it, on random models with dead arcs and on the three tracked fixtures.

Sections:

- **What each array becomes**: initial probabilities, transitions and fibres,
  entry by entry, against the incumbent's.
- **What the result must satisfy as a model**: stochastic rows and live fibres,
  zero reserved fibres, and the same vocabulary.
- **The likelihood**: unchanged without the seed, and moved by at most the bound
  the seed's weight implies with it.
- **The domain**: what ``state`` may be.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams
from pfsmgraph.hmm import _topology
from pfsmgraph.hmm._mdl import _corpus_description_length
from pfsmgraph.hmm._topology import _INBOUND_WIDTH, _SEED_NOISE, _SEED_WEIGHT, _split_state

from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params

SEED = 20260916

#: A row or fibre built by one rounded mixture of two unit-sum vectors, or by a
#: stationary solve: well under an ulp per element over a few dozen elements.
SUM_TOL = 1e-12


def _vocabulary(n_user):
    return SymbolTable([f"s{i}" for i in range(n_user)])


def _random_params(rng, size, n_user, dead_fraction, zero_init_fraction):
    """A random model with some arcs exactly dead (diagonal kept live) and some
    initial probabilities exactly zero, so the split meets both kinds of zero."""
    transition = rng.dirichlet(np.ones(size), size=size)
    dead = rng.random((size, size)) < dead_fraction
    dead[np.arange(size), np.arange(size)] = False
    transition = np.where(dead, 0.0, transition)
    transition /= transition.sum(axis=1, keepdims=True)
    init = rng.dirichlet(np.ones(size))
    if size > 1:
        zero = rng.random(size) < zero_init_fraction
        zero[int(np.argmax(init))] = False
        init = np.where(zero, 0.0, init)
        init /= init.sum()
    output = np.zeros((size, size, USER_BASE + n_user))
    output[..., USER_BASE:] = rng.dirichlet(np.full(n_user, 0.5), size=(size, size))
    return HMMParams(init, transition, output, _vocabulary(n_user))


def _random_records(rng, n_user, lengths):
    return [SequenceRecord(rng.integers(USER_BASE, USER_BASE + n_user, n)) for n in lengths]


def _cases():
    rng = np.random.default_rng(SEED)
    cases = []
    for size, n_user, dead, zero_init in ((1, 3, 0.0, 0.0), (3, 4, 0.3, 0.3), (8, 6, 0.5, 0.5)):
        params = _random_params(rng, size, n_user, dead, zero_init)
        records = _random_records(rng, n_user, (40, 0, 25))
        cases.append(pytest.param((params, records), id=f"random-S{size}-A{n_user}"))
    record = load_corpus_record()
    for name in SAVED_MODELS:
        stem = name.removesuffix(".hmm")
        cases.append(pytest.param((load_params(FIXTURES / name), [record]), id=stem))
    return cases


def _splits():
    """Every state of every case, each split with its own generator."""
    out = []
    for case in _cases():
        params, records = case.values[0]
        for state in range(params.n_states):
            out.append(pytest.param((params, records, state), id=f"{case.id}-s{state}"))
    return out


@pytest.fixture(scope="module", params=_splits())
def split(request):
    params, records, state = request.param
    rng = np.random.default_rng([SEED, params.n_states, state])
    return params, records, state, _split_state(params, state, rng=rng)


# --- what each array becomes ------------------------------------------------------


def test_the_result_has_one_more_state_and_the_same_vocabulary(split):
    params, _, _, result = split
    assert result.n_states == params.n_states + 1
    assert result.n_symbols == params.n_symbols
    assert result.vocabulary is params.vocabulary


def test_uninvolved_states_keep_their_learned_initial_probability(split):
    """ADR 0022 section 1: the original's `:172` read `state_p` here instead."""
    params, _, s, result = split
    others = [i for i in range(params.n_states) if i != s]
    assert np.array_equal(result.init_state_p[others], params.init_state_p[others])


def test_the_twins_share_the_split_state_initial_probability_in_halves(split):
    params, _, s, result = split
    half = 0.5 * params.init_state_p[s]
    assert result.init_state_p[s] == half
    assert result.init_state_p[params.n_states] == half


def test_every_arc_into_the_split_state_is_divided_exactly_between_the_twins(split):
    """The remainder form is exact by Sterbenz's lemma, so this is `==`, not a tolerance."""
    params, _, s, result = split
    size = params.n_states
    old = params.transition_p[:, s]
    assert np.array_equal(result.transition_p[:size, s] + result.transition_p[:size, size], old)


def test_each_inbound_share_lies_within_the_perturbation_width(split):
    params, _, s, result = split
    size = params.n_states
    old = params.transition_p[:, s]
    live = old > 0
    fraction = result.transition_p[:size, s][live] / old[live]
    assert np.all(fraction >= 0.5 - _INBOUND_WIDTH / 2 - 1e-15)
    assert np.all(fraction <= 0.5 + _INBOUND_WIDTH / 2 + 1e-15)


def test_a_dead_arc_into_the_split_state_stays_dead_for_both_twins(split):
    params, _, s, result = split
    size = params.n_states
    dead = params.transition_p[:, s] == 0
    assert np.all(result.transition_p[:size, s][dead] == 0)
    assert np.all(result.transition_p[:size, size][dead] == 0)


def test_transitions_that_touch_neither_twin_are_unchanged(split):
    params, _, s, result = split
    others = [j for j in range(params.n_states) if j != s]
    assert np.array_equal(
        result.transition_p[np.ix_(others, others)], params.transition_p[np.ix_(others, others)]
    )


def test_the_twin_copies_the_split_state_outbound_row(split):
    """Taken after the inbound split, so the self-loop's division carries over."""
    params, _, s, result = split
    assert np.array_equal(result.transition_p[params.n_states], result.transition_p[s])


def test_fibres_into_the_twin_copy_the_fibres_into_the_split_state(split):
    params, _, s, result = split
    size = params.n_states
    others = [j for j in range(size) if j != s]
    assert np.array_equal(result.output_p[others, size], params.output_p[others, s])


def test_fibres_that_touch_neither_twin_are_unchanged(split):
    params, _, s, result = split
    others = [j for j in range(params.n_states) if j != s]
    assert np.array_equal(
        result.output_p[np.ix_(others, others)], params.output_p[np.ix_(others, others)]
    )


def test_a_fibre_leaving_a_twin_on_a_dead_arc_is_copied_unseeded(split):
    params, _, s, result = split
    size = params.n_states
    learned = np.concatenate([params.output_p[s], params.output_p[s, s][None]], axis=0)
    dead = result.transition_p[s] == 0
    for twin in (s, size):
        assert np.array_equal(result.output_p[twin, dead], learned[dead])


def test_a_fibre_leaving_a_twin_on_a_live_arc_is_one_percent_seeded(split):
    """Undo the mixture and what is left must be a `rand_p_vector` draw: a unit-sum
    vector whose elements lie within its noise width of uniform."""
    params, _, s, result = split
    size = params.n_states
    n_user = params.n_symbols - USER_BASE
    learned = np.concatenate([params.output_p[s], params.output_p[s, s][None]], axis=0)
    live = result.transition_p[s] > 0
    lowest = (1 - _SEED_NOISE) / ((1 + _SEED_NOISE) * n_user)
    highest = (1 + _SEED_NOISE) / ((1 - _SEED_NOISE) * n_user)
    for twin in (s, size):
        drawn = (
            result.output_p[twin, live, USER_BASE:] - (1 - _SEED_WEIGHT) * learned[live, USER_BASE:]
        ) / _SEED_WEIGHT
        assert np.allclose(drawn.sum(axis=-1), 1.0, rtol=0, atol=1e-9)
        assert np.all(drawn >= lowest - 1e-9)
        assert np.all(drawn <= highest + 1e-9)


def test_the_twins_draw_different_seeds(split):
    """Otherwise the seed would break no symmetry at all."""
    params, _, s, result = split
    live = result.transition_p[s] > 0
    assert not np.array_equal(result.output_p[s, live], result.output_p[params.n_states, live])


# --- what the result must satisfy as a model --------------------------------------


def test_the_initial_distribution_sums_to_one(split):
    _, _, _, result = split
    assert abs(result.init_state_p.sum() - 1.0) <= SUM_TOL


def test_every_transition_row_sums_to_one(split):
    """Each predecessor row holds its sum exactly, by the division tested above; the
    twins' rows are the split state's row, divided the same way."""
    _, _, _, result = split
    assert np.all(np.abs(result.transition_p.sum(axis=1) - 1.0) <= SUM_TOL)


def test_every_live_fibre_sums_to_one(split):
    _, _, _, result = split
    live = result.transition_p > 0
    assert np.all(np.abs(result.output_p[live].sum(axis=-1) - 1.0) <= SUM_TOL)


def test_the_reserved_fibres_stay_exactly_zero(split):
    _, _, _, result = split
    assert np.all(result.output_p[..., :USER_BASE] == 0.0)


def test_the_stationary_distribution_gives_the_twins_the_split_state_share(split):
    """Before the seed the twins together occupy the chain exactly as `s` did; the
    transitions carry no seed, so this holds of the result as returned."""
    params, _, s, result = split
    size = params.n_states
    together = result.state_p[s] + result.state_p[size]
    assert math.isclose(together, params.state_p[s], rel_tol=1e-9, abs_tol=1e-12)


# --- the likelihood ---------------------------------------------------------------


def _bits(params, records):
    return _corpus_description_length(
        params.init_state_p, params.transition_p, params.output_p, records
    )


def test_without_the_seed_the_split_is_an_exact_reparameterisation(split, monkeypatch):
    """Every sequence keeps the probability the incumbent gave it (ADR 0022,
    Consequences). Differences are summation rounding over more states."""
    params, records, s, _ = split
    monkeypatch.setattr(_topology, "_SEED_WEIGHT", 0.0)
    unseeded = _split_state(params, s, rng=np.random.default_rng(SEED))
    before, after = _bits(params, records), _bits(unseeded, records)
    assert math.isclose(after, before, rel_tol=1e-12, abs_tol=1e-9)


def test_with_the_seed_no_record_costs_more_than_the_seed_weight_allows(split):
    """A seeded fibre is at least `(1 - weight)` times the learned one, so no path
    loses more than that factor per symbol, and neither does the sum over paths."""
    params, records, _, result = split
    n_symbols = sum(record.codes.size for record in records)
    bound = n_symbols * -math.log2(1 - _SEED_WEIGHT)
    assert _bits(result, records) <= _bits(params, records) + bound + 1e-9


# --- the domain -------------------------------------------------------------------


@pytest.fixture(scope="module")
def three_states():
    return _random_params(np.random.default_rng(SEED), 3, 4, 0.0, 0.0)


@pytest.mark.parametrize("state", [-1, 3, 10])
def test_a_state_outside_the_model_raises_value_error(three_states, state):
    with pytest.raises(ValueError, match=r"state must lie in \[0, 3\)"):
        _split_state(three_states, state, rng=np.random.default_rng(SEED))


@pytest.mark.parametrize("state", [1.0, True, "1", None])
def test_a_state_that_is_not_an_integer_raises_type_error(three_states, state):
    """`bool` is refused although it subclasses `int`: `True` naming state 1 is a bug."""
    with pytest.raises(TypeError, match="state must be an integer"):
        _split_state(three_states, state, rng=np.random.default_rng(SEED))


def test_a_numpy_integer_state_is_accepted(three_states):
    result = _split_state(three_states, np.int64(2), rng=np.random.default_rng(SEED))
    assert result.n_states == 4
