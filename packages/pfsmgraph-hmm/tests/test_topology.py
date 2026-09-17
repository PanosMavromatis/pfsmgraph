"""The state split and the state merge, against the invariants their decisions make contract.

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
- **The decode's zero initial probability**: the case revision 02's fixtures could
  not exhibit, a state with ``init_p == 0`` that can emit ``begin``, built by a
  split and decoded.
- **Reproducibility**: the same generator state gives a bit-identical candidate,
  and the draws follow ADR 0022's contract in count and in order.
- **The domain**: what ``state`` may be.
- **The merge**: each array against the incumbent's, exactly where the
  arithmetic allows; the result as a model; the merge as an exact *lumping* of
  the stationary chain; its domain; and a split-then-merge round trip.

The merge has an oracle in the original only up to the three places it departs
from ``merge-states`` (the branch plan for ``feat/hmm-merge-states``, goals
2-4), so it too is tested by properties. The lumping is the one that is not a
restatement of the formulas: weighting a merged state's outbound arcs by
stationary mass makes the stationary flow on every arc and symbol add up
exactly, which an independent stationary solve of the result can check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, viterbi
from pfsmgraph.hmm import _topology
from pfsmgraph.hmm._mdl import _corpus_description_length
from pfsmgraph.hmm._numeric import closed_classes, rand_p_vector
from pfsmgraph.hmm._viterbi import _viterbi
from pfsmgraph.hmm._topology import (
    _INBOUND_WIDTH,
    _SEED_NOISE,
    _SEED_WEIGHT,
    _merge_states,
    _split_state,
)

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


# --- the decode's zero initial probability ---------------------------------------
#
# Revision 02 fixed the original's `update-viterbi-path`, which seeded the decode's
# first column with the raw initial probability instead of its bits, so a state
# with `init_p == 0` cost nothing to start in. Every zero-init state in the
# tracked models also cannot emit `begin`, which hid the defect, and a split was
# expected to be what breaks that link. It does, but only through ADR 0022's 1%
# seed, which is too weak to make the defect win on the fixtures. So the first
# test shows the link broken, and the second builds a model in which the defect
# would change the decode, so the fix is exercised rather than merely
# unchallenged.

#: The fixtures' `begin`: Lush code 0, renumbered onto the user block.
BEGIN = USER_BASE


def _defective_start(params, codes):
    """The start state the original's seeding would choose: pass `2**-init` to the
    kernel, which takes `bits` of it, so its first column is the raw `init`."""
    states, _ = _viterbi(2.0 ** -params.init_state_p, params.transition_p, params.output_p, codes)
    return int(states[0])


def _zero_init_splits_of_states_that_cannot_emit_begin():
    out = []
    record = load_corpus_record()
    for name in SAVED_MODELS:
        params = load_params(FIXTURES / name)
        live = params.transition_p > 0
        emits_begin = (live & (params.output_p[..., BEGIN] > 0)).any(axis=1)
        for s in range(params.n_states):
            if params.init_state_p[s] == 0 and not emits_begin[s]:
                out.append(pytest.param((params, record, s), id=f"{name.removesuffix('.hmm')}-s{s}"))
    return out


@pytest.mark.parametrize("case", _zero_init_splits_of_states_that_cannot_emit_begin())
def test_a_split_lets_zero_init_twins_emit_begin_and_the_decode_never_starts_there(case):
    params, record, s = case
    result = _split_state(params, s, rng=np.random.default_rng([SEED, s]))
    twins = (s, params.n_states)
    for twin in twins:
        assert result.init_state_p[twin] == 0.0
        live = result.transition_p[twin] > 0
        assert np.any(result.output_p[twin, live, BEGIN] > 0)
    assert record.codes[0] == BEGIN
    path = viterbi(result, record)
    assert math.isfinite(path.total_bits)
    assert int(path.states[0]) not in twins


def _begin_trap():
    """Two states over `begin`, `x`, `y`. State 0 starts every path and rarely
    emits `begin`; state 1 can never start one and emits `begin` with 0.9, so the
    original's free start makes state 1 the cheaper choice."""
    output = np.zeros((2, 2, USER_BASE + 3))
    output[0, :, USER_BASE:] = [0.1, 0.45, 0.45]
    output[1, :, USER_BASE:] = [0.9, 0.05, 0.05]
    return HMMParams(
        np.array([1.0, 0.0]), np.array([[0.5, 0.5], [0.5, 0.5]]), output, _vocabulary(3)
    )


@pytest.mark.parametrize("split_first", [False, True], ids=["incumbent", "after-split"])
def test_the_fixed_seeding_never_starts_where_the_original_would(split_first):
    params = _begin_trap()
    if split_first:
        params = _split_state(params, 1, rng=np.random.default_rng(SEED))
    record = SequenceRecord(np.array([BEGIN, BEGIN + 1, BEGIN + 2, BEGIN]))
    zero_init = set(np.flatnonzero(params.init_state_p == 0).tolist())
    assert _defective_start(params, record.codes) in zero_init
    path = viterbi(params, record)
    assert int(path.states[0]) not in zero_init
    assert math.isfinite(path.total_bits)


# --- reproducibility --------------------------------------------------------------
#
# ADR 0022, **Resolved**: the number of draws depends on `S` alone, and they come in
# a fixed order. The order test rebuilds the candidate from an independent generator
# in the contract's order, so drawing the seeds before the perturbations, the twins
# in the other order, or only on live arcs all fail it.


def _arrays(params):
    return params.init_state_p.tobytes(), params.transition_p.tobytes(), params.output_p.tobytes()


def test_the_same_generator_state_gives_a_bit_identical_candidate(split):
    params, _, s, _ = split
    first = _split_state(params, s, rng=np.random.default_rng([SEED, 1]))
    second = _split_state(params, s, rng=np.random.default_rng([SEED, 1]))
    assert _arrays(first) == _arrays(second)


def test_a_different_generator_state_gives_a_different_candidate(split):
    params, _, s, _ = split
    first = _split_state(params, s, rng=np.random.default_rng([SEED, 1]))
    second = _split_state(params, s, rng=np.random.default_rng([SEED, 2]))
    assert first.output_p.tobytes() != second.output_p.tobytes()


def test_a_split_consumes_exactly_the_contracted_number_of_uniforms(split):
    """`S + 2 * (S + 1) * (n_symbols - USER_BASE)`, whatever arcs are live: the next
    value from the split's generator is the value that many uniforms later."""
    params, _, s, _ = split
    size, n_user = params.n_states, params.n_symbols - USER_BASE
    used = np.random.default_rng([SEED, 3])
    _split_state(params, s, rng=used)
    skipped = np.random.default_rng([SEED, 3])
    skipped.uniform(size=size + 2 * (size + 1) * n_user)
    assert used.uniform() == skipped.uniform()


def test_the_draws_follow_the_contracted_order(split):
    """Rebuilt by hand from the same generator state: perturbations first, then twin
    `s`'s seed fibres over every destination, then twin `p`'s."""
    params, _, s, result = split
    size, n_user = params.n_states, params.n_symbols - USER_BASE
    rng = np.random.default_rng([SEED, params.n_states, s])
    u = rng.uniform(-0.5, 0.5, size=size)
    seeds = [[rand_p_vector(n_user, _SEED_NOISE, rng) for _ in range(size + 1)] for _ in range(2)]

    inbound = params.transition_p[:, s]
    expected_s = (0.5 + _INBOUND_WIDTH * u) * inbound
    assert np.array_equal(result.transition_p[:size, s], expected_s)
    assert np.array_equal(result.transition_p[:size, size], inbound - expected_s)

    learned = np.concatenate([params.output_p[s], params.output_p[s, s][None]], axis=0)
    live = result.transition_p[s] > 0
    for twin, t in enumerate((s, size)):
        for j in np.flatnonzero(live):
            expected = (1.0 - _SEED_WEIGHT) * learned[j, USER_BASE:] + _SEED_WEIGHT * seeds[twin][j]
            assert np.array_equal(result.output_p[t, j, USER_BASE:], expected)


def test_the_draw_count_does_not_depend_on_which_arcs_are_live():
    """Two models with the same state and symbol counts but different dead arcs
    leave their generators in the same state, so a later draw is not shifted."""
    rng = np.random.default_rng(SEED)
    dense = _random_params(rng, 5, 4, 0.0, 0.0)
    sparse = _random_params(rng, 5, 4, 0.7, 0.5)
    assert np.count_nonzero(dense.transition_p) != np.count_nonzero(sparse.transition_p)
    after_dense, after_sparse = np.random.default_rng(SEED), np.random.default_rng(SEED)
    _split_state(dense, 2, rng=after_dense)
    _split_state(sparse, 2, rng=after_sparse)
    assert after_dense.uniform() == after_sparse.uniform()


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


# --- the merge --------------------------------------------------------------------

#: Relative tolerance for quantities that pass through a stationary solve or a
#: fibre average, which divides a sum by its own total.
MERGE_RTOL = 1e-11
MERGE_ATOL = 1e-14


def _transient_params(rng, size, n_transient, n_user):
    """A model with one closed class over its last ``size - n_transient`` states and
    ``n_transient`` transient states that lead into it, with dead arcs, zero fibres on
    some dead arcs, and some zero initial probabilities."""
    core = np.arange(n_transient, size)
    transition = np.zeros((size, size))
    for i in core:
        row = rng.dirichlet(np.ones(core.size)) * (rng.random(core.size) < 0.6)
        row[(i - n_transient + 1) % core.size] += 0.2  # a cycle keeps the core closed
        transition[i, core] = row
    for i in range(n_transient):
        row = rng.dirichlet(np.ones(size)) * (rng.random(size) < 0.5)
        row[rng.choice(core)] += 0.2  # every transient state leaks into the core
        transition[i] = row
    transition /= transition.sum(axis=1, keepdims=True)
    labels = closed_classes(transition)
    assert labels.max() == 0 and np.flatnonzero(labels < 0).tolist() == list(range(n_transient))
    output = np.zeros((size, size, USER_BASE + n_user))
    output[..., USER_BASE:] = rng.dirichlet(np.full(n_user, 0.5), size=(size, size))
    output[(transition == 0) & (rng.random((size, size)) < 0.5)] = 0.0
    init = rng.dirichlet(np.ones(size)) * (rng.random(size) < 0.7)
    init[core[0]] += 0.1
    return HMMParams(init / init.sum(), transition, output, _vocabulary(n_user))


def _merge_incumbents():
    rng = np.random.default_rng([SEED, 1])
    cases = [
        pytest.param(_transient_params(rng, size, n_transient, n_user), id=f"random-S{size}-T{n_transient}")
        for size, n_transient, n_user in ((3, 1, 3), (5, 2, 4), (8, 3, 6))
    ]
    for name in SAVED_MODELS:
        params = load_params(FIXTURES / name)
        cases.append(pytest.param(params, id=name.removesuffix(".hmm")))
    return cases


def _merge_pairs(transients):
    """Every pair with at least one recurrent state, and with exactly
    ``transients`` transient ones if that is given."""
    out = []
    for case in _merge_incumbents():
        params = case.values[0]
        transient = closed_classes(params.transition_p) < 0
        for a in range(params.n_states):
            for b in range(a + 1, params.n_states):
                count = int(transient[a]) + int(transient[b])
                if count < 2 and transients in (None, count):
                    out.append(pytest.param((params, a, b), id=f"{case.id}-{a}+{b}"))
    return out


@pytest.fixture(scope="module", params=_merge_pairs(None))
def merge(request):
    params, a, b = request.param
    return params, a, b, _merge_states(params, a, b)


def _keep(params, b):
    return np.flatnonzero(np.arange(params.n_states) != b)


def _new_index(params, b):
    """Where each old state other than ``b`` lands; sending ``b`` to ``a`` is the caller's."""
    return np.arange(params.n_states) - (np.arange(params.n_states) > b)


# what each array becomes


def test_a_merge_has_one_state_fewer_and_the_same_vocabulary(merge):
    params, _, _, result = merge
    assert result.n_states == params.n_states - 1
    assert result.n_symbols == params.n_symbols
    assert result.vocabulary is params.vocabulary


def test_a_merge_does_not_depend_on_the_order_of_the_pair(merge):
    params, a, b, result = merge
    swapped = _merge_states(params, b, a)
    for name in ("init_state_p", "transition_p", "output_p"):
        assert np.array_equal(getattr(swapped, name), getattr(result, name))


def test_the_merged_state_sums_the_pair_initial_probability_and_the_rest_keep_theirs(merge):
    """ADR 0022 section 1: the original's `:262` read `state_p` for every other state."""
    params, a, b, result = merge
    keep = _keep(params, b)
    others = keep != a
    assert np.array_equal(result.init_state_p[others], params.init_state_p[keep][others])
    assert result.init_state_p[a] == params.init_state_p[a] + params.init_state_p[b]


def test_arcs_and_fibres_that_touch_neither_state_are_unchanged(merge):
    params, a, b, result = merge
    keep = _keep(params, b)
    others = np.flatnonzero(keep != a)
    old = keep[others]
    assert np.array_equal(result.transition_p[np.ix_(others, others)], params.transition_p[np.ix_(old, old)])
    assert np.array_equal(result.output_p[np.ix_(others, others)], params.output_p[np.ix_(old, old)])


def test_every_arc_into_the_merged_state_sums_the_pair_arcs(merge):
    params, a, b, result = merge
    keep = _keep(params, b)
    others = np.flatnonzero(keep != a)
    old = keep[others]
    expected = params.transition_p[old, a] + params.transition_p[old, b]
    assert np.array_equal(result.transition_p[others, a], expected)


@pytest.mark.parametrize("pair", _merge_pairs(1))
def test_a_transient_partner_contributes_nothing_to_the_merged_row(pair):
    """Its weight is exactly 0 and the other's exactly 1, so the row is the recurrent
    state's with the pair's columns collapsed, bit for bit. The fibres divide
    `T*O` by `T` and so agree only to rounding."""
    params, a, b = pair
    result = _merge_states(params, a, b)
    r = b if closed_classes(params.transition_p)[a] < 0 else a
    keep = _keep(params, b)
    expected = params.transition_p[r, keep].copy()
    expected[a] = params.transition_p[r, a] + params.transition_p[r, b]
    assert np.array_equal(result.transition_p[a], expected)
    others = np.flatnonzero(keep != a)
    live = result.transition_p[a, others] > 0
    assert np.allclose(
        result.output_p[a, others[live]], params.output_p[r, keep[others][live]], rtol=MERGE_RTOL, atol=0
    )


# the result as a model


def test_a_merge_leaves_every_distribution_normalised(merge):
    _, _, _, result = merge
    assert abs(result.init_state_p.sum() - 1.0) <= SUM_TOL
    assert np.all(np.abs(result.transition_p.sum(axis=1) - 1.0) <= SUM_TOL)
    live = result.transition_p > 0
    assert np.all(np.abs(result.output_p[live].sum(axis=-1) - 1.0) <= SUM_TOL)
    assert np.all(result.output_p[..., :USER_BASE] == 0.0)


def test_a_dead_arc_in_the_merge_has_an_all_zero_fibre(merge):
    params, a, b, result = merge
    dead = result.transition_p == 0
    touched = np.zeros_like(dead)
    touched[a, :] = touched[:, a] = True
    assert np.all(result.output_p[dead & touched] == 0.0)


# the merge as a lumping of the stationary chain


def test_the_merged_state_takes_the_pair_stationary_mass(merge):
    params, a, b, result = merge
    lumped = params.state_p[_keep(params, b)].copy()
    lumped[a] = params.state_p[a] + params.state_p[b]
    assert np.allclose(result.state_p, lumped, rtol=MERGE_RTOL, atol=MERGE_ATOL)


def test_the_stationary_flow_on_every_arc_and_symbol_adds_up(merge):
    """`pi[i] * T[i, j] * O[i, j, k]` summed over the old arcs each new arc stands for.

    This is what checks the three blocks together, including the self-loop's four
    arcs, without restating any of their formulas."""
    params, a, b, result = merge
    old_flow = params.state_p[:, None, None] * params.transition_p[..., None] * params.output_p
    new_index = _new_index(params, b)
    new_index[b] = a
    expected = np.zeros((result.n_states, result.n_states, result.n_symbols))
    np.add.at(expected, (new_index[:, None], new_index[None, :]), old_flow)
    new_flow = result.state_p[:, None, None] * result.transition_p[..., None] * result.output_p
    assert np.allclose(new_flow, expected, rtol=MERGE_RTOL, atol=MERGE_ATOL)


# the domain of the merge


@pytest.fixture(scope="module")
def with_transients():
    return _transient_params(np.random.default_rng([SEED, 2]), 5, 2, 4)


@pytest.mark.parametrize("pair", [(1, 1), (4, 4)])
def test_a_state_cannot_be_merged_with_itself(with_transients, pair):
    with pytest.raises(ValueError, match="with itself"):
        _merge_states(with_transients, *pair)


@pytest.mark.parametrize("pair", [(-1, 2), (2, 5), (7, 0)])
def test_a_merge_outside_the_model_raises_value_error(with_transients, pair):
    with pytest.raises(ValueError, match=r"must lie in \[0, 5\)"):
        _merge_states(with_transients, *pair)


@pytest.mark.parametrize("pair", [(1.0, 2), (2, True), ("1", 2), (None, 2)])
def test_a_merge_of_a_non_integer_raises_type_error(with_transients, pair):
    with pytest.raises(TypeError, match="must be an integer"):
        _merge_states(with_transients, *pair)


def test_a_merge_accepts_numpy_integers(with_transients):
    assert _merge_states(with_transients, np.int64(2), np.int32(4)).n_states == 4


def test_two_transient_states_are_refused(with_transients):
    """The original's `safe-/` built an all-zero row here, which `HMMParams` rejects."""
    assert closed_classes(with_transients.transition_p)[:2].tolist() == [-1, -1]
    with pytest.raises(ValueError, match="cannot merge states 0 and 1: both are transient"):
        _merge_states(with_transients, 1, 0)


def test_a_reducible_incumbent_is_refused():
    vocabulary = _vocabulary(3)
    output = np.zeros((4, 4, USER_BASE + 3))
    output[..., USER_BASE:] = 1 / 3
    params = HMMParams(
        np.full(4, 0.25),
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
        output,
        vocabulary,
    )
    with pytest.raises(ValueError, match="reducible"):
        _merge_states(params, 0, 2)


# a split undone by a merge


def _round_trip_cases():
    out = []
    for case in _merge_incumbents():
        params = case.values[0]
        for s in range(params.n_states):
            out.append(pytest.param((params, s), id=f"{case.id}-s{s}"))
    out.append(pytest.param((load_params(FIXTURES / "m001_0001_001.hmm"), 0), id="m001_0001_001-s0"))
    return out


@pytest.mark.parametrize("case", _round_trip_cases())
def test_merging_the_twins_of_an_unseeded_split_restores_the_incumbent(case, monkeypatch):
    """Exact where the split's halves sum back by Sterbenz's lemma; within rounding
    where a fibre average or the stationary weights enter.

    A transient state's twins are both transient, so that split cannot be undone."""
    params, s = case
    monkeypatch.setattr(_topology, "_SEED_WEIGHT", 0.0)
    split = _split_state(params, s, rng=np.random.default_rng([SEED, s]))
    size = params.n_states
    if closed_classes(params.transition_p)[s] < 0:
        with pytest.raises(ValueError, match="both are transient"):
            _merge_states(split, s, size)
        return

    restored = _merge_states(split, size, s)
    assert np.array_equal(restored.init_state_p, params.init_state_p)
    others = np.flatnonzero(np.arange(size) != s)
    assert np.array_equal(restored.transition_p[others, s], params.transition_p[others, s])
    assert np.array_equal(
        restored.transition_p[np.ix_(others, others)], params.transition_p[np.ix_(others, others)]
    )
    assert np.allclose(restored.transition_p, params.transition_p, rtol=MERGE_RTOL, atol=MERGE_ATOL)
    live = params.transition_p > 0
    assert np.allclose(restored.output_p[live], params.output_p[live], rtol=MERGE_RTOL, atol=MERGE_ATOL)
