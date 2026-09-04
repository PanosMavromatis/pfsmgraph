"""`HMMParams` -- the ADR 0017 frozen parameter value.

Organized by what is being defended rather than by method: the shape contract,
the distribution contract, the reserved block, the freezing discipline, the
derived quantities, and a differential check against the original's own saved
models.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pfsmgraph.dataseq import PAD, UNK, USER_BASE, SymbolTable

from pfsmgraph.hmm import HMMParams
from pfsmgraph.hmm._numeric import bits
from pfsmgraph.hmm._params import SUM_TOL

from _lush_fixtures import (
    FIXTURES,
    SAVED_MODELS,
    load_params,
    read_ascii_matrix,
    read_scalar,
)


def _vocabulary(n_user=3):
    return SymbolTable([f"s{i}" for i in range(n_user)])


def _uniform_output(size, vocabulary):
    """Emission fibres uniform over the user symbols, zero on the reserved block."""
    n_user = vocabulary.size - USER_BASE
    output_p = np.zeros((size, size, vocabulary.size))
    output_p[:, :, USER_BASE:] = 1.0 / n_user
    return output_p


def _params(size=2, vocabulary=None, **overrides):
    """A well-formed parameter set, with any field replaceable."""
    vocabulary = _vocabulary() if vocabulary is None else vocabulary
    fields = {
        "init_state_p": np.full(size, 1.0 / size),
        "transition_p": np.full((size, size), 1.0 / size),
        "output_p": _uniform_output(size, vocabulary),
        "vocabulary": vocabulary,
    }
    fields.update(overrides)
    return HMMParams(**fields)


# ---------------------------------------------------------------- shape


def test_it_accepts_a_well_formed_model():
    params = _params()
    assert params.n_states == 2
    assert params.n_symbols == _vocabulary().size


def test_n_symbols_is_derived_from_output_p_not_stored():
    """ADR 0017 is explicit that this is derived, so the two cannot disagree."""
    params = _params()
    assert params.n_symbols == params.output_p.shape[2]
    assert not hasattr(params, "_n_symbols")


def test_repr_does_not_dump_the_arrays():
    assert repr(_params()) == "HMMParams(n_states=2, n_symbols=9)"


def test_init_state_p_must_be_one_dimensional():
    with pytest.raises(ValueError, match=r"init_state_p must be 1-D"):
        _params(init_state_p=np.full((2, 2), 0.25))


def test_transition_p_must_be_square_and_match_the_state_count():
    with pytest.raises(ValueError, match=r"transition_p must be \(2, 2\)"):
        _params(transition_p=np.full((2, 3), 1.0 / 3.0))


def test_output_p_must_be_three_dimensional():
    """Two axes is the state-emission shape, which is the wrong formulation."""
    with pytest.raises(ValueError, match=r"output_p must be 3-D"):
        _params(output_p=np.full((2, 9), 1.0 / 9.0))


def test_output_p_first_two_axes_are_source_and_destination_state():
    vocabulary = _vocabulary()
    wrong = np.zeros((2, 3, vocabulary.size))
    wrong[:, :, USER_BASE:] = 1.0 / 3.0
    with pytest.raises(ValueError, match=r"source and destination state"):
        _params(output_p=wrong)


def test_the_symbol_axis_must_equal_the_vocabulary_size():
    """And the message names the alternative that fails silently."""
    vocabulary = _vocabulary()
    short = np.full((2, 2, vocabulary.size - USER_BASE), 1.0 / 3.0)
    with pytest.raises(ValueError, match=r"index backwards without error"):
        _params(output_p=short)


# -------------------------------------------------------- distributions


@pytest.mark.parametrize("field", ["init_state_p", "transition_p", "output_p"])
def test_a_negative_value_is_not_a_probability(field):
    params = _params()
    array = np.array(getattr(params, field))
    array.flat[-1] = -0.5
    with pytest.raises(ValueError, match=rf"{field} contains a negative value"):
        _params(**{field: array})


def test_a_non_finite_value_is_rejected():
    """+inf means 'impossible' in the bit domain, not in the probability domain."""
    params = _params()
    array = np.array(params.init_state_p)
    array[0] = np.inf
    with pytest.raises(ValueError, match=r"init_state_p contains a non-finite value"):
        _params(init_state_p=array)


def test_init_state_p_must_sum_to_one():
    with pytest.raises(ValueError, match=r"init_state_p sums to 0\.75"):
        _params(init_state_p=np.array([0.5, 0.25]))


def test_every_transition_row_must_sum_to_one_and_the_message_names_them():
    bad = np.array([[0.5, 0.5], [0.2, 0.2]])
    with pytest.raises(ValueError, match=r"row 1 sums to 0\.4.*Rows are \[1\]"):
        _params(transition_p=bad)


def test_a_zero_transition_row_is_rejected_rather_than_exempted():
    """Revision 04 can construct one, and it should learn that here.

    `HMMLIB-ACCOUNT.md` section 5: `merge-states` divides with `safe-/`, so
    merging two unreachable states yields zeros. Accepting the result would hand
    a chain with no stationary distribution to a solve that has to invent one.
    """
    with pytest.raises(ValueError, match=r"row 1 sums to 0\.0"):
        _params(transition_p=np.array([[0.5, 0.5], [0.0, 0.0]]))


def test_an_emission_fibre_on_a_live_arc_must_sum_to_one():
    params = _params()
    output_p = np.array(params.output_p)
    output_p[0, 1, USER_BASE:] = 0.1
    with pytest.raises(ValueError, match=r"output_p\[0, 1\] sums to 0\.3"):
        _params(output_p=output_p)


def test_an_emission_fibre_on_a_dead_arc_is_not_checked():
    """It cannot reach the recurrence: bits(0) on the transition absorbs the path.

    This exemption is load-bearing rather than theoretical -- the original's own
    saved models are full of all-zero fibres on arcs of probability zero, and a
    blanket rule would reject them.
    """
    vocabulary = _vocabulary()
    output_p = _uniform_output(2, vocabulary)
    output_p[0, 1, :] = 0.0
    params = _params(
        transition_p=np.array([[1.0, 0.0], [0.5, 0.5]]), output_p=output_p
    )
    assert params.output_p[0, 1].sum() == 0.0


def test_the_tolerance_admits_float32_noise_but_not_four_decimal_drift():
    """`SUM_TOL` is chosen to sit between two known magnitudes, not picked round.

    The lower bound is the one that bit: an earlier `1e-6` was *below* the drift
    of a vector normalized in float32 over a symbol axis of a few dozen, so it
    would have rejected the output of the torch backend ADR 0017 anticipates.
    """
    float32_drift = float(np.sqrt(100) * np.finfo(np.float32).eps)
    assert float32_drift < SUM_TOL < 1e-4


def test_a_float32_normalized_vector_is_accepted():
    """The concrete case behind the bound above, rather than only the arithmetic."""
    rng = np.random.default_rng(20260903)
    row = rng.random(64).astype(np.float32)
    row /= row.sum()
    assert abs(float(np.float64(row).sum()) - 1.0) < SUM_TOL


# ------------------------------------------------------- reserved block


def test_a_non_zero_reserved_fibre_is_rejected_and_named():
    params = _params()
    output_p = np.array(params.output_p)
    output_p[0, 0, PAD] = 0.5
    with pytest.raises(ValueError, match=r"reserved symbol PAD \(code 0\)"):
        _params(output_p=output_p)


def test_unk_is_the_reserved_code_that_actually_arises():
    """`encode(..., on_unknown="unk")` is a documented dataseq path, so UNK reaches
    a record where PAD, BOS, EOS, GAP and MSK do not."""
    params = _params()
    output_p = np.array(params.output_p)
    output_p[1, 1, UNK] = 0.25
    with pytest.raises(ValueError, match=r"reserved symbol UNK \(code 1\)"):
        _params(output_p=output_p)


def test_emitting_a_reserved_symbol_is_impossible_by_arithmetic():
    """The payoff for spending six fibres: no special case anywhere downstream.

    A zero emission probability becomes +inf bits, which absorbs under addition,
    so a path through a reserved symbol is reported impossible rather than
    scored with some other symbol's probability.
    """
    params = _params()
    for code in range(USER_BASE):
        assert params.output_p[0, 0, code] == 0.0
        assert bits(params.output_p[0, 0, code]) == np.inf


# ------------------------------------------------------------- freezing


def test_rebinding_an_attribute_raises():
    params = _params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.init_state_p = np.array([1.0, 0.0])


@pytest.mark.parametrize("field", ["init_state_p", "transition_p", "output_p"])
def test_the_buffers_are_read_only_not_merely_the_bindings(field):
    """ADR 0017 states this in its Decision because omitting it raises nothing."""
    array = getattr(_params(), field)
    assert not array.flags.writeable
    with pytest.raises(ValueError, match=r"read-only"):
        array[...] = 0.0


def test_the_callers_array_is_not_aliased():
    """asarray may hand back the caller's own array; freezing that would mutate
    an object this type does not own."""
    caller = np.array([0.5, 0.5])
    params = _params(init_state_p=caller)
    assert caller.flags.writeable
    caller[0] = 99.0
    assert params.init_state_p[0] == 0.5


@pytest.mark.parametrize("name", ["state_p", "state_entropies"])
def test_derived_arrays_are_frozen_too(name):
    """Freezing the inputs alone does not deliver the ADR's claim.

    A cached property hands back the same object every time, so a caller who
    writes into it corrupts the value every later reader sees -- the
    stored-and-stale failure ADR 0017 removes, reintroduced one level out.
    """
    array = getattr(_params(), name)
    assert not array.flags.writeable
    with pytest.raises(ValueError, match=r"read-only"):
        array[...] = 0.0


def test_cached_property_works_on_a_frozen_dataclass():
    """An implementation detail of functools, not a documented guarantee.

    `cached_property` stores through `instance.__dict__[name]` rather than via
    `__setattr__`, which is the only reason a frozen dataclass tolerates it. It
    would break under `slots=True`, so the compatibility is asserted rather than
    assumed.
    """
    params = _params()
    assert params.state_p is params.state_p
    assert params.state_entropies is params.state_entropies


# ------------------------------------------------------------- derived


def test_state_p_matches_a_hand_solvable_chain():
    """pi = pi P for [[0.7, 0.3], [0.5, 0.5]] gives pi_0 = 0.625 exactly."""
    params = _params(transition_p=np.array([[0.7, 0.3], [0.5, 0.5]]))
    assert params.state_p == pytest.approx([0.625, 0.375])


def test_state_entropies_of_a_uniform_marginal_is_log2_of_the_symbol_count():
    vocabulary = _vocabulary(n_user=4)
    params = _params(vocabulary=vocabulary)
    assert params.state_entropies == pytest.approx([np.log2(4)] * 2)


def test_the_marginal_sums_over_arcs_not_over_states():
    """p_i(k) = sum_j transition_p[i, j] * output_p[i, j, k] -- the arc-emission
    marginal. A state-emission reading would index output_p[i, k] and there is no
    such array."""
    vocabulary = _vocabulary(n_user=2)
    output_p = np.zeros((2, 2, vocabulary.size))
    # From state 0: arc to 0 always emits the first symbol, arc to 1 the second.
    output_p[0, 0, USER_BASE] = 1.0
    output_p[0, 1, USER_BASE + 1] = 1.0
    output_p[1, :, USER_BASE] = 1.0
    params = _params(
        vocabulary=vocabulary,
        transition_p=np.array([[0.5, 0.5], [1.0, 0.0]]),
        output_p=output_p,
    )
    # State 0 marginalizes to a fair coin over two symbols; state 1 is certain.
    assert params.state_entropies == pytest.approx([1.0, 0.0])


def test_model_entropy_is_the_stationary_weighted_average():
    params = _params(transition_p=np.array([[0.7, 0.3], [0.5, 0.5]]))
    assert params.entropy == pytest.approx(
        float(params.state_p @ params.state_entropies)
    )


def test_a_reducible_chain_fails_on_the_attribute_access():
    """ADR 0017 makes state_p derived, so the failure surfaces here rather than at
    construction -- which is why the message has to name the cause."""
    reducible = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    vocabulary = _vocabulary()
    params = _params(
        size=4,
        vocabulary=vocabulary,
        transition_p=reducible,
        output_p=_uniform_output(4, vocabulary),
        init_state_p=np.full(4, 0.25),
    )
    with pytest.raises(ValueError, match=r"reducible"):
        params.state_p


# -------------------------------------------------------- differential


@pytest.mark.parametrize("model", SAVED_MODELS)
def test_a_saved_model_is_constructible_after_the_renumbering(model):
    """The fixture's alphabet starts at code 2; ADR 0011 puts user symbols at 6."""
    saved_output_p = read_ascii_matrix(FIXTURES / model / "output_p")
    params = load_params(FIXTURES / model)
    assert params.n_symbols == USER_BASE + saved_output_p.shape[2]
    assert np.all(params.output_p[:, :, :USER_BASE] == 0.0)


@pytest.mark.parametrize("model", SAVED_MODELS)
def test_state_entropies_reproduce_the_originals_own_numbers(model):
    """`update-entropy` (hmm.lsh:247-255) marginalizes over arcs, then takes the
    Shannon entropy. The saved values are what it produced."""
    params = load_params(FIXTURES / model)
    saved = read_ascii_matrix(FIXTURES / model / "state_entropies")
    # 5e-4, matching the bound test_numeric.py derived for this same quantity in
    # PR #16, and measured here at 1.27e-4 on the 8-state model. Both inputs are
    # rounded to +/- 5e-5, which for a probability near 0.1 is a *relative* error
    # of 5e-4; d(entropy)/dp is about 1.9 there, so the print's own precision is
    # not the binding term. Renormalizing fixes the sums, not the propagation.
    assert params.state_entropies == pytest.approx(saved, abs=5e-4)


@pytest.mark.parametrize("model", SAVED_MODELS)
def test_model_entropy_reproduces_the_originals_own_number(model):
    """entropy = sum_i state_p[i] * state_entropies[i] (hmm.lsh:258-262)."""
    params = load_params(FIXTURES / model)
    # Same bound as state_entropies: a stationary-weighted average of quantities
    # that can each be off by 5e-4 is not more accurate than they are, even
    # though this one happens to land inside 1e-4 on all three fixtures today.
    assert params.entropy == pytest.approx(
        read_scalar(FIXTURES / model / "entropy"), abs=5e-4
    )
