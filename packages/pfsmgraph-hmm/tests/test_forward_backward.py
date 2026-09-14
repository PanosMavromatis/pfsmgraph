"""Forward-backward, checked for internal consistency.

**Every test here reaches a private function**, and that is labelled rather than
hidden, as ADR 0003 asks. There is no public forward-backward yet: the kernel is
revision 03's numpy reference, and what a caller eventually sees, whether a
trainer or a scoring function, is decided by later subgoals. Until then the
kernel is the only thing there is to test.

This module holds the identities that must hold between the kernel's outputs
whatever the model, which are the acceptance criteria of the goal that wrote the
posteriors and counts. They are necessary and not sufficient: a kernel that
computes a consistent but *wrong* distribution passes all of them. Exact
enumeration over every state path is the check for that, and joins this module
next, together with the constructed cases (an impossible sequence, an exactly
uniform model) that learned parameters never exhibit.

The identities compare different sums of the same terms, so they hold to
rounding rather than bit for bit. The tolerance is chosen per identity from the
number of terms summed, not loosened until green.
"""

from __future__ import annotations

import numpy as np
import pytest

from pfsmgraph.hmm._forward_backward import (
    _description_length,
    _expected_counts,
    _forward_backward,
    _state_posteriors,
)

from _lush_fixtures import FIXTURES, load_corpus_record, load_params

SEED = 20260914

#: Per-position identities sum `S` terms that are each within an ulp or so of
#: their true value; with `S` at most a few dozen here that is ~1e-14. 1e-12
#: leaves headroom for the float64 rounding of `alpha * beta * scale`, and is
#: still six orders below any error a wrong index would produce.
ROW_TOL = 1e-12


def _random_model(rng, size, n_symbols, dead_fraction):
    """A random arc-emission model with a fraction of its arcs exactly dead.

    Arrays only, not an `HMMParams`: the kernel is purely numeric and never
    reads a vocabulary. The diagonal is kept alive so every row stays
    normalisable.
    """
    init = rng.dirichlet(np.ones(size))
    transition = rng.dirichlet(np.ones(size), size=size)
    dead = rng.random((size, size)) < dead_fraction
    dead[np.arange(size), np.arange(size)] = False
    transition = np.where(dead, 0.0, transition)
    transition /= transition.sum(axis=1, keepdims=True)
    output = rng.dirichlet(np.ones(n_symbols), size=(size, size))
    return init, transition, output


def _random_cases():
    rng = np.random.default_rng(SEED)
    cases = []
    for size, n_symbols, length, dead in (
        (1, 3, 12, 0.0),
        (2, 2, 1, 0.0),
        (3, 4, 40, 0.3),
        (8, 6, 200, 0.4),
        (24, 10, 300, 0.5),
    ):
        init, transition, output = _random_model(rng, size, n_symbols, dead)
        codes = rng.integers(0, n_symbols, length)
        cases.append(
            pytest.param(
                (init, transition, output, codes),
                id=f"S{size}-A{n_symbols}-N{length}-dead{dead}",
            )
        )
    return cases


def _fixture_case():
    """The tracked 8-state model over the 1268-symbol corpus it was trained on."""
    params = load_params(FIXTURES / "m008_0001_008.hmm")
    record = load_corpus_record()
    return pytest.param(
        (params.init_state_p, params.transition_p, params.output_p, record.codes),
        id="m008_0001_008-corpus",
    )


@pytest.fixture(params=[*_random_cases(), _fixture_case()])
def solved(request):
    init, transition, output, codes = request.param
    alpha, beta, scale = _forward_backward(init, transition, output, codes)
    counts = _expected_counts(alpha, beta, transition, output, codes)
    return {
        "init": init,
        "transition": transition,
        "output": output,
        "codes": np.asarray(codes),
        "alpha": alpha,
        "beta": beta,
        "scale": scale,
        "gamma": _state_posteriors(alpha, beta, scale),
        "counts": counts,
    }


# --- shapes ---------------------------------------------------------------------


def test_the_outputs_have_the_arc_emission_geometry(solved):
    n = solved["codes"].shape[0]
    size = solved["transition"].shape[0]
    n_symbols = solved["output"].shape[2]
    init_counts, transition_counts, emission_counts = solved["counts"]
    assert solved["alpha"].shape == (n + 1, size)
    assert solved["beta"].shape == (n + 1, size)
    assert solved["scale"].shape == (n + 1,)
    assert solved["gamma"].shape == (n + 1, size)
    assert init_counts.shape == (size,)
    assert transition_counts.shape == (size, size)
    assert emission_counts.shape == (size, size, n_symbols)


# --- the identities -------------------------------------------------------------


def test_every_state_posterior_sums_to_one(solved):
    np.testing.assert_allclose(solved["gamma"].sum(axis=1), 1.0, rtol=0, atol=ROW_TOL)


def test_the_backward_pass_gives_the_forward_total(solved):
    # sum_i init[i] * beta[0, i] is P(codes) / (scale[1] * ... * scale[N]),
    # which is 1 exactly when the two passes agree about the likelihood.
    backward_total = float(np.dot(solved["init"], solved["beta"][0]))
    assert backward_total == pytest.approx(1.0, abs=ROW_TOL)


def test_the_pairwise_posteriors_marginalise_to_the_state_posteriors(solved):
    # sum_j xi_t[i, j] = gamma_t[i] at every t < N. Summed over t that is the
    # row sums of transition_counts; each side carries N rounded terms.
    _, transition_counts, _ = solved["counts"]
    n = solved["codes"].shape[0]
    expected = solved["gamma"][:n].sum(axis=0)
    np.testing.assert_allclose(
        transition_counts.sum(axis=1), expected, rtol=0, atol=max(n, 1) * ROW_TOL
    )


def test_the_pairwise_posteriors_also_marginalise_on_the_destination(solved):
    # sum_i xi_t[i, j] = gamma_{t+1}[j]: the same identity from the other end,
    # and the one a transposed index in beta would break.
    _, transition_counts, _ = solved["counts"]
    n = solved["codes"].shape[0]
    expected = solved["gamma"][1:].sum(axis=0)
    np.testing.assert_allclose(
        transition_counts.sum(axis=0), expected, rtol=0, atol=max(n, 1) * ROW_TOL
    )


def test_every_arc_crossing_emits_exactly_one_symbol(solved):
    _, transition_counts, emission_counts = solved["counts"]
    np.testing.assert_allclose(
        emission_counts.sum(axis=2), transition_counts, rtol=0, atol=ROW_TOL
    )


def test_the_counts_total_one_crossing_per_symbol(solved):
    _, transition_counts, _ = solved["counts"]
    n = solved["codes"].shape[0]
    assert transition_counts.sum() == pytest.approx(n, abs=max(n, 1) * ROW_TOL)


def test_emission_counts_sit_only_at_the_symbols_the_record_uses(solved):
    _, _, emission_counts = solved["counts"]
    unused = np.setdiff1d(np.arange(emission_counts.shape[2]), solved["codes"])
    assert not emission_counts[:, :, unused].any()


def test_emission_counts_sit_only_on_live_arcs(solved):
    _, transition_counts, emission_counts = solved["counts"]
    dead = solved["transition"] == 0.0
    assert not transition_counts[dead].any()
    assert not emission_counts[dead].any()


def test_the_initial_counts_are_the_first_state_posterior(solved):
    init_counts, _, _ = solved["counts"]
    np.testing.assert_allclose(init_counts, solved["gamma"][0], rtol=0, atol=ROW_TOL)


def test_the_description_length_is_finite_and_non_negative(solved):
    bits_total = _description_length(solved["scale"])
    assert np.isfinite(bits_total)
    assert bits_total >= 0.0


# --- the empty record -----------------------------------------------------------


def test_an_empty_record_has_no_crossings_and_costs_nothing():
    init, transition, output = _random_model(np.random.default_rng(SEED), 3, 4, 0.0)
    codes = np.array([], dtype=np.int64)
    alpha, beta, scale = _forward_backward(init, transition, output, codes)
    init_counts, transition_counts, emission_counts = _expected_counts(
        alpha, beta, transition, output, codes
    )
    np.testing.assert_array_equal(alpha, init[np.newaxis, :])
    np.testing.assert_array_equal(scale, [1.0])
    assert _description_length(scale) == 0.0
    np.testing.assert_array_equal(_state_posteriors(alpha, beta, scale), alpha)
    # run-add forms C-in from xi_0, and an empty record has none.
    assert not init_counts.any()
    assert not transition_counts.any()
    assert not emission_counts.any()
