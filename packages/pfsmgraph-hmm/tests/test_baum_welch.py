"""The M-step, checked against exact rational division and constructed zero counts.

**Every test here reaches a private function**, labelled rather than hidden as
ADR 0003 asks: there is no public trainer yet, and what a caller eventually sees
is decided by later subgoals.

The count arrays come from the real E-step over random models and over the
tracked 8-state fixture, not from invented numbers. The M-step's contract is
about *consistent* counts, where the emission counts summed over symbols are the
transition counts, and a synthetic array would not have that property unless the
test reimplemented the E-step to give it one.

Four sections:

- **The oracle**: each division redone in exact rational arithmetic over the very
  floats the M-step read, so the only disagreement allowed is the float64
  rounding of a sum of `S` terms and one division.
- **Stochasticity and structure**: live rows and live fibres sum to 1, dead arcs
  stay dead, reserved fibres stay zero, and the fixture's re-estimate is a valid
  `HMMParams`.
- **Constructed zero counts** the fixtures never exhibit: a state reachable only
  by the final symbol, and a record with no counts at all.
- **ADR 0020's order**, pinned bit for bit against literal ascending loops.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SymbolTable
from pfsmgraph.hmm import HMMParams
from pfsmgraph.hmm._baum_welch import _m_step
from pfsmgraph.hmm._forward_backward import _expected_counts, _forward_backward

from _lush_fixtures import FIXTURES, load_corpus_record, load_params

SEED = 20260915

#: A re-estimated row is `S` quotients whose numerators were summed by the
#: E-step and whose denominator was summed again here, each within an ulp or so;
#: a fibre is the same over `A` symbols. 1e-12 leaves headroom at the fixture's
#: 1268 positions and is still orders below any misindexed division.
ROW_TOL = 1e-12

#: The M-step's result against exact division of the same floats: one rounded
#: sum of at most a few dozen positive terms, then one correctly rounded division.
EXACT_RTOL = 1e-12


def _random_model(rng, size, n_symbols, dead_fraction):
    """A random arc-emission model with some arcs exactly dead, diagonal kept live."""
    init = rng.dirichlet(np.ones(size))
    transition = rng.dirichlet(np.ones(size), size=size)
    dead = rng.random((size, size)) < dead_fraction
    dead[np.arange(size), np.arange(size)] = False
    transition = np.where(dead, 0.0, transition)
    transition /= transition.sum(axis=1, keepdims=True)
    output = rng.dirichlet(np.ones(n_symbols), size=(size, size))
    return init, transition, output


def _e_step(init, transition, output, codes):
    alpha, beta, _ = _forward_backward(init, transition, output, codes)
    return _expected_counts(alpha, beta, transition, output, codes)


def _random_cases():
    rng = np.random.default_rng(SEED)
    cases = []
    for size, n_symbols, length, dead in (
        (1, 3, 12, 0.0),
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
    params = load_params(FIXTURES / "m008_0001_008.hmm")
    record = load_corpus_record()
    return pytest.param(
        (params.init_state_p, params.transition_p, params.output_p, record.codes),
        id="m008_0001_008-corpus",
    )


@pytest.fixture(scope="module", params=[*_random_cases(), _fixture_case()])
def stepped(request):
    init, transition, output, codes = request.param
    counts = _e_step(init, transition, output, codes)
    return {
        "transition": transition,
        "counts": counts,
        "result": _m_step(*counts),
    }


# --- the oracle: exact rational division ----------------------------------------


def _exact_quotients(numerators, denominators):
    """`numerators / denominators` in `Fraction`, with the M-step's 0-for-x/0 rule."""
    num = np.vectorize(Fraction, otypes=[object])(numerators)
    den = np.vectorize(Fraction, otypes=[object])(denominators)
    quotient = np.vectorize(
        lambda n, d: float(n / d) if d else 0.0, otypes=[np.float64]
    )
    return quotient(num, den)


def test_the_initial_distribution_is_the_exact_normalised_count(stepped):
    init_counts = stepped["counts"][0]
    total = sum((Fraction(float(c)) for c in init_counts), Fraction(0))
    want = np.array([float(Fraction(float(c)) / total) for c in init_counts])
    np.testing.assert_allclose(stepped["result"][0], want, rtol=EXACT_RTOL, atol=0)


def test_the_outgoing_counts_are_the_exact_row_sums(stepped):
    transition_counts = stepped["counts"][1]
    want = [
        float(sum((Fraction(float(c)) for c in row), Fraction(0)))
        for row in transition_counts
    ]
    np.testing.assert_allclose(stepped["result"][3], want, rtol=EXACT_RTOL, atol=0)


def test_the_transitions_are_the_exact_quotients(stepped):
    transition_counts = stepped["counts"][1]
    out = np.array(
        [sum((Fraction(float(c)) for c in row), Fraction(0)) for row in transition_counts],
        dtype=object,
    )
    want = _exact_quotients(transition_counts, out[:, np.newaxis])
    np.testing.assert_allclose(stepped["result"][1], want, rtol=EXACT_RTOL, atol=0)


def test_the_emissions_are_the_exact_quotients(stepped):
    _, transition_counts, emission_counts = stepped["counts"]
    want = _exact_quotients(emission_counts, transition_counts[:, :, np.newaxis])
    np.testing.assert_allclose(stepped["result"][2], want, rtol=EXACT_RTOL, atol=0)


# --- stochasticity and structure ------------------------------------------------


def test_the_initial_distribution_sums_to_one(stepped):
    assert stepped["result"][0].sum() == pytest.approx(1.0, abs=ROW_TOL)


def test_every_occupied_state_has_a_stochastic_row(stepped):
    transition_p, out_counts = stepped["result"][1], stepped["result"][3]
    occupied = out_counts > 0
    np.testing.assert_allclose(transition_p[occupied].sum(axis=1), 1.0, atol=ROW_TOL)


def test_every_live_arc_has_a_stochastic_fibre(stepped):
    transition_p, output_p = stepped["result"][1], stepped["result"][2]
    live = transition_p > 0
    np.testing.assert_allclose(output_p[live].sum(axis=1), 1.0, atol=ROW_TOL)


def test_a_dead_arc_stays_dead(stepped):
    # Zero is absorbing under EM: every count on an arc is a product containing
    # its old transition probability, so the re-estimate is an exact zero.
    dead = stepped["transition"] == 0
    assert not stepped["result"][1][dead].any()
    assert not stepped["result"][2][dead].any()


def test_the_fixture_re_estimate_is_a_valid_model_with_zero_reserved_fibres():
    params = load_params(FIXTURES / "m008_0001_008.hmm")
    counts = _e_step(
        params.init_state_p, params.transition_p, params.output_p,
        load_corpus_record().codes,
    )
    init_state_p, transition_p, output_p, out_counts = _m_step(*counts)

    assert (out_counts > 0).all()
    # Exact zeros, not small numbers: no reserved code occurs in the record, so no
    # count is ever added there, and 0 / x is exactly 0.
    assert not output_p[:, :, :USER_BASE].any()
    HMMParams(init_state_p, transition_p, output_p, params.vocabulary)


# --- constructed zero counts ----------------------------------------------------

A, B, C = 6, 7, 8


def _sink_at_the_end_model():
    """State 2 is reachable only by the final `b`, and can emit only `c`."""
    init = np.array([1.0, 0.0, 0.0])
    transition = np.array([[0.5, 0.5, 0.0], [0.4, 0.3, 0.3], [0.5, 0.5, 0.0]])
    output = np.zeros((3, 3, 9))
    output[0, 0, A] = output[0, 1, A] = 1.0
    output[1, 0, B] = output[1, 1, A] = output[1, 2, B] = 1.0
    output[2, 0, C] = output[2, 1, C] = 1.0
    return init, transition, output


def test_a_state_occupied_only_at_the_last_position_gets_a_zero_row():
    init, transition, output = _sink_at_the_end_model()
    counts = _e_step(init, transition, output, np.array([A, A, B, A, B]))

    with np.errstate(all="raise"):
        init_state_p, transition_p, output_p, out_counts = _m_step(*counts)

    # The case goal 1 of this branch measured: a valid row becomes all zero in
    # one step. The arc into state 2 survives, so the row is not unreachable.
    assert out_counts[2] == 0.0 and (out_counts[:2] > 0).all()
    assert not transition_p[2].any() and not output_p[2].any()
    assert init_state_p[2] == 0.0
    assert transition_p[1, 2] > 0
    with pytest.raises(ValueError):
        HMMParams(init_state_p, transition_p, output_p, _vocabulary(9))


def _vocabulary(size):
    return SymbolTable([f"s{i}" for i in range(size - USER_BASE)])


@pytest.mark.parametrize(
    "codes", [np.array([C, C]), np.array([], dtype=np.int64)], ids=["impossible", "empty"]
)
def test_a_record_with_no_counts_re_estimates_to_all_zeros_without_nan(codes):
    init, transition, output = _sink_at_the_end_model()
    counts = _e_step(init, transition, output, codes)

    with np.errstate(all="raise"):
        result = _m_step(*counts)

    for array in result:
        assert not array.any()


# --- ADR 0020's evaluation order, pinned ----------------------------------------


def test_both_reductions_are_the_literal_ascending_loops_bit_for_bit(stepped):
    init_counts, transition_counts, _ = stepped["counts"]
    init_state_p, _, _, out_counts = stepped["result"]

    init_total = 0.0
    for c in init_counts:
        init_total += c
    want_out = []
    for row in transition_counts:
        acc = 0.0
        for c in row:
            acc += c
        want_out.append(acc)

    assert out_counts.tolist() == want_out
    assert init_state_p.tolist() == [c / init_total for c in init_counts]
