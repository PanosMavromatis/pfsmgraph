"""Forward-backward, checked against exact enumeration and against itself.

**Every test here reaches a private function**, and that is labelled rather than
hidden, as ADR 0003 asks. There is no public forward-backward yet: the kernel is
revision 03's numpy reference, and what a caller eventually sees, whether a
trainer or a scoring function, is decided by later subgoals. Until then the
kernel is the only thing there is to test.

Four sections, each catching what the one before cannot:

- **Identities** that hold between the kernel's outputs whatever the model: γ
  sums to 1, ξ marginalises to γ from both ends, and so on. Necessary and not
  sufficient: a kernel that is correct for the *wrong* model passes all of them.
  An arc table transposed everywhere was caught by this section only through
  dead-arc structure, never through a value.
- **The oracle**: every state path enumerated in exact rational arithmetic, which
  shares neither the recurrence nor any rounding with the kernel. This is what
  catches the transposed table on a dense model.
- **Constructed cases** the learned fixtures never exhibit: an impossible
  sequence, an exactly uniform model, a state reachable only through dead arcs.
- **ADR 0020's evaluation order**, pinned bit for bit against literal scalar
  loops, since bit-exactness with the compiled phases depends on it and no
  tolerance-based test can see a reordered sum.

Tolerances are chosen per assertion from the number of terms summed, not
loosened until green, and the uniform model and the order pin assert equality.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

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


# --- the oracle: exact enumeration over every state path ------------------------
#
# The identities above hold for any model, so a kernel that is correct for the
# wrong one passes them. This section computes the answer independently: every
# state path is enumerated and weighted in exact rational arithmetic, which
# shares neither the recurrence nor any rounding with the kernel. A float is
# converted with `Fraction(x)`, which is exact, so the oracle reads precisely the
# values the kernel reads, and any disagreement beyond the float64 rounding of
# the kernel's own result is the kernel's.
#
# Models are kept to about 2000 paths: rationals cost ~0.1-0.3 s at that size and
# grow as S**(N+1). That bounds S, not the kinds of structure covered, which is
# what the cases below vary instead.

#: The kernel's results are float64 sums of positive terms, each within an ulp or
#: so of exact. Relative error at the size of these models is ~1e-15, measured
#: at 5.9e-15 in the goal that wrote the kernel; 1e-12 leaves headroom and would
#: still expose a single mis-weighted path among ~2000. `atol=0` is deliberate: an
#: exact zero in the oracle is a zero-probability event, and the kernel must
#: produce an exact zero there too, not a small number.
EXACT_RTOL = 1e-12


def _exact(init, transition, output, codes):
    """Exact `(bits, gamma, init_counts, transition_counts, emission_counts)`.

    Brute force, in `Fraction`, over all `S ** (N + 1)` state paths. A path's
    weight is `init[s0] * prod_t transition[s_t, s_t+1] * output[s_t, s_t+1,
    codes[t]]`; posteriors and counts are weight-averaged indicators, divided by
    the total at the end.
    """
    size, n_symbols = transition.shape[0], output.shape[2]
    n = len(codes)
    I = [Fraction(float(x)) for x in init]
    T = [[Fraction(float(x)) for x in row] for row in transition]
    O = [[[Fraction(float(x)) for x in fibre] for fibre in row] for row in output]

    total = Fraction(0)
    gamma = [[Fraction(0)] * size for _ in range(n + 1)]
    trans = [[Fraction(0)] * size for _ in range(size)]
    emit = [[[Fraction(0)] * n_symbols for _ in range(size)] for _ in range(size)]
    for path in itertools.product(range(size), repeat=n + 1):
        weight = I[path[0]]
        for t, code in enumerate(codes):
            weight *= T[path[t]][path[t + 1]] * O[path[t]][path[t + 1]][code]
            if not weight:
                break
        if not weight:
            continue
        total += weight
        for t, state in enumerate(path):
            gamma[t][state] += weight
        for t, code in enumerate(codes):
            trans[path[t]][path[t + 1]] += weight
            emit[path[t]][path[t + 1]][code] += weight

    assert total > 0, "the oracle is for possible sequences; see the constructed cases"
    bits_total = -(math.log2(total.numerator) - math.log2(total.denominator))
    as_float = np.vectorize(lambda x: float(x / total), otypes=[np.float64])
    gamma = as_float(np.array(gamma, dtype=object))
    init_counts = gamma[0] if n else np.zeros(size)
    return (
        bits_total,
        gamma,
        init_counts,
        as_float(np.array(trans, dtype=object)),
        as_float(np.array(emit, dtype=object)),
    )


def _enumeration_cases():
    rng = np.random.default_rng(SEED + 1)
    cases = []
    for size, n_symbols, length, dead in (
        (1, 3, 6, 0.0),
        (2, 3, 6, 0.0),
        (2, 2, 9, 0.5),
        (3, 4, 5, 0.0),
        (3, 4, 6, 0.4),
        (4, 3, 4, 0.5),
    ):
        init, transition, output = _random_model(rng, size, n_symbols, dead)
        codes = rng.integers(0, n_symbols, length)
        cases.append(
            pytest.param(
                (init, transition, output, codes),
                id=f"S{size}-A{n_symbols}-N{length}-dead{dead}",
            )
        )
    # A seed summing to 1 + 1e-6, inside HMMParams's SUM_TOL. The kernel does not
    # renormalise row 0, so it must agree with enumeration under the parameters
    # exactly as given. A kernel that renormalised would be 1.4e-6 bits off,
    # which EXACT_RTOL resolves.
    init, transition, output = _random_model(rng, 3, 3, 0.3)
    cases.append(
        pytest.param(
            (init * (1.0 + 1e-6), transition, output, rng.integers(0, 3, 6)),
            id="seed-sums-to-1+1e-6",
        )
    )
    return cases


@pytest.fixture(scope="module", params=_enumeration_cases())
def enumerated(request):
    init, transition, output, codes = request.param
    alpha, beta, scale = _forward_backward(init, transition, output, codes)
    return {
        "kernel": (
            _description_length(scale),
            _state_posteriors(alpha, beta, scale),
            *_expected_counts(alpha, beta, transition, output, codes),
        ),
        "exact": _exact(init, transition, output, [int(c) for c in codes]),
    }


def test_the_description_length_is_minus_log2_of_the_exact_probability(enumerated):
    got, want = enumerated["kernel"][0], enumerated["exact"][0]
    assert got == pytest.approx(want, rel=EXACT_RTOL, abs=0)


def test_the_state_posteriors_are_the_exact_occupation_probabilities(enumerated):
    np.testing.assert_allclose(
        enumerated["kernel"][1], enumerated["exact"][1], rtol=EXACT_RTOL, atol=0
    )


def test_the_initial_counts_are_exact(enumerated):
    np.testing.assert_allclose(
        enumerated["kernel"][2], enumerated["exact"][2], rtol=EXACT_RTOL, atol=0
    )


def test_the_transition_counts_are_exact(enumerated):
    np.testing.assert_allclose(
        enumerated["kernel"][3], enumerated["exact"][3], rtol=EXACT_RTOL, atol=0
    )


def test_the_emission_counts_are_exact(enumerated):
    np.testing.assert_allclose(
        enumerated["kernel"][4], enumerated["exact"][4], rtol=EXACT_RTOL, atol=0
    )


# --- constructed cases the learned fixtures never exhibit -----------------------


def test_an_impossible_sequence_is_infinite_bits_and_zero_everything_else():
    # State 0 can only emit symbol 0 and stay; state 1 only symbol 1 and stay;
    # the chain starts in state 0. Symbol index 2 of the record is a 1, which
    # no reachable arc emits.
    init = np.array([1.0, 0.0])
    transition = np.eye(2)
    output = np.zeros((2, 2, 2))
    output[0, 0, 0] = 1.0
    output[1, 1, 1] = 1.0
    codes = np.array([0, 0, 1, 0])

    # "raise" turns any invalid or dividing operation into an error, so this
    # asserts no 0/0 or x/0 happens anywhere on the path, not only that none
    # survives into the outputs.
    with np.errstate(all="raise"):
        alpha, beta, scale = _forward_backward(init, transition, output, codes)
        gamma = _state_posteriors(alpha, beta, scale)
        counts = _expected_counts(alpha, beta, transition, output, codes)
    bits_total = _description_length(scale)

    np.testing.assert_array_equal(scale, [1.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(alpha[:3], [[1.0, 0.0]] * 3)
    assert not alpha[3:].any()
    assert bits_total == math.inf
    assert not beta.any()
    assert not gamma.any()
    for array in (alpha, beta, gamma, *counts):
        assert not np.isnan(array).any()
    for array in counts:
        assert not array.any()


@pytest.mark.parametrize("length", [1, 7, 40])
def test_an_exactly_uniform_model_is_computed_exactly(length):
    # Every probability is 1/4, which float64 represents exactly, so every
    # intermediate value is dyadic and nothing rounds. The assertions are
    # therefore equalities. This is the model rand_p_vector(noise_width=0)
    # initialises with, and it ties at every position.
    size = n_symbols = 4
    init = np.full(size, 0.25)
    transition = np.full((size, size), 0.25)
    output = np.full((size, size, n_symbols), 0.25)
    codes = np.random.default_rng(SEED).integers(0, n_symbols, length)

    alpha, beta, scale = _forward_backward(init, transition, output, codes)
    init_counts, transition_counts, emission_counts = _expected_counts(
        alpha, beta, transition, output, codes
    )

    np.testing.assert_array_equal(scale[1:], 0.25)
    assert _description_length(scale) == 2.0 * length
    np.testing.assert_array_equal(_state_posteriors(alpha, beta, scale), 0.25)
    np.testing.assert_array_equal(init_counts, 0.25)
    np.testing.assert_array_equal(transition_counts, length / 16)
    for symbol in range(n_symbols):
        expected = np.count_nonzero(codes == symbol) / 16
        np.testing.assert_array_equal(emission_counts[:, :, symbol], expected)


def test_a_state_reachable_only_through_dead_arcs_is_never_occupied():
    # The 3-state model ADR 0020 used for its autograd evidence: state 2 cannot
    # start and no live arc enters it, so it is unreachable at every position.
    # In log space torch's gradient was nan in every transition there. Here the
    # posteriors and counts touching state 2 must be exact zeros, with no nan.
    init = np.array([0.5, 0.5, 0.0])
    transition = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
    output = np.full((3, 3, 2), 0.5)
    codes = np.array([0, 1, 1, 0])

    with np.errstate(all="raise"):
        alpha, beta, scale = _forward_backward(init, transition, output, codes)
        gamma = _state_posteriors(alpha, beta, scale)
        init_counts, transition_counts, emission_counts = _expected_counts(
            alpha, beta, transition, output, codes
        )

    assert not gamma[:, 2].any()
    assert init_counts[2] == 0.0
    assert not transition_counts[2].any() and not transition_counts[:, 2].any()
    assert not emission_counts[2].any() and not emission_counts[:, 2].any()
    np.testing.assert_allclose(gamma[:, :2], 0.5, rtol=0, atol=ROW_TOL)
    # Every live probability is 1/2 and every path uses 2 of the 4 live arcs
    # per step with weight 1/4 each: P = (1/2)**N, exactly 1 bit per symbol.
    assert _description_length(scale) == pytest.approx(len(codes), abs=ROW_TOL)


# --- ADR 0020's evaluation order, pinned ----------------------------------------
#
# The reference reduces with np.add.accumulate because it returns every partial
# sum, which fixes the order by definition. That is a property of numpy relied
# on for bit-exactness, so it is pinned here against the literal specification:
# scalar Python floats, ascending loops, nothing vectorised. Equality, not
# closeness: a reduction that reordered would differ by an ulp in about half of
# all scale factors, which a tolerance would accept.


def _loop_reference(init, transition, output, codes):
    size, n = transition.shape[0], len(codes)
    n_symbols = output.shape[2]
    T = [[float(x) for x in row] for row in transition]
    O = [[[float(x) for x in fibre] for fibre in row] for row in output]
    w = [[[T[i][j] * O[i][j][k] for k in range(n_symbols)] for j in range(size)] for i in range(size)]

    alpha = [[float(x) for x in init]]
    scale = [1.0]
    for t in range(1, n + 1):
        k = codes[t - 1]
        column = []
        for j in range(size):
            acc = 0.0
            for i in range(size):
                acc = acc + alpha[t - 1][i] * w[i][j][k]
            column.append(acc)
        q = 0.0
        for j in range(size):
            q = q + column[j]
        scale.append(q)
        alpha.append([c / q if q != 0.0 else 0.0 for c in column])

    beta = [None] * (n + 1)
    beta[n] = [1.0 / scale[n] if scale[n] != 0.0 else 0.0] * size
    for t in range(n - 1, -1, -1):
        k = codes[t]
        row = []
        for i in range(size):
            acc = 0.0
            for j in range(size):
                acc = acc + w[i][j][k] * beta[t + 1][j]
            row.append(acc / scale[t] if scale[t] != 0.0 else 0.0)
        beta[t] = row

    bits_total = 0.0
    for q in scale:
        bits_total = bits_total + (-math.log2(q) if q > 0.0 else math.inf)

    init_counts = [0.0] * size
    trans = [[0.0] * size for _ in range(size)]
    emit = [[[0.0] * n_symbols for _ in range(size)] for _ in range(size)]
    for t in range(n):
        k = codes[t]
        for i in range(size):
            row_total = 0.0
            for j in range(size):
                xi = (alpha[t][i] * w[i][j][k]) * beta[t + 1][j]
                row_total = row_total + xi
                trans[i][j] = trans[i][j] + xi
                emit[i][j][k] = emit[i][j][k] + xi
            if t == 0:
                init_counts[i] = row_total

    return (
        np.array(alpha),
        np.array(beta),
        np.array(scale),
        bits_total,
        np.array(init_counts),
        np.array(trans),
        np.array(emit),
    )


def test_the_reference_reproduces_the_literal_ascending_loops_bit_for_bit():
    rng = np.random.default_rng(SEED + 2)
    init, transition, output = _random_model(rng, 16, 6, 0.3)
    codes = rng.integers(0, 6, 50)

    alpha, beta, scale = _forward_backward(init, transition, output, codes)
    got = (
        alpha,
        beta,
        scale,
        _description_length(scale),
        *_expected_counts(alpha, beta, transition, output, codes),
    )
    want = _loop_reference(init, transition, output, [int(c) for c in codes])

    names = ("alpha", "beta", "scale", "bits", "init_counts", "transition_counts", "emission_counts")
    for name, g, w in zip(names, got, want):
        np.testing.assert_array_equal(g, w, err_msg=name)
