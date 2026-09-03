"""The numeric primitives: the log-domain arithmetic and the stationary solve.

Several tests here assert the *absence* of the original's machinery rather than
the presence of ours. That is deliberate: the `-1` log-zero sentinel was
replaced by `+inf` on the grounds that IEEE-754 reproduces every property it
was built for, and the way to keep that claim honest is to test the properties
themselves, so that reinstating a sentinel breaks something. The stationary
solve's tests do the same in the other direction -- one of them asserts that
the homogeneous system *cannot* be solved as stated, so that the row
replacement cannot be "simplified" away without a failure.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import pfsmgraph.hmm
from pfsmgraph.hmm import _numeric
from pfsmgraph.hmm._numeric import bits, safe_divide, stationary_distribution


# --- bits: the description length ------------------------------------------


def test_certainty_costs_nothing():
    assert bits(1.0) == 0.0


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.5, 1.0), (0.25, 2.0), (0.125, 3.0), (1 / 1024, 10.0)],
)
def test_each_halving_costs_one_more_bit(p, expected):
    assert bits(p) == pytest.approx(expected)


def test_bits_grow_as_probability_falls():
    """The orientation that makes Viterbi a min-sum rather than a max-product.

    `HMMLIB-ACCOUNT.md` §3: smaller bits mean higher probability. A port that
    reaches for `max` inverts every comparison, so the direction is asserted
    rather than left to the reader.
    """
    p = np.array([0.9, 0.5, 0.1, 0.01])
    assert np.all(np.diff(bits(p)) > 0)


def test_impossible_costs_infinite_bits():
    """`+inf` replaces the original's `-1` sentinel (`util.lsh:434-437`)."""
    assert bits(0.0) == np.inf


def test_the_sentinel_is_not_representable_as_a_real_length():
    """Why `-1` was available to the original as a sentinel at all.

    A genuine description length is non-negative, so `-1` was unreachable and
    therefore unambiguous. `+inf` is unreachable for the same reason and needs
    no such argument -- but the non-negativity is worth pinning down, since it
    is what made both choices possible.
    """
    p = np.array([1.0, 0.9, 0.5, 1e-12, 0.0])
    assert np.all(bits(p) >= 0.0)


# --- what the sentinel bought, now bought by IEEE-754 -----------------------


def test_infinity_absorbs_under_addition_in_both_orders():
    """`safe-add--log2` was absorbing by an explicit `(= sum -1)` branch.

    Addition does it unaided, which is why :func:`bits` is unary where the
    original was binary.
    """
    assert bits(0.0) + 5.0 == np.inf
    assert 5.0 + bits(0.0) == np.inf
    assert bits(0.0) + bits(0.0) == np.inf


def test_absorption_holds_across_a_whole_accumulation():
    """Once impossible, impossible for the rest of the path."""
    total = 0.0
    for p in (0.5, 0.25, 0.0, 0.5, 0.5):
        total = total + bits(p)
    assert total == np.inf


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        # The original's own two call sites read `safe->--log(delta, cand)`,
        # meaning "y is strictly better than x", with the sentinel worst.
        (np.inf, 5.0, True),  # x impossible, y finite -> y is better
        (5.0, np.inf, False),  # y impossible -> nothing is better than x
        (np.inf, np.inf, False),  # both impossible
        (7.0, 5.0, True),  # fewer bits is better
        (5.0, 7.0, False),
        (5.0, 5.0, False),  # strict
    ],
)
def test_the_comparator_dissolved_into_plain_greater_than(x, y, expected):
    """`safe->--log` (`util.lsh:439-444`) has no counterpart, on purpose.

    The original spelled `(and (<> y -1) (or (= x -1) (> x y)))` because `-1`
    sorted numerically *below* every real description length while meaning
    "worse than all of them". `+inf` sorts where it means, so `>` is the whole
    function. This table is the original's truth table; if someone reinstates a
    sentinel, it fails.
    """
    assert (x > y) is expected


# --- warnings: the one suppression, and the one deliberately left alone -----


def test_log_zero_is_silent():
    """The `errstate` inside :func:`bits` is the reason it exists as a function."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bits(0.0) == np.inf
        assert np.all(np.isinf(bits(np.zeros(4))))


def test_a_negative_probability_still_complains():
    """`invalid` is not suppressed: a negative input is a bug, not a boundary."""
    with pytest.warns(RuntimeWarning, match="invalid value"):
        result = bits(-0.5)
    assert np.isnan(result)


# --- bits: array behaviour --------------------------------------------------


def test_bits_preserves_shape():
    p = np.array([[0.5, 1.0], [0.25, 0.0]])
    result = bits(p)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[1.0, 0.0], [2.0, np.inf]])


def test_bits_accepts_a_python_list():
    np.testing.assert_allclose(bits([0.5, 0.25]), [1.0, 2.0])


# --- safe_divide ------------------------------------------------------------


def test_ordinary_division_is_ordinary():
    assert safe_divide(1.0, 2.0) == pytest.approx(0.5)


def test_dividing_by_zero_yields_zero_not_infinity():
    """`util.lsh:47-48` returns `0.0`, where numpy alone would give `inf`."""
    assert safe_divide(1.0, 0.0) == 0.0


def test_zero_over_zero_yields_zero_not_nan():
    """The other half of the same branch, where numpy alone would give `nan`."""
    assert safe_divide(0.0, 0.0) == 0.0


def test_normalizing_an_array_by_a_scalar():
    """The shape of thirteen of the fifteen call sites."""
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        safe_divide(counts, counts.sum()), [0.1, 0.2, 0.3, 0.4]
    )


def test_normalizing_by_a_zero_scalar_zeroes_the_whole_array():
    counts = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(safe_divide(counts, 0.0), [0.0, 0.0, 0.0])


def test_elementwise_with_zeros_scattered_through_the_denominator():
    num = np.array([1.0, 2.0, 3.0, 0.0])
    den = np.array([2.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(safe_divide(num, den), [0.5, 0.0, 1.0, 0.0])


def test_safe_divide_broadcasts():
    num = np.ones((2, 3))
    den = np.array([2.0, 0.0, 4.0])
    np.testing.assert_allclose(
        safe_divide(num, den), [[0.5, 0.0, 0.25], [0.5, 0.0, 0.25]]
    )


def test_safe_divide_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        safe_divide(np.array([1.0, 0.0]), np.array([0.0, 0.0]))


def test_a_zero_denominator_never_produces_a_non_finite_value():
    """The property the zero exists to guarantee.

    Both `nan` and `inf` would propagate into a parameter array silently; a
    zero is at least visible in the array that carries it. Scoped to a zero
    denominator on purpose -- this function does nothing about overflow, and
    `1e300 / 1e-300` is still `inf`.
    """
    num = np.array([1.0, 0.0, -1.0])
    den = np.zeros(3)
    result = safe_divide(num, den)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, 0.0)


# --- stationary_distribution: the row-replacement solve ---------------------

# Read from `.scratch/` in place rather than copied here. Those files are
# *tracked*, so unlike `.notebooks/` and `.data/` they exist in every clone, and
# they were tracked for exactly this purpose. `tests/` never ships either --
# the wheel packages only `src/pfsmgraph` -- so "the fixture is absent in an
# installed wheel" is not a scenario this repo has.
_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / ".scratch"
    / "hmm-lush"
    / "Training"
    / "set02a"
    / "set02a_200"
)

_SAVED_MODELS = ("m001_0001_001.hmm", "m001_0005_005.hmm", "m008_0001_008.hmm")

# Two closed communicating classes, {0,1} and {2,3}, so the stationary space is
# two-dimensional and one replaced row cannot pin it down.
_REDUCIBLE = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


def _read_ascii_matrix(path):
    """Read Lush's `save-ascii-matrix` format: `.MAT <ndim> <dims...>`, then values.

    Written here rather than in a `conftest.py` for the reason `_numeric.py` is
    one module: there is one consumer. When revision 03's differential tests
    want it too, that is the moment a shared fixture earns itself.
    """
    tokens = path.read_text().split()
    if tokens[0] != ".MAT":
        raise ValueError(f"{path} is not a Lush ASCII matrix")
    ndim = int(tokens[1])
    dims = [int(t) for t in tokens[2 : 2 + ndim]]
    values = np.array([float(t) for t in tokens[2 + ndim :]], dtype=np.float64)
    return values.reshape(dims)


def _irreducible_chain(size=6, seed=20260903):
    rng = np.random.default_rng(seed)
    transition_p = rng.random((size, size)) + 0.1
    return transition_p / transition_p.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("model", _SAVED_MODELS)
def test_reproduces_the_originals_own_state_p(model):
    """A differential test against the Lush implementation's saved output.

    Each `.hmm` directory holds `transition_p` (the input) beside `state_p`
    (the output), so this compares our solve against the original's on numbers
    the original itself produced -- which is a stronger check than any chain we
    could construct, because it also catches a misreading of what `state_p`
    *is*.

    The tolerance is the saved format's, not slack, and it has two parts.
    `save-ascii-matrix` prints four decimals, so `state_p` carries up to 5e-5 of
    rounding; and `transition_p` was printed the same way, so its own rounding
    propagates through the solve as well (renormalising the rows first moves the
    8-state residual from 4.975e-5 to 4.784e-5, which is that contribution made
    visible). 5e-5 is therefore the wrong bound in two ways: it omits the second
    term, and a value can land *exactly* on it -- the 5-state model's true `pi_0`
    is 0.10135, printed as "0.1014", so the residual is 5.000000000000837e-05
    and an `abs=5e-5` comparison fails by less than an ulp. 1e-4 is one clean
    doubling above the observed worst case, and still three orders tighter than
    any genuine error in the solve could be.

    There is deliberately no skip when the fixtures are missing. They are
    tracked, so their absence means a broken checkout, not a configuration this
    repository supports.
    """
    directory = _FIXTURES / model
    transition_p = _read_ascii_matrix(directory / "transition_p")
    saved = _read_ascii_matrix(directory / "state_p")

    assert stationary_distribution(transition_p) == pytest.approx(saved, abs=1e-4)


@pytest.mark.parametrize("model", _SAVED_MODELS)
def test_the_homogeneous_system_needs_the_replacement(model):
    """Why the trick is the algorithm: without it there is nothing to solve.

    `(P.T - I)` is singular by construction -- that is what makes pi an
    eigenvector -- so its rank is short of full on every real transition matrix,
    including the original's own. Asserting it means a port that "simplifies"
    the row replacement away fails here rather than quietly returning zeros.

    The rows are renormalised first, and that is the point rather than a
    convenience. Row-stochasticity is the *hypothesis* of the claim being
    tested; the four-decimal print violates it by up to 1e-4 on the 8-state
    model, which lifts the smallest singular value of `(P.T - I)` from 6.6e-17
    to 1.2e-5 -- nine orders above `matrix_rank`'s 3.6e-15 tolerance, so it
    reports full rank. Testing the conclusion on data that breaks the hypothesis
    would test nothing at all.
    """
    saved = _read_ascii_matrix(_FIXTURES / model / "transition_p")
    transition_p = saved / saved.sum(axis=1, keepdims=True)
    size = transition_p.shape[0]

    assert np.linalg.matrix_rank(transition_p.T - np.eye(size)) < size


def test_a_two_state_chain_matches_the_closed_form():
    """For `P = [[1-a, a], [b, 1-b]]` the stationary distribution is `[b, a]/(a+b)`.

    The check the plan originally proposed, kept alongside the differential
    tests because it depends on nothing under `.scratch/`.
    """
    a, b = 0.25, 0.75
    transition_p = np.array([[1 - a, a], [b, 1 - b]])

    assert stationary_distribution(transition_p) == pytest.approx(
        np.array([b, a]) / (a + b)
    )


def test_the_result_is_a_fixed_point_of_the_transition_matrix():
    """The defining property, checked without reference to any expected value."""
    transition_p = _irreducible_chain()
    pi = stationary_distribution(transition_p)

    assert pi @ transition_p == pytest.approx(pi)


def test_the_result_is_a_distribution():
    pi = stationary_distribution(_irreducible_chain())

    assert pi.sum() == pytest.approx(1.0)
    assert (pi >= 0.0).all()


def test_a_doubly_stochastic_chain_is_uniform():
    transition_p = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )

    assert stationary_distribution(transition_p) == pytest.approx(np.full(3, 1 / 3))


def test_an_absorbing_state_takes_all_the_mass():
    transition_p = np.array([[0.5, 0.5], [0.0, 1.0]])

    assert stationary_distribution(transition_p) == pytest.approx([0.0, 1.0])


def test_the_single_state_chain_is_certain():
    """The degenerate case needs no special handling.

    `(P.T - I)` is `[[0.0]]`, which the row replacement turns into `[[1.0]]`
    against `b = [1.0]`.
    """
    assert stationary_distribution(np.array([[1.0]])) == pytest.approx([1.0])


def test_a_reducible_chain_is_refused():
    """The row replacement supplies one equation, so it rescues nullity 1 and no more.

    Revision 04 searches topology by state merge and split, which makes a
    disconnected component a plausible search outcome rather than a malformed
    input -- so the failure is named rather than left as "Singular matrix".
    """
    assert np.linalg.matrix_rank(_REDUCIBLE.T - np.eye(4)) == 2  # nullity 2

    with pytest.raises(ValueError, match="reducible"):
        stationary_distribution(_REDUCIBLE)


def test_the_reducible_error_keeps_the_numerical_cause():
    """`raise ... from err`, so the LinAlgError is still reachable for debugging."""
    with pytest.raises(ValueError) as excinfo:
        stationary_distribution(_REDUCIBLE)

    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)


def test_a_non_square_matrix_is_refused():
    """A structural precondition, not a check on the model's semantics.

    Without it this surfaces as a broadcasting error naming shapes the caller
    never wrote. Row-stochasticity is the other kind of check and is
    deliberately absent -- it belongs to `HMMParams` at construction (ADR 0017).
    """
    with pytest.raises(ValueError, match="square"):
        stationary_distribution(np.zeros((3, 5)))


def test_a_frozen_transition_matrix_is_accepted():
    """The ADR 0017 scenario: parameter arrays are held with `writeable = False`.

    `p.T` is then a read-only view, so the row replacement has to write into the
    fresh array `p.T - np.eye(size)` produces. This fails loudly if anyone
    "optimizes" that subtraction into an in-place assignment.
    """
    transition_p = np.array([[0.5, 0.5], [0.25, 0.75]])
    transition_p.setflags(write=False)
    before = transition_p.copy()

    pi = stationary_distribution(transition_p)

    assert pi @ transition_p == pytest.approx(pi)
    assert (transition_p == before).all()


# --- the package boundary ---------------------------------------------------


def test_the_numeric_helpers_are_private_to_the_package():
    """Migrated "private to the package" (master plan, revision 02).

    They are Utility-library internals, not part of what `pfsmgraph.hmm`
    offers a caller, and nothing outside this distribution may import them.

    Note what is deliberately *not* asserted. `pfsmgraph.hmm._numeric` becomes
    an attribute of the package the moment anything imports the submodule --
    including this test file -- because that is how Python binds a submodule on
    its parent. Asserting its absence looks like the stronger check and can
    never pass. The underscore in the module name is what carries the intent;
    what is actually checkable is that neither helper is reachable as a
    top-level name on the package.
    """
    for name in ("bits", "safe_divide", "stationary_distribution"):
        assert not hasattr(pfsmgraph.hmm, name)
    assert _numeric.__name__.rpartition(".")[2].startswith("_")
