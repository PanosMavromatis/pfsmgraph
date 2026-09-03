"""The log-domain primitives, and the two functions that dissolved into numpy.

Several tests here assert the *absence* of the original's machinery rather than
the presence of ours. That is deliberate: the `-1` log-zero sentinel was
replaced by `+inf` on the grounds that IEEE-754 reproduces every property it
was built for, and the way to keep that claim honest is to test the properties
themselves, so that reinstating a sentinel breaks something.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pfsmgraph.hmm
from pfsmgraph.hmm import _numeric
from pfsmgraph.hmm._numeric import bits, safe_divide


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
    for name in ("bits", "safe_divide"):
        assert not hasattr(pfsmgraph.hmm, name)
    assert _numeric.__name__.rpartition(".")[2].startswith("_")
