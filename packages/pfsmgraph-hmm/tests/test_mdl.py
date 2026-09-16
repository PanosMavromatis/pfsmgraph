"""The two code-length primitives, against exact values and an independent form.

**Every test here reaches a private function**, labelled rather than hidden as
ADR 0003 asks: ``_mdl`` is private to ``pfsmgraph.hmm`` and nothing in it is
exported, so there is no public call to reach it through.

The oracle is not the Lush original, because there is no Lush runtime in this
repository to run it. It is arithmetic instead, in three independent forms:
powers of two, where the iterated logarithm is exact and the expected value can
be written down; :func:`math.comb`, which evaluates the binomial the
implementation deliberately never forms; and a ``lgamma`` closed form, which
computes the same quantity by a route sharing no code with the loop under test.

Five sections:

- **Exact values**, where float arithmetic is exact and ``==`` is the right
  assertion rather than a tolerance.
- **The two shape decisions**: that ``int-code-length`` codes ``n + 1``, and
  that its loop runs while the term is positive rather than while it exceeds 1.
- **Independent agreement**, against ``math.comb`` and against ``lgamma``.
- **Monotonicity**, which is the property the search relies on and neither
  oracle above asserts.
- **The domain**, including the one guard reproduced from the original and the
  two that are not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfsmgraph.hmm._mdl import RISSANEN_C, _comb_code_length, _int_code_length

# Through numpy, as the module does. The two agree on this input, but pinning
# exact values against a different log2 implementation would be luck.
LOG2_C = float(np.log2(RISSANEN_C))
LOG2E = 1.0 / math.log(2.0)


def _comb_via_lgamma(total: float, m: int) -> float:
    """``log2(total + m) + log2 C(total + m - 1, m - 1)`` through ``lgamma``.

    Shares no code with the implementation: no loop, no elementwise difference,
    and the binomial evaluated by the gamma function rather than by a sum of
    logarithms.
    """
    binomial = (
        math.lgamma(total + m) - math.lgamma(total + 1.0) - math.lgamma(m)
    ) * LOG2E
    return math.log2(total + m) + binomial


# --------------------------------------------------------------------------
# Exact values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "iterated"),
    [
        (0, 0.0),  # codes 1: the loop never runs
        (1, 1.0),  # codes 2: log2 2 = 1, then 1 is not > 1
        (3, 3.0),  # codes 4: 2, then 1
        (15, 7.0),  # codes 16: 4, 2, 1
        (65535, 23.0),  # codes 2**16: 16 + 4 + 2 + 1, since log2 16 is 4
    ],
)
def test_the_integer_code_is_exact_at_iterated_powers_of_two(n, iterated):
    """At these ``n`` every logarithm is an integer, so the sum is exact."""
    assert _int_code_length(n) == LOG2_C + iterated


def test_the_composition_code_of_nothing_is_the_cost_of_naming_the_parts():
    """``total = 0`` kills every term, leaving ``log2(m)`` exactly.

    Each term is ``log2(0 + i) - log2(i)``, which is exactly zero in floating
    point for every ``i`` -- the same value is subtracted from itself, not two
    values that happen to be close.
    """
    for m in (2, 3, 7, 32, 51):
        assert _comb_code_length(0.0, m) == math.log2(m)


def test_two_parts_is_the_pair_of_logarithms_with_no_loop_left():
    """``m = 2`` leaves a single term whose ``log2(1)`` vanishes."""
    for total in (1.0, 10.0, 10000.0):
        expected = math.log2(total + 2.0) + math.log2(total + 1.0)
        assert _comb_code_length(total, 2) == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# The two shape decisions
# --------------------------------------------------------------------------


def test_the_integer_code_codes_n_plus_one():
    """``util.lsh:467`` increments before the loop, so ``0`` is representable.

    Without the increment ``_int_code_length(0)`` would take ``log2(0)``. The
    model description length calls this on a state count and the search starts
    from one state, so the small arguments are the ones that occur.
    """
    assert _int_code_length(0) == LOG2_C
    assert math.isfinite(_int_code_length(0))
    # 1 codes 2, whose logarithm is exactly 1, so the whole of n = 1 is the
    # constant plus one bit. Asserted as the sum rather than as the difference
    # from n = 0: (LOG2_C + 1.0) - LOG2_C is 0.9999999999999998, which is the
    # subtraction losing a bit, not the code being wrong.
    assert _int_code_length(1) == LOG2_C + 1.0


def test_the_last_term_added_is_below_one_and_above_zero():
    """The loop guards the value, not the term, so the final term is < 1.

    ``HMMLIB-ACCOUNT.md`` §8 describes the rule as "while the term exceeds 1",
    which would stop one term earlier. This pins the generated C's rule
    (``util.c:645-661``) instead: a term is added exactly when the value it came
    from exceeded 1, which is exactly when the term itself is above zero.
    """
    # n = 4 codes 5: log2 5 = 2.32..., then log2 2.32... = 1.21..., then
    # log2 1.21... = 0.28..., which is added although it is below 1.
    n = 5.0
    terms = []
    while n > 1.0:
        n = math.log2(n)
        terms.append(n)
    assert terms[-1] < 1.0
    assert terms[-1] > 0.0
    assert _int_code_length(4) == pytest.approx(LOG2_C + sum(terms), abs=1e-15)


# --------------------------------------------------------------------------
# Independent agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("total", [0, 1, 5, 10, 100])
@pytest.mark.parametrize("m", [2, 3, 5, 13])
def test_the_composition_code_is_the_binomial_it_never_forms(total, m):
    """Against :func:`math.comb`, which forms the integer exactly.

    Only small arguments: this is the range where the binomial is representable
    at all, which is the point -- the implementation is checked where the naive
    form still works, and the ``lgamma`` test below covers where it does not.
    """
    expected = math.log2(total + m) + math.log2(math.comb(total + m - 1, m - 1))
    assert _comb_code_length(total, m) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("total", [0.5, 10.0, 1000.0, 10000.0])
@pytest.mark.parametrize("m", [2, 6, 9, 13, 32, 51])
def test_the_composition_code_agrees_with_a_closed_form(total, m):
    """Against ``lgamma``, at the real arguments the criterion uses.

    ``d = 10000`` is the original's default and ``m = 1 + n_states`` or
    ``1 + n_symbols``, so these are the values a score is actually built from.
    At ``d = 10000, m = 51`` the binomial has over 400 digits and
    :func:`math.comb` on floats is not available at all.
    """
    assert _comb_code_length(total, m) == pytest.approx(
        _comb_via_lgamma(total, m), rel=1e-10
    )


def test_the_composition_code_is_not_claimed_bit_identical_to_the_original():
    """The vectorised difference and the original's interleaving are close, not equal.

    The docstring claims agreement to 1e-13 bits rather than bit-identity, and
    that is a claim worth holding to: a future rewrite that reached for
    :func:`numpy.sum` would drift further without any test noticing.
    """
    total, m = 10000.0, 51
    interleaved = math.log2(total + m)
    for i in range(1, m):
        interleaved += math.log2(total + i)
        interleaved -= math.log2(i)
    assert _comb_code_length(total, m) == pytest.approx(interleaved, abs=1e-13)


# --------------------------------------------------------------------------
# Monotonicity
# --------------------------------------------------------------------------


def test_both_codes_grow_with_what_they_describe():
    """More states, or a finer grid, costs more bits.

    Neither oracle above asserts this, and it is the property the search leans
    on: a move that adds a state must pay for it, or the criterion cannot
    prefer a smaller model.
    """
    integer_codes = [_int_code_length(n) for n in range(0, 200)]
    assert all(b >= a for a, b in zip(integer_codes, integer_codes[1:]))

    by_parts = [_comb_code_length(10000.0, m) for m in range(2, 60)]
    assert all(b > a for a, b in zip(by_parts, by_parts[1:]))

    by_resolution = [_comb_code_length(d, 9) for d in (10.0, 100.0, 1e3, 1e4)]
    assert all(b > a for a, b in zip(by_resolution, by_resolution[1:]))


# --------------------------------------------------------------------------
# The domain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 0, -3])
def test_one_part_or_fewer_costs_nothing(m):
    """The one guard reproduced from the original, because it is not a guard.

    A composition into a single part is forced, so it carries no information.
    The search starts from a one-state model, so this is reached in normal
    operation rather than only by a caller error.
    """
    assert _comb_code_length(10000.0, m) == 0.0


def test_a_negative_size_raises_rather_than_scoring_zero():
    """Where the original returned 0.0 -- see the module docstring.

    Zero is the *cheapest* possible code, so the original's guard makes an
    impossible model the best-scoring one. That is the failure the ``1e100``
    sentinel was introduced to paper over elsewhere in the original, and this
    revision declined to port the sentinel.
    """
    with pytest.raises(ValueError, match="must be non-negative"):
        _int_code_length(-1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        _comb_code_length(-1.0, 5)


def test_a_nan_size_raises_rather_than_propagating():
    """``not (n >= 0)`` catches ``nan``, where ``n < 0`` would not.

    A ``nan`` reaching a description length makes every comparison in the search
    false, so a move is neither accepted nor rejected on its merits.
    """
    with pytest.raises(ValueError, match="must be non-negative"):
        _int_code_length(np.nan)
    with pytest.raises(ValueError, match="must be non-negative"):
        _comb_code_length(np.nan, 5)
