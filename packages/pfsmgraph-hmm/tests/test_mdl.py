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

Nine sections:

- **Exact values**, where float arithmetic is exact and ``==`` is the right
  assertion rather than a tolerance.
- **The two shape decisions**: that ``int-code-length`` codes ``n + 1``, and
  that its loop runs while the term is positive rather than while it exceeds 1.
- **Independent agreement**, against ``math.comb`` and against ``lgamma``.
- **Monotonicity**, which is the property the search relies on and neither
  oracle above asserts.
- **The domain**, including the one guard reproduced from the original and the
  two that are not.
- **The data description length at precision `d`**, against the `data-dl` each
  tracked model's own training log recorded. Moved here from
  `test_baum_welch.py` with the functions it covers.
- **The model description length**, against the `model-dl` column of the same
  logs, with both of its contested choices pinned against the alternative
  that would otherwise pass unnoticed.
- **The total**, against the `_total_dl` each model stores -- a fourth oracle,
  at four decimals rather than the log's `%g`, and the only one that constrains
  both halves at once.
- **Choosing `d`**, against each model's stored `d`, with the original's local
  minimum and the optimizer's `1e100` sentinel both pinned as decisions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams
from pfsmgraph.hmm._mdl import (
    RISSANEN_C,
    _comb_code_length,
    _corpus_description_length,
    _data_description_length,
    _int_code_length,
    _model_description_length,
    _IMPOSSIBLE_TOTAL,
    _minimize,
    _minimize_int,
    _quantize,
    _suggest_d,
    _total_description_length,
)

from _lush_fixtures import FIXTURES, load_corpus_record, load_params, read_scalar

#: Shared with `test_baum_welch.py`, whose data-description-length section moved
#: here. Copied rather than extracted: every test module in this package defines
#: its own `_vocabulary`/`_random_params` with the signature it needs -- three do
#: already, and two of them disagree about the arguments -- so a shared helper
#: would be a refactor of four files rather than a move of one section.
SEED = 20260915

#: A quantized vector's sum, over at most a few dozen symbols.
ROW_TOL = 1e-12


def _vocabulary(size):
    return SymbolTable([f"s{i}" for i in range(size - USER_BASE)])


def _random_params(rng, size, n_user):
    output = np.zeros((size, size, USER_BASE + n_user))
    output[:, :, USER_BASE:] = rng.dirichlet(np.ones(n_user), size=(size, size))
    return HMMParams(
        rng.dirichlet(np.ones(size)),
        rng.dirichlet(np.ones(size), size=size),
        output,
        _vocabulary(USER_BASE + n_user),
    )


def _random_corpus(rng, n_user, lengths):
    return [
        SequenceRecord(rng.integers(USER_BASE, USER_BASE + n_user, length))
        for length in lengths
    ]

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


# === the data description length at precision d ================================
#
# Each tracked model directory stores the `d` it was saved at and a training log
# whose last row printed `data-dl` at that `d`, with `%g`, so to six significant
# figures. That is an oracle computed by the original from these very parameters.

#: Two sources, both measured. The log prints six significant figures, so half a
#: printed unit is 0.005 bits at these magnitudes. And the parameters were saved
#: with four decimals, so `x * d` near a half can round the other way from the
#: original's full-precision value: measured, `m001_0001_001` at `d = 29` lands
#: 0.009 bits below its log, the other two within the print. 0.02 covers both,
#: and every unrounded description length is more than a bit away (asserted).
LOGGED_DL_TOL = 0.02


def _logged(model):
    directory = FIXTURES / f"{model}.hmm"
    last = (directory / "_training_log").read_text().strip().splitlines()[-1].split()
    # A row ends `data-dl model-dl total-dl d flag`; what precedes varies with the
    # move the row records, so read from the end.
    return load_params(directory), float((directory / "d").read_text()), float(last[-5])


@pytest.mark.parametrize("model", ["m001_0001_001", "m001_0005_005", "m008_0001_008"])
def test_the_data_description_length_is_the_originals_logged_value(model):
    params, d, logged = _logged(model)
    records = [load_corpus_record()]

    got = _data_description_length(params, records, d)
    unrounded = _corpus_description_length(
        params.init_state_p, params.transition_p, params.output_p, records
    )

    assert got == pytest.approx(logged, abs=LOGGED_DL_TOL)
    # Rounding is what the oracle sees: without it every model misses by bits.
    assert abs(unrounded - logged) > 1.0


def test_parameters_already_on_the_grid_give_the_unrounded_length_bit_for_bit():
    # Every probability a multiple of 1/4 and every vector summing to 1 exactly,
    # so rounding at d = 4 returns the same arrays and the forward pass is shared.
    output = np.zeros((2, 2, USER_BASE + 2))
    output[:, :, USER_BASE:] = [[[0.25, 0.75], [0.5, 0.5]], [[1.0, 0.0], [0.75, 0.25]]]
    params = HMMParams(
        np.array([0.75, 0.25]),
        np.array([[0.5, 0.5], [0.25, 0.75]]),
        output,
        _vocabulary(USER_BASE + 2),
    )
    records = _random_corpus(np.random.default_rng(SEED), 2, (30, 0, 17))

    for array in (params.init_state_p, params.transition_p, params.output_p):
        assert np.array_equal(_quantize(array, 4), array)
    assert _data_description_length(params, records, 4) == _corpus_description_length(
        params.init_state_p, params.transition_p, params.output_p, records
    )


def test_rounding_is_round_half_up_then_an_ascending_renormalisation():
    # 0.125 * 4 = 0.5 exactly, which rounds up to 1/4; 0.0625 * 4 + 0.5 = 0.75,
    # which rounds down to 0. Values are binary fractions, so the grid is exact.
    p = np.array([0.125, 0.0625, 0.8125])
    rounded = np.array([0.25, 0.0, 0.75])
    total = 0.0
    for r in rounded:
        total += r
    assert _quantize(p, 4).tolist() == [r / total for r in rounded]


def test_a_vector_below_half_a_step_rounds_to_zeros_and_can_make_a_record_impossible():
    # Ten equal emissions of 0.1 at d = 3: 0.3 + 0.5 truncates to 0 everywhere.
    n_user = 10
    output = np.zeros((1, 1, USER_BASE + n_user))
    output[:, :, USER_BASE:] = 0.1
    params = HMMParams(np.ones(1), np.ones((1, 1)), output, _vocabulary(USER_BASE + n_user))

    with np.errstate(all="raise"):
        quantized = _quantize(params.output_p, 3)
        bits = _data_description_length(params, [SequenceRecord(np.array([USER_BASE]))], 3)

    assert not quantized.any()
    assert bits == np.inf
    # At d = 10 every emission is on the grid and the record costs log2(10) bits.
    assert _data_description_length(
        params, [SequenceRecord(np.array([USER_BASE]))], 10
    ) == pytest.approx(np.log2(10), abs=ROW_TOL)


@pytest.mark.parametrize("d", [0, -1.0, np.inf, np.nan])
def test_a_precision_that_is_not_positive_and_finite_is_rejected(d):
    params = _random_params(np.random.default_rng(SEED), 2, 3)
    with pytest.raises(ValueError, match="positive and finite"):
        _data_description_length(params, [SequenceRecord(np.array([USER_BASE]))], d)


def test_the_renormalisation_is_the_literal_ascending_loop_bit_for_bit():
    # Long vectors at a precision whose steps are not binary fractions, so the
    # order of the sum shows in the last bits. np.sum's pairwise blocks disagree.
    rng = np.random.default_rng(SEED + 3)
    p = rng.dirichlet(np.ones(64), size=(6, 6))
    d = 29.0
    got = _quantize(p, d)

    for index in np.ndindex(p.shape[:-1]):
        rounded = [float(np.floor(x * d + 0.5) / d) for x in p[index]]
        total = 0.0
        for r in rounded:
            total += r
        assert got[index].tolist() == [r / total if total else 0.0 for r in rounded]


# === the model description length ==============================================
#
# The same three logs, one column further along: `model-dl` at the model's own
# stored `d`. Unlike the data half this needs no forward pass, so the only way a
# saved parameter's four-decimal print can reach it is by moving a transition
# across the `1 / (2 d)` threshold and changing the non-zero count. It does not,
# for any of the three, so the tolerance is the log's print precision alone.

#: `%g` at six significant figures, so half a printed unit is 0.0005 at 439 bits
#: and 0.00005 at 58. Measured residuals: 0.00000, 0.00015, 0.00044. 0.001 leaves
#: headroom while staying far below the 12-to-105 bits the rejected variants miss
#: by, which is what makes it a real assertion rather than a wide net.
LOGGED_MODEL_DL_TOL = 0.001


def _logged_model_dl(model):
    directory = FIXTURES / f"{model}.hmm"
    last = (directory / "_training_log").read_text().strip().splitlines()[-1].split()
    # A row ends `data-dl model-dl total-dl d flag`; read from the end, as the
    # data-half helper above does, since what precedes varies with the move.
    return load_params(directory), float((directory / "d").read_text()), float(last[-4])


@pytest.mark.parametrize("model", ["m001_0001_001", "m001_0005_005", "m008_0001_008"])
def test_the_model_description_length_is_the_originals_logged_value(model):
    params, d, logged = _logged_model_dl(model)
    assert _model_description_length(params, d) == pytest.approx(
        logged, abs=LOGGED_MODEL_DL_TOL
    )


@pytest.mark.parametrize("model", ["m001_0001_001", "m001_0005_005", "m008_0001_008"])
def test_charging_for_the_reserved_block_misses_the_logged_value(model):
    """Pins the symbol-axis decision against the literal `output_p.shape[-1]`.

    `HMMParams` requires the six ADR 0011 reserved fibres to be exactly zero, so
    those codes carry no information; charging for them is an encoding artifact.
    The point of the test is that it is a *large* artifact, biased one way: it
    inflates the per-arc term only, so it grows with the transition count and
    pushes the search toward sparse topologies.
    """
    params, d, logged = _logged_model_dl(model)
    n_states = params.n_states
    whole_vocabulary = (
        _int_code_length(n_states)
        + _int_code_length(d)
        + (1 + n_states) * _comb_code_length(d, 1 + n_states)
        + int(np.count_nonzero(_quantize(params.transition_p, d)))
        * _comb_code_length(d, 1 + params.n_symbols)
    )
    assert whole_vocabulary > logged + 10.0
    assert whole_vocabulary > _model_description_length(params, d)


@pytest.mark.parametrize("model", ["m001_0001_001", "m001_0005_005", "m008_0001_008"])
def test_the_vector_code_takes_one_part_more_than_the_state_count(model):
    """Pins `comb_code_length(d, 1 + n_states)` against the arithmetically natural
    `comb_code_length(d, n_states)`.

    A vector of `n_states` probabilities quantized to `1 / d` is a composition of
    `d` into `n_states` parts, so `m = n_states` is what the combinatorics call
    for and `1 + n_states` is what the original wrote. The oracle settles it: the
    natural form misses every logged value. Reproduced rather than corrected --
    the master plan fixes the criterion as the original's two-part code, and
    PRD section 8 owns the question of whether it should be.
    """
    params, d, logged = _logged_model_dl(model)
    n_states = params.n_states
    natural = (
        _int_code_length(n_states)
        + _int_code_length(d)
        + (1 + n_states) * _comb_code_length(d, n_states)
        + int(np.count_nonzero(_quantize(params.transition_p, d)))
        * _comb_code_length(d, 1 + (params.n_symbols - USER_BASE))
    )
    assert natural < logged - 5.0


def _two_states_one_arc_below_the_grid():
    """A model whose 0.1 transition rounds to zero at `d = 4`, and 0.5 does not."""
    n_user = 2
    output = np.zeros((2, 2, USER_BASE + n_user))
    output[:, :, USER_BASE:] = 0.5
    return HMMParams(
        np.array([0.6, 0.4]),
        np.array([[0.9, 0.1], [0.5, 0.5]]),
        output,
        _vocabulary(USER_BASE + n_user),
    )


def test_transitions_are_counted_after_quantization_not_before():
    """The whole reason a sparse topology is cheaper to describe.

    All four transitions are non-zero as stored. At `d = 4` the 0.1 is below half
    a grid step and rounds away, so three arcs are paid for, not four. Counting
    before quantization would make `d` a cosmetic rounding parameter instead of
    the thing that prices the topology.
    """
    params = _two_states_one_arc_below_the_grid()
    assert np.count_nonzero(params.transition_p) == 4
    assert np.count_nonzero(_quantize(params.transition_p, 4.0)) == 3

    per_arc = _comb_code_length(4.0, 1 + (params.n_symbols - USER_BASE))
    got = _model_description_length(params, 4.0)
    fixed = (
        _int_code_length(2) + _int_code_length(4.0) + 3 * _comb_code_length(4.0, 3)
    )
    assert got == pytest.approx(fixed + 3 * per_arc, abs=1e-12)
    # And it is exactly one arc's worth below what counting before quantization
    # would charge -- stated against the naive count rather than against 3, so
    # the assertion is about the decision and not about arithmetic.
    naive = fixed + np.count_nonzero(params.transition_p) * per_arc
    assert naive - got == pytest.approx(per_arc, abs=1e-12)


def test_a_coarser_grid_can_only_remove_arcs_so_the_per_arc_cost_falls():
    """Monotone in `d` through the count, which is what makes `suggest-d` a search.

    Coarsening `d` cuts the per-arc term by zeroing transitions, and raises the
    two integer codes and the vector term. The criterion is the trade between
    them, so neither direction is monotone overall -- only the arc count is.
    """
    params = _two_states_one_arc_below_the_grid()
    counts = [
        np.count_nonzero(_quantize(params.transition_p, d))
        for d in (2.0, 4.0, 20.0, 1000.0)
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_more_states_costs_more_bits_at_a_fixed_grid():
    """The property the search leans on: a split must pay for itself."""
    rng = np.random.default_rng(SEED)
    lengths = [_model_description_length(_random_params(rng, size, 3), 1000.0)
               for size in (1, 2, 4, 8)]
    assert all(b > a for a, b in zip(lengths, lengths[1:]))


@pytest.mark.parametrize("d", [0, -1.0, np.inf, np.nan])
def test_the_model_half_rejects_a_precision_the_data_half_would(d):
    """Both halves guard `d` the same way, since the total calls them with one value."""
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="d must be positive and finite"):
        _model_description_length(_random_params(rng, 3, 3), d)


# === the total ==================================================================
#
# `_total_dl` is a file in each saved model directory, written at four decimals
# where the training log prints `%g`. It is the only oracle here that constrains
# the two halves jointly: either could be wrong in a way its own test tolerates
# and still sum correctly, but not while both sums land within a hundredth of a
# bit on three models of different sizes at three different `d`.

#: The data half's own tolerance, inherited: the saved parameters are four-decimal
#: prints, so `x * d` near a half can round the other way from the original's
#: full-precision value. Measured residuals here are -0.0131, +0.0007 and -0.0025,
#: the first being the known `d = 29` offset rather than anything new.
STORED_TOTAL_DL_TOL = 0.02


@pytest.mark.parametrize("model", ["m001_0001_001", "m001_0005_005", "m008_0001_008"])
def test_the_total_is_the_models_own_stored_total_dl(model):
    directory = FIXTURES / f"{model}.hmm"
    params = load_params(directory)
    d = read_scalar(directory / "d")
    stored = read_scalar(directory / "_total_dl")
    records = [load_corpus_record()]
    assert _total_description_length(params, records, d) == pytest.approx(
        stored, abs=STORED_TOTAL_DL_TOL
    )


@pytest.mark.parametrize("model", ["m001_0001_001", "m008_0001_008"])
def test_the_total_adds_the_two_halves_and_nothing_else(model):
    """Bit for bit, so no stray term can hide inside the tolerance above.

    The original's forward pass ends by adding `bits` of the last column's sum,
    which is 1 within rounding; `baum_welch` omits it and so does the data half,
    and this pins that the total does not quietly reintroduce it or anything like
    it. A term worth a bit would pass `STORED_TOTAL_DL_TOL` unnoticed.
    """
    directory = FIXTURES / f"{model}.hmm"
    params = load_params(directory)
    d = read_scalar(directory / "d")
    records = [load_corpus_record()]
    assert _total_description_length(params, records, d) == (
        _data_description_length(params, records, d)
        + _model_description_length(params, d)
    )


def test_the_total_is_a_plain_float_the_search_can_compare():
    """Pins the return type, which is a decision rather than a convenience.

    PRD section 8 leaves open whether this project should end up with a refined
    one-part code, which has no data/model split. A return value carrying those
    two fields would assert the two-part structure in the seam that exists to
    make the swap cheap, so the criterion is a bare number and the training log
    calls the halves itself.
    """
    rng = np.random.default_rng(SEED)
    params = _random_params(rng, 3, 3)
    got = _total_description_length(params, _random_corpus(rng, 3, (20,)), 1000.0)
    assert type(got) is float
    assert got < float("inf")


def test_an_impossible_model_scores_inf_rather_than_a_comparable_sentinel():
    """The `1e100` mapping dissolves, and the difference is not cosmetic.

    `update-total-dl` substitutes `1e100` for its `-1` log-zero sentinel so an
    impossible model sorts last. Under `+inf` that happens by arithmetic instead,
    and -- the part a sentinel gets wrong -- every impossible model scores the
    *same*, where `1e100 + model_dl` would rank them by the cost of describing a
    model that cannot generate the data at all.
    """
    n_user = 10
    output = np.zeros((1, 1, USER_BASE + n_user))
    output[:, :, USER_BASE:] = 0.1
    params = HMMParams(np.ones(1), np.ones((1, 1)), output, _vocabulary(USER_BASE + n_user))
    records = [SequenceRecord(np.array([USER_BASE]))]

    impossible = _total_description_length(params, records, 3)
    assert impossible == np.inf
    # It absorbs, so a second impossible model is not ranked against the first by
    # its model cost -- which is exactly what a large finite sentinel would do.
    assert impossible == _total_description_length(params, records + records, 3)
    assert impossible > _total_description_length(params, records, 10)


@pytest.mark.parametrize("d", [0, -1.0, np.inf, np.nan])
def test_the_total_rejects_a_precision_either_half_would(d):
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="d must be positive and finite"):
        _total_description_length(_random_params(rng, 3, 3), _random_corpus(rng, 3, (20,)), d)


# === choosing d ==================================================================
#
# `suggest-d` is Brent's method over the total, so the stored `d` is the outcome of
# a search rather than of a formula. Reproducing it exactly means reproducing the
# probe sequence, which is why the optimizer is transliterated statement for
# statement and why these tests assert `==` on the answer.

MODELS = ["m001_0001_001", "m001_0005_005", "m008_0001_008"]


@pytest.mark.parametrize("model", MODELS)
def test_suggest_d_chooses_the_d_the_original_stored(model):
    directory = FIXTURES / f"{model}.hmm"
    got = _suggest_d(load_params(directory), [load_corpus_record()])
    assert got == read_scalar(directory / "d")
    assert type(got) is float and got.is_integer()


def test_suggest_d_keeps_the_originals_local_minimum_on_the_one_state_model():
    """Pins a decision: the port reproduces where the original chose badly.

    The data half is not monotone in `d`, so the total has several basins and Brent
    keeps the one it slides into from 3821. On the one-state model -- where every
    search starts -- `d = 13` is more than ten bits cheaper than the stored 29.
    Replacing Brent with a global scan would make this test fail, which is the
    point: that is a search-loop design decision, and it should arrive as one.
    """
    directory = FIXTURES / "m001_0001_001.hmm"
    params, records = load_params(directory), [load_corpus_record()]
    assert _suggest_d(params, records) == 29.0
    assert (
        _total_description_length(params, records, 13.0)
        < _total_description_length(params, records, 29.0) - 10.0
    )


def _rare_symbol_model():
    """One state whose first symbol has probability 1e-4.

    `floor(1e-4 * d + 0.5)` is zero below `d = 5000`, so a record containing that
    symbol is impossible there -- including at the search's starting probe, 3821.
    """
    n_user = 3
    output = np.zeros((1, 1, USER_BASE + n_user))
    output[0, 0, USER_BASE:] = [1e-4, 0.5 - 5e-5, 0.5 - 5e-5]
    params = HMMParams(np.ones(1), np.ones((1, 1)), output, _vocabulary(USER_BASE + n_user))
    records = [SequenceRecord(np.array([USER_BASE, USER_BASE + 1, USER_BASE + 2]))]
    return params, records


def test_suggest_d_scores_an_impossible_d_with_the_originals_sentinel_not_inf():
    """`_total_description_length` returns `inf`; the optimizer must not see it.

    Brent's parabolic fit subtracts function values, and `inf - inf` is `nan`, which
    fails the step-size guard and collapses the step to `-tol1`. So the same search
    over raw `inf` takes a different probe path from the original's. Asserted by
    running both and requiring they differ, which is what shows the sentinel is in
    use rather than merely defined.
    """
    params, records = _rare_symbol_model()
    assert _total_description_length(params, records, 3821.0) == np.inf

    chosen = _suggest_d(params, records)
    assert np.isfinite(_total_description_length(params, records, chosen))

    raw, _ = _minimize_int(
        lambda d: _total_description_length(params, records, d), 1.0, 3821.0, 10000.0
    )
    assert raw != chosen


def test_raw_inf_can_drive_brent_to_an_impossible_point_where_the_sentinel_does_not():
    """The failure the sentinel exists for, on an objective that exhibits it plainly.

    Impossible below 9000, a parabola above. From 3821 every early probe is `inf`,
    and with raw `inf` the search creeps to an impossible answer.
    """

    def raw(x):
        return np.inf if x < 9000.0 else (x - 9005.0) ** 2

    def sentinel(x):
        return _IMPOSSIBLE_TOTAL if x < 9000.0 else (x - 9005.0) ** 2

    _, f_raw = _minimize(raw, 1.0, 3821.0, 10000.0, 1e-2)
    x_sentinel, f_sentinel = _minimize(sentinel, 1.0, 3821.0, 10000.0, 1e-2)
    assert f_raw == np.inf
    assert x_sentinel >= 9000.0 and f_sentinel < 100.0


def test_a_tie_between_the_two_integers_goes_to_the_larger():
    """`util.lsh:113` tests `f(floor) < f(floor + 1)` strictly, so a tie keeps the upper.

    The objective is flat across a wide floor, so the two integers tie wherever
    Brent's real answer lands. That matters: its tolerance is *relative*, `1e-2 *
    |x|`, so near 5000 it promises only about fifty units, and a test that assumed
    it landed at a chosen half-integer would be testing luck rather than the rule.
    """

    def flat_bottomed(x):
        return max(0.0, abs(x - 5000.0) - 200.0)

    real_x, _ = _minimize(flat_bottomed, 1.0, 3821.0, 10000.0, 1e-2)
    lower = float(np.floor(real_x))
    assert flat_bottomed(lower) == flat_bottomed(lower + 1.0) == 0.0

    chosen, f_chosen = _minimize_int(flat_bottomed, 1.0, 3821.0, 10000.0)
    assert chosen == lower + 1.0
    assert f_chosen == 0.0


def test_brent_finds_the_minimum_of_a_single_basin_to_its_tolerance():
    """The ordinary case, so a transliteration slip cannot hide behind the fixtures."""
    x, fx = _minimize(lambda x: (x - 4242.4) ** 2 + 7.0, 1.0, 3821.0, 10000.0, 1e-2)
    assert abs(x - 4242.4) < 1e-2 * 4242.4
    assert fx == pytest.approx(7.0, abs=1e-6 * 4242.4**2)
