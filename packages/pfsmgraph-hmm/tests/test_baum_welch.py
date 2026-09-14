"""The M-step, checked against exact rational division and constructed zero counts.

**Every test here reaches a private function**, labelled rather than hidden as
ADR 0003 asks: there is no public trainer yet, and what a caller eventually sees
is decided by later subgoals.

The count arrays come from the real E-step over random models and over the
tracked 8-state fixture, not from invented numbers. The M-step's contract is
about *consistent* counts, where the emission counts summed over symbols are the
transition counts, and a synthetic array would not have that property unless the
test reimplemented the E-step to give it one.

Six sections:

- **The oracle**: each division redone in exact rational arithmetic over the very
  floats the M-step read, so the only disagreement allowed is the float64
  rounding of a sum of `S` terms and one division.
- **Stochasticity and structure**: live rows and live fibres sum to 1, dead arcs
  stay dead, reserved fibres stay zero, and the fixture's re-estimate is a valid
  `HMMParams`.
- **Constructed zero counts** the fixtures never exhibit: a state reachable only
  by the final symbol, and a record with no counts at all.
- **ADR 0020's order**, pinned bit for bit against literal ascending loops.
- **The EM loop**: the description length never rises, a fixed point stays
  fixed bit for bit, the stopping rule is `run-converge`'s read back from the
  trace, and a degenerate state tracks the original's zero-row likelihood.
- **The data description length at precision `d`**, against the `data-dl` each
  tracked model's own training log recorded.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, ImpossibleSequenceError
from pfsmgraph.hmm._baum_welch import (
    BATCH_CYCLES,
    CHANGE_BITS,
    PATIENCE,
    _corpus_description_length,
    _data_description_length,
    _em,
    _m_step,
    _quantize,
)
from pfsmgraph.hmm._forward_backward import (
    _description_length,
    _expected_counts,
    _forward_backward,
)

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


# === the EM loop ================================================================
#
# Over records rather than raw arrays, since the loop is where records, the
# vocabulary and HMMParams validation meet. The properties are Baum's: the
# description length never rises between cycles, and a fixed point stays fixed.

#: Baum's inequality is exact in the mathematics, and a cycle's description
#: length is a float64 sum over records and positions. Measured over 20 random
#: multi-record corpora and 60 cycles each, the largest rise was exactly 0; this
#: allows a few ulps per summed term at the sizes used here and would still
#: expose a mis-normalised M-step, which rises by whole bits.
RISE_RTOL = 1e-12


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


@pytest.mark.parametrize(
    "size, n_user, lengths",
    [(1, 3, (20,)), (3, 4, (15, 0, 40)), (5, 5, (60, 7, 33, 1)), (8, 6, (200,))],
    ids=["S1", "S3-with-empty", "S5-four-records", "S8"],
)
def test_the_description_length_never_rises_between_cycles(size, n_user, lengths):
    rng = np.random.default_rng(SEED + size)
    params = _random_params(rng, size, n_user)
    result = _em(params, _random_corpus(rng, n_user, lengths), max_cycles=40)

    history = np.array(result.description_lengths)
    assert len(history) == result.cycles + 1 == 41
    assert (np.diff(history) <= RISE_RTOL * history[0]).all()
    assert history[-1] < history[0]


def test_a_model_at_a_fixed_point_stays_there_bit_for_bit():
    # Two symmetric states whose every fibre is the corpus's symbol frequency,
    # 1/2 each, and whose transitions and seed are uniform. Every count is then
    # the same for both states, so re-estimation reproduces each parameter, and
    # every value involved is a power of two, so it does so exactly.
    output = np.zeros((2, 2, USER_BASE + 2))
    output[:, :, USER_BASE:] = 0.5
    params = HMMParams(
        np.full(2, 0.5), np.full((2, 2), 0.5), output, _vocabulary(USER_BASE + 2)
    )
    a, b = USER_BASE, USER_BASE + 1
    records = [SequenceRecord(np.array([a, b, b, a])), SequenceRecord(np.array([b, a]))]

    result = _em(params, records)

    assert np.array_equal(result.params.init_state_p, params.init_state_p)
    assert np.array_equal(result.params.transition_p, params.transition_p)
    assert np.array_equal(result.params.output_p, params.output_p)
    assert set(result.description_lengths) == {6.0}
    # No batch changes at all, so the first PATIENCE batches are the whole run.
    assert result.converged and result.cycles == PATIENCE * BATCH_CYCLES


def test_the_stopping_rule_is_run_converge():
    # A small threshold so the run takes several batches. Read back from the
    # returned trace: the last PATIENCE batch changes are below it, the one
    # before them (if any) is not, and no earlier run of PATIENCE exists.
    rng = np.random.default_rng(SEED + 99)
    params = _random_params(rng, 4, 4)
    batch, threshold, patience = 5, 1e-3, 3
    result = _em(
        params,
        _random_corpus(rng, 4, (80, 50)),
        batch_cycles=batch,
        change_bits=threshold,
        patience=patience,
    )

    assert result.converged and result.cycles % batch == 0
    ends = np.array(result.description_lengths[::batch])
    unchanged = np.abs(np.diff(ends)) < threshold
    assert unchanged[-patience:].all()
    run = 0
    for flag in unchanged[:-1]:
        run = run + 1 if flag else 0
        assert run < patience
    assert len(unchanged) == patience or not unchanged[-patience - 1]


def _near_saddle(eps):
    """Two states a perturbation `eps` away from exact symmetry, over `abab...`.

    The symmetric model is a fixed point of EM, a saddle, and the corpus is
    deterministic, so the optimum is 0 bits. The smaller `eps`, the longer the
    loop crawls near the saddle before it escapes.
    """
    a, b = USER_BASE, USER_BASE + 1
    transition = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
    output = np.zeros((2, 2, USER_BASE + 2))
    output[:, :, USER_BASE:] = 0.5
    output[0, :, a] += eps
    output[0, :, b] -= eps
    params = HMMParams(np.full(2, 0.5), transition, output, _vocabulary(USER_BASE + 2))
    return params, [SequenceRecord(np.array([a, b] * 200))]


def test_a_changed_batch_resets_the_unchanged_count():
    # Measured batch flags (1 = moved less than the threshold): 11, then seven
    # 0s while EM escapes the saddle, then 111. Without the reset the two early
    # 1s would carry over and the run would stop four cycles early, mid-escape.
    params, records = _near_saddle(1e-4)
    result = _em(params, records, batch_cycles=2, change_bits=1e-3, patience=3)

    ends = np.array(result.description_lengths[::2])
    flags = "".join("1" if f else "0" for f in np.abs(np.diff(ends)) < 1e-3)
    assert flags == "110000000111"
    assert result.cycles == 24 and result.description_lengths[-1] == 0.0


def test_run_converge_can_stop_on_the_saddle_itself():
    # Not a defect of the port but of the rule, and pinned so it is not mistaken
    # for one: close enough to symmetry, PATIENCE early batches all move by less
    # than the threshold and the loop stops about 400 bits above the optimum.
    # An exactly uniform start, which rand_p_vector(noise_width=0) returns, is
    # this case with eps = 0.
    params, records = _near_saddle(1e-5)
    result = _em(params, records, batch_cycles=2, change_bits=1e-3, patience=4)

    assert result.converged and result.cycles == 8
    assert result.description_lengths[-1] > 399


def test_the_lush_trained_fixture_is_already_near_its_fixed_point():
    # m008_0001_008 is the original's converged output, trained on this very
    # stream, so the loop should leave it almost where it is. Passing the whole
    # corpus as one record is the original's flat stream.
    params = load_params(FIXTURES / "m008_0001_008.hmm")
    result = _em(params, [load_corpus_record()])

    first, last = result.description_lengths[0], result.description_lengths[-1]
    assert result.converged and result.cycles == PATIENCE * BATCH_CYCLES
    assert 0 <= first - last < CHANGE_BITS
    assert result.degenerate_states == ()


def test_one_cycle_over_several_records_is_the_m_step_of_their_summed_counts():
    rng = np.random.default_rng(SEED + 7)
    params = _random_params(rng, 3, 4)
    records = _random_corpus(rng, 4, (25, 9, 31))

    result = _em(params, records, max_cycles=1)

    summed = [np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3, USER_BASE + 4))]
    for record in records:
        for total, part in zip(
            summed,
            _e_step(params.init_state_p, params.transition_p, params.output_p, record.codes),
        ):
            total += part
    want = _m_step(*summed)
    assert np.array_equal(result.params.init_state_p, want[0])
    assert np.array_equal(result.params.transition_p, want[1])
    assert np.array_equal(result.params.output_p, want[2])


def test_a_degenerate_state_keeps_its_row_and_tracks_the_zero_row_likelihood():
    # Goal 1's measured claim, pinned: restoring the old row and fibres leaves the
    # description length bit-identical to letting the row go to zero.
    init, transition, output = _sink_at_the_end_model()
    params = HMMParams(init, transition, output, _vocabulary(9))
    codes = np.array([A, A, B, A, B])

    result = _em(params, [SequenceRecord(codes)], max_cycles=5)

    assert result.degenerate_states == (2,)
    assert np.array_equal(result.params.transition_p[2], transition[2])
    assert np.array_equal(result.params.output_p[2], output[2])

    zero_row = []
    arrays = (init, transition, output)
    for _ in range(6):
        alpha, beta, scale = _forward_backward(*arrays, codes)
        zero_row.append(_description_length(scale))
        arrays = _m_step(*_expected_counts(alpha, beta, *arrays[1:], codes))[:3]
    assert list(result.description_lengths) == zero_row


def test_an_impossible_record_raises_before_training():
    init, transition, output = _sink_at_the_end_model()
    params = HMMParams(init, transition, output, _vocabulary(9))
    records = [SequenceRecord(np.array([A, B])), SequenceRecord(np.array([C, C]))]
    with pytest.raises(ImpossibleSequenceError, match="record 1"):
        _em(params, records)


@pytest.mark.parametrize(
    "records, kwargs, message",
    [
        ([np.array([USER_BASE, 99])], {}, "outside the model's symbol axis"),
        ([np.array([], dtype=np.int64)], {}, "no symbols"),
        ([], {}, "no symbols"),
        ([np.array([USER_BASE])], {"change_bits": 0.0}, "must be positive"),
        ([np.array([USER_BASE])], {"batch_cycles": 0}, "at least 1"),
        ([np.array([USER_BASE])], {"max_cycles": -1}, "non-negative"),
    ],
    ids=["code-out-of-range", "only-empty", "no-records", "zero-threshold", "zero-batch", "negative-budget"],
)
def test_the_loop_rejects_what_it_cannot_train_on(records, kwargs, message):
    rng = np.random.default_rng(SEED)
    params = _random_params(rng, 2, 3)
    with pytest.raises(ValueError, match=message):
        _em(params, [SequenceRecord(codes) for codes in records], **kwargs)


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
