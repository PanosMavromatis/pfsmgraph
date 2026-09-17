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
- **The progress log**: its rows are the returned trace's check values, its
  columns stay aligned at extreme values, a budget stop says so, and every
  line is flushed as written.
"""

from __future__ import annotations

import io
import re
from fractions import Fraction

import numpy as np
import pytest

from pfsmgraph.dataseq import PAD, UNK, USER_BASE, SequenceRecord, SymbolTable, pad_collate
from pfsmgraph.hmm import HMMParams, ImpossibleSequenceError, baum_welch
from pfsmgraph.hmm._baum_welch import (
    BATCH_CYCLES,
    CHANGE_BITS,
    PATIENCE,
    _log_row,
    _log_start,
    _m_step,
)
from pfsmgraph.hmm._forward_backward import (
    _description_length,
    _e_step_batch,
    _expected_counts,
    _forward_backward,
)
from pfsmgraph.hmm._forward_backward import _e_step as _record_step

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
    result = baum_welch(params, _random_corpus(rng, n_user, lengths), max_cycles=40)

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

    result = baum_welch(params, records)

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
    result = baum_welch(
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
    result = baum_welch(params, records, batch_cycles=2, change_bits=1e-3, patience=3)

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
    result = baum_welch(params, records, batch_cycles=2, change_bits=1e-3, patience=4)

    assert result.converged and result.cycles == 8
    assert result.description_lengths[-1] > 399


def test_a_minimum_budget_carries_the_saddle_run_past_its_early_stop():
    # The same run with the rule held off for 10 cycles. Measured with the rule
    # disabled, its batch flags are 1111, then seven 0s while EM escapes, then
    # 1s: the check at cycle 10 lands on a changed batch, which resets the count,
    # and four quiet batches after the escape stop it at 30. This is ADR 0022
    # section 4's failure and its remedy on one fixture.
    params, records = _near_saddle(1e-5)
    result = baum_welch(
        params, records, batch_cycles=2, change_bits=1e-3, patience=4, min_cycles=10
    )

    assert result.converged and result.cycles == 30
    assert result.description_lengths[-1] == 0.0


def test_no_minimum_budget_is_the_original_rule_bit_for_bit():
    rng = np.random.default_rng(SEED + 43)
    params, records = _random_params(rng, 3, 4), _random_corpus(rng, 4, (70, 30))
    plain = baum_welch(params, records, change_bits=1e-3)
    floored = baum_welch(params, records, change_bits=1e-3, min_cycles=0)

    assert floored.description_lengths == plain.description_lengths
    assert floored.cycles == plain.cycles and floored.converged == plain.converged
    for name in ("init_state_p", "transition_p", "output_p"):
        assert getattr(floored.params, name).tobytes() == getattr(plain.params, name).tobytes()


def test_a_minimum_budget_takes_effect_at_the_next_check():
    # The fixed point is quiet from the first batch, so the default stops at
    # PATIENCE batches; a floor between checks runs on to the check past it.
    output = np.zeros((2, 2, USER_BASE + 2))
    output[:, :, USER_BASE:] = 0.5
    params = HMMParams(
        np.full(2, 0.5), np.full((2, 2), 0.5), output, _vocabulary(USER_BASE + 2)
    )
    a, b = USER_BASE, USER_BASE + 1
    records = [SequenceRecord(np.array([a, b, b, a]))]

    result = baum_welch(params, records, min_cycles=PATIENCE * BATCH_CYCLES + 1)

    assert result.converged and result.cycles == (PATIENCE + 1) * BATCH_CYCLES


def test_a_minimum_budget_stops_at_the_first_quiet_check_past_it():
    # Read back from the trace, as the plain rule's test does: the stop is a
    # check at or past the floor ending PATIENCE unchanged batches, and no
    # earlier check at or past the floor ended such a run.
    rng = np.random.default_rng(SEED + 99)
    params, records = _random_params(rng, 4, 4), _random_corpus(rng, 4, (80, 50))
    batch, threshold, patience = 5, 1e-3, 3
    plain = baum_welch(
        params, records, batch_cycles=batch, change_bits=threshold, patience=patience
    )
    floor = plain.cycles + 2 * batch + 1
    result = baum_welch(
        params,
        records,
        batch_cycles=batch,
        change_bits=threshold,
        patience=patience,
        min_cycles=floor,
    )

    assert result.converged and result.cycles >= floor and result.cycles % batch == 0
    ends = np.array(result.description_lengths[::batch])
    quiet = 0
    for check, flag in enumerate(np.abs(np.diff(ends)) < threshold, start=1):
        quiet = quiet + 1 if flag else 0
        stops = quiet >= patience and check * batch >= floor
        assert stops == (check * batch == result.cycles)


def test_a_smaller_max_cycles_wins_over_a_minimum_budget():
    rng = np.random.default_rng(SEED + 44)
    params, records = _random_params(rng, 3, 3), _random_corpus(rng, 3, (50,))
    result = baum_welch(params, records, max_cycles=20, min_cycles=50)
    assert result.cycles == 20 and not result.converged


def test_a_negative_minimum_budget_is_refused():
    rng = np.random.default_rng(SEED + 45)
    params, records = _random_params(rng, 2, 3), _random_corpus(rng, 3, (20,))
    with pytest.raises(ValueError, match="min_cycles must be non-negative, got -1"):
        baum_welch(params, records, min_cycles=-1)


def test_the_lush_trained_fixture_is_already_near_its_fixed_point():
    # m008_0001_008 is the original's converged output, trained on this very
    # stream, so the loop should leave it almost where it is. Passing the whole
    # corpus as one record is the original's flat stream.
    params = load_params(FIXTURES / "m008_0001_008.hmm")
    result = baum_welch(params, [load_corpus_record()])

    first, last = result.description_lengths[0], result.description_lengths[-1]
    assert result.converged and result.cycles == PATIENCE * BATCH_CYCLES
    assert 0 <= first - last < CHANGE_BITS
    assert result.degenerate_states == ()


def test_one_cycle_over_several_records_is_the_m_step_of_their_summed_counts():
    rng = np.random.default_rng(SEED + 7)
    params = _random_params(rng, 3, 4)
    records = _random_corpus(rng, 4, (25, 9, 31))

    result = baum_welch(params, records, max_cycles=1)

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

    result = baum_welch(params, [SequenceRecord(codes)], max_cycles=5)

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
        baum_welch(params, records)


@pytest.mark.parametrize(
    "records, kwargs, message",
    [
        ([np.array([USER_BASE, 99])], {}, "outside the model's symbol axis"),
        ([np.array([], dtype=np.int64)], {}, "no symbols"),
        ([], {}, "no symbols"),
        ([np.array([USER_BASE])], {"change_bits": 0.0}, "must be positive"),
        ([np.array([USER_BASE])], {"batch_cycles": 0}, "at least 1"),
        ([np.array([USER_BASE])], {"max_cycles": -1}, "non-negative"),
        ([np.array([USER_BASE])], {"batch_size": 0}, "batch_size must be at least 1"),
    ],
    ids=[
        "code-out-of-range",
        "only-empty",
        "no-records",
        "zero-threshold",
        "zero-batch",
        "negative-budget",
        "zero-batch-size",
    ],
)
def test_the_loop_rejects_what_it_cannot_train_on(records, kwargs, message):
    rng = np.random.default_rng(SEED)
    params = _random_params(rng, 2, 3)
    with pytest.raises(ValueError, match=message):
        baum_welch(params, [SequenceRecord(codes) for codes in records], **kwargs)


# === batching =====================================================================
#
# `batch_size` is a memory knob and must be nothing else. The kernel returns each
# record's counts bit for bit as that record alone would produce them, and the loop
# sums them in record order, so training is bit-identical at every batch size. A
# mask that lets a padded step touch α, β or ξ breaks the first test at once, and
# only when records of different lengths share a batch, which is why every case
# below has padding in it and the test asserts that it does.


def _ragged_cases():
    rng = np.random.default_rng(SEED + 100)
    cases = []
    for size, n_user, lengths, dead in [
        (3, 4, (15, 0, 40, 1), 0.0),
        (5, 3, (200, 1, 1, 1), 0.3),  # almost all padding
        (1, 5, (7, 30), 0.0),
        (6, 6, (0, 0, 9), 0.4),
    ]:
        init, transition, output = _random_model(rng, size, n_user, dead)
        full = np.zeros((size, size, USER_BASE + n_user))
        full[:, :, USER_BASE:] = output
        records = _random_corpus(rng, n_user, lengths)
        cases.append((init, transition, full, records))
    # One record made impossible by UNK, whose fibres are zero in every model.
    init, transition, full, records = cases[0]
    impossible = SequenceRecord(np.array([USER_BASE, UNK, USER_BASE]))
    cases.append((init, transition, full, [records[2], impossible, records[3]]))
    # PAD's fibre is zero in every valid model, so padding already weighs 0 and ξ
    # there is 0 without the kernel's mask: dropping the mask is an equivalent
    # mutant on any HMMParams. Raw arrays with PAD emittable make the mask the only
    # thing standing between padding and the counts. The per-record step never
    # reads PAD on real codes, so the comparison is still exact.
    init, transition, full, records = cases[1]
    leaky = full.copy()
    leaky[:, :, PAD] = 0.5
    leaky /= leaky.sum(axis=2, keepdims=True)
    cases.append((init, transition, leaky, records))
    return cases


@pytest.mark.parametrize(
    "case",
    _ragged_cases(),
    ids=["mixed", "mostly-padding", "S1", "empties", "impossible", "pad-emittable"],
)
def test_every_row_of_a_batch_is_the_per_record_step_bit_for_bit(case):
    init, transition, output, records = case
    batch = pad_collate(records)
    assert not batch["mask"].all(), "no padding in the batch, so the test checks nothing"

    counts, bits_per_record = _e_step_batch(init, transition, output, batch["codes"], batch["lengths"])
    for b, record in enumerate(records):
        (init_c, transition_c, emission_c), bits = _record_step(init, transition, output, record.codes)
        np.testing.assert_array_equal(counts[0][b], init_c)
        np.testing.assert_array_equal(counts[1][b], transition_c)
        np.testing.assert_array_equal(counts[2][b], emission_c)
        assert bits_per_record[b] == bits or (np.isinf(bits) and np.isinf(bits_per_record[b]))


@pytest.mark.parametrize(
    "size, n_user, lengths",
    [(3, 4, (15, 0, 40, 1, 22)), (5, 5, (60, 7, 33, 1, 2, 90, 14))],
    ids=["S3-five-records", "S5-seven-records"],
)
def test_training_is_bit_identical_at_every_batch_size(size, n_user, lengths):
    rng = np.random.default_rng(SEED + 200 + size)
    params = _random_params(rng, size, n_user)
    records = _random_corpus(rng, n_user, lengths)

    runs = [baum_welch(params, records, batch_size=b, max_cycles=25) for b in (1, 2, 3, None)]
    first = runs[0]
    for run in runs[1:]:
        assert run.cycles == first.cycles
        assert run.description_lengths == first.description_lengths
        assert run.degenerate_states == first.degenerate_states
        np.testing.assert_array_equal(run.params.init_state_p, first.params.init_state_p)
        np.testing.assert_array_equal(run.params.transition_p, first.params.transition_p)
        np.testing.assert_array_equal(run.params.output_p, first.params.output_p)


# === the progress log =============================================================


class _FlushRecorder(io.StringIO):
    """A stream that records how many lines it held at each flush."""

    def __init__(self):
        super().__init__()
        self.lines_at_flush = []

    def flush(self):
        self.lines_at_flush.append(self.getvalue().count("\n"))
        super().flush()


def test_no_log_writes_nothing(capsys):
    rng = np.random.default_rng(SEED + 40)
    baum_welch(_random_params(rng, 3, 3), _random_corpus(rng, 3, (30, 20)))
    assert capsys.readouterr() == ("", "")


def test_the_log_rows_are_the_results_check_values_and_count_unchanged_checks():
    # On the reference, a check's forward pass equals that cycle's E-step total
    # bit for bit (0 mismatches in 2935 cycles, measured 2026-09-15), so each row
    # is predictable from the returned trace. The saddle run's flags are
    # 110000000111, so the unchanged count also shows its reset.
    params, records = _near_saddle(1e-4)
    log = io.StringIO()
    result = baum_welch(params, records, batch_cycles=2, change_bits=1e-3, patience=3, log=log)

    h = result.description_lengths
    width = len(f"{h[0]:.6f}")
    unchanged = [1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
    expected = [f"  cycle  {'bits':>{width}}         change  quiet", f"      0  {h[0]:.6f}"]
    for cycle, count in zip(range(2, 25, 2), unchanged):
        change = h[cycle] - h[cycle - 2]
        expected.append(f"  {cycle:>5}  {h[cycle]:>{width}.6f}  {change:>13.6e}  {count}/3")
    expected.append("  converged after 24 cycles")

    assert result.cycles == 24
    assert log.getvalue().splitlines() == expected


def test_the_columns_stay_aligned_from_a_huge_start_to_tiny_and_infinite_changes():
    # EM never raises the description length, so a width taken from cycle 0
    # holds for every later bits value; change is fixed-width scientific.
    log = io.StringIO()
    width = _log_start(log, 12345678.901234)
    _log_row(log, width, 10, 9876543.21, -2469135.69, 0, 3)
    _log_row(log, width, 20, 9876543.2099, -1.1e-7, 1, 3)
    _log_row(log, width, 990, 0.0, -np.inf, 3, 3)
    _log_row(log, width, 12345, 1.5, np.inf)
    header, *rows = log.getvalue().splitlines()

    ends = [match.end() for match in re.finditer(r"\S+", header)]
    for row in rows:
        row_ends = [match.end() for match in re.finditer(r"\S+", row)]
        assert row_ends[:3] == ends[: len(row_ends[:3])]
    assert "-1.100000e-07" in rows[2]
    assert rows[3].split()[2] == "-inf" and rows[4].split()[2] == "inf"


@pytest.mark.parametrize("max_cycles", [0, 7, 10])
def test_a_budget_stop_logs_the_last_bits_and_says_why_it_stopped(max_cycles):
    # 7 stops between checks, so its row has no unchanged count; 10 stops on a
    # check, whose row is already written; 0 stops before any cycle runs.
    rng = np.random.default_rng(SEED + 41)
    params, records = _random_params(rng, 3, 3), _random_corpus(rng, 3, (60, 40))
    log = io.StringIO()
    result = baum_welch(params, records, max_cycles=max_cycles, change_bits=1e-12, log=log)
    lines = log.getvalue().splitlines()
    h = result.description_lengths

    assert not result.converged
    assert lines[-1] == f"  stopped at max_cycles after {max_cycles} cycles"
    last = lines[-2].split()
    assert int(last[0]) == max_cycles and last[1] == f"{h[-1]:.6f}"
    assert len(last) == {0: 2, 7: 3, 10: 4}[max_cycles]
    if max_cycles == 7:
        assert last[2] == f"{h[7] - h[0]:.6e}"


def test_every_line_is_flushed_as_it_is_written():
    rng = np.random.default_rng(SEED + 42)
    log = _FlushRecorder()
    baum_welch(_random_params(rng, 3, 3), _random_corpus(rng, 3, (40,)), log=log)
    written = log.getvalue().count("\n")
    assert written > 3 and log.lines_at_flush == list(range(1, written + 1))
