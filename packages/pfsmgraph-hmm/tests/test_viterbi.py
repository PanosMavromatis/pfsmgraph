"""The Viterbi decode, checked against the original's own saved decodes.

This suite has an oracle, which is unusual enough to be worth stating. The Lush
trainer's `save-viterbi-path` wrote a `<model>.vpath.xls` beside every saved
model -- `Output / States / Entropy`, one row per position -- and all three
tracked models have one, with the corpus that produced them tracked alongside.
So the port is not checked against numbers we chose, and **cannot accidentally
validate its kernel against itself**, which is the standing hazard in porting a
known defect: the comparison target was written by the program being replaced.

Three properties of the oracle are load-bearing and are asserted rather than
assumed. The corpus concatenation must reproduce the `Output` column, or the
`States` column is not aligned to anything. The file is `N + 1` rows for `N`
symbols, which is ADR 0015's arc-emission geometry written to disk by the
original's own author. And the `Entropy` column is `state_entropies[state]`,
which `ViterbiPath` deliberately does not carry -- so deriving it here turns a
column we could have stored into a fourth independent check.

The one place the port and the oracle disagree is the seeding defect, and the
disagreement is asserted in both directions: every position after the seed
agrees on all three models, and position 0 of `m008_0001_008` diverges *toward
the more probable start state*. A test that only asserted agreement would pass
just as well against a port that reproduced the bug.

**Every shared test runs once per backend.** ADR 0003 asks for one suite per
algorithm, written against the public API with the backend as a fixture
parameter, and ADR 0021's `viterbi(..., backend=...)` is what makes both halves
hold at once: the `backend` fixture in `conftest.py` names each backend the
table has, and an unavailable one skips with its reason. So a green run means
each shared case ran on each available backend, and the equivalence tests below
compare every backend against `backend="python"` exactly, not approximately.

Below the labelled line are the tests that cannot be parameter values: the
kernel contract (a kernel returns `inf` rather than raising), behaviour only one
backend has (thread counts, block sizes), and the check that the tie fixture can
tell two logarithms apart. Until 2026-09-14 that section also held a copy of the
oracle, seeding, tie and boundary tests per phase, because the public call had
nowhere to put a backend; those copies are folded into the shared tests above.
"""

from __future__ import annotations

import functools
import importlib
import itertools
import math

import numpy as np
import pytest
from pfsmgraph.dataseq import (
    BOS,
    EOS,
    PAD,
    UNK,
    USER_BASE,
    SequenceRecord,
    SymbolTable,
)

import pfsmgraph.hmm
from pfsmgraph.hmm import (
    HMMParams,
    ImpossibleSequenceError,
    ViterbiPath,
    backends,
    viterbi,
)
from pfsmgraph.hmm._backends import _resolve
from pfsmgraph.hmm._numeric import bits
from pfsmgraph.hmm._viterbi import STATE_DTYPE

from _lush_fixtures import (
    FIXTURES,
    SAVED_MODELS,
    corpus_alphabet,
    load_corpus_codes,
    load_corpus_record,
    load_params,
    load_vpath,
    read_ascii_matrix,
)

#: The four-decimal prints in a `.vpath.xls` propagate through `d(entropy)/dp`,
#: which is about 1.9 near these probabilities. `test_numeric.py` derived this
#: bound for `state_entropies` in PR #16 and `test_params.py` re-derived it in
#: goal 1 at 1.27e-4 observed; the same magnitude turns up here from a different
#: file written by a different code path. Reused rather than re-chosen.
ENTROPY_TOL = 5e-4


# --- building small models by hand -----------------------------------------


def build(init_p, transition_p, user_output, symbols=("a", "b")):
    """An `HMMParams` from user-symbol emissions, reserved block zero-padded.

    `user_output` is `(S, S, len(symbols))`; it is placed at `[..., USER_BASE:]`
    exactly as `load_params` places a Lush fixture's, so a hand-built model and
    a migrated one are indexed the same way.
    """
    vocabulary = SymbolTable(list(symbols))
    user_output = np.asarray(user_output, dtype=np.float64)
    size = user_output.shape[0]
    output_p = np.zeros((size, size, vocabulary.size), dtype=np.float64)
    output_p[:, :, USER_BASE:] = user_output
    return HMMParams(
        init_state_p=np.asarray(init_p, dtype=np.float64),
        transition_p=np.asarray(transition_p, dtype=np.float64),
        output_p=output_p,
        vocabulary=vocabulary,
    )


def code(index: int) -> int:
    """The ADR 0011 code of the `index`-th user symbol."""
    return USER_BASE + index


def record(*indices, label=None) -> SequenceRecord:
    return SequenceRecord(
        np.asarray([code(i) for i in indices], dtype=np.int32), label=label
    )


def path_cost(params, codes, states) -> float:
    """The description length of one path, summed arc by arc.

    Written out longhand rather than reusing the kernel, so that a test of the
    kernel's total is a test rather than a restatement.
    """
    total = float(bits(params.init_state_p[states[0]]))
    for position, symbol in enumerate(codes):
        source, destination = int(states[position]), int(states[position + 1])
        total += float(
            bits(
                params.transition_p[source, destination]
                * params.output_p[source, destination, symbol]
            )
        )
    return total


def _random_model(rng, size, n_symbols):
    """A valid random `HMMParams`: rows and live fibres normalized."""
    transition_p = rng.random((size, size))
    transition_p /= transition_p.sum(axis=1, keepdims=True)
    user_output = rng.random((size, size, n_symbols))
    user_output /= user_output.sum(axis=2, keepdims=True)
    init_p = rng.random(size)
    init_p /= init_p.sum()
    return build(init_p, transition_p, user_output, symbols=tuple(
        f"s{i}" for i in range(n_symbols)
    ))


def _generated_models(rng, count, max_size, max_length=40):
    """Seeded `(params, record)` pairs over many shapes, empty records included."""
    for _ in range(count):
        size = int(rng.integers(1, max_size + 1))
        n_symbols = int(rng.integers(1, 6))
        params = _random_model(rng, size, n_symbols)
        codes = np.asarray(
            [code(int(i)) for i in rng.integers(0, n_symbols, int(rng.integers(0, max_length)))],
            dtype=np.int32,
        )
        yield params, SequenceRecord(codes)


def _uniform_model(size, n_symbols):
    """Exactly uniform everywhere, so every candidate at every position ties."""
    return build(
        np.full(size, 1.0 / size),
        np.full((size, size), 1.0 / size),
        np.full((size, size, n_symbols), 1.0 / n_symbols),
        symbols=tuple(f"s{i}" for i in range(n_symbols)),
    )


# --- the differential decode ------------------------------------------------


class Oracle:
    """One saved model, its saved decode, and one backend's decode of the corpus."""

    def __init__(self, model, backend):
        self.model = model
        self.backend = backend
        self.params = load_params(FIXTURES / model)
        self.record = load_corpus_record()
        self.symbols, self.states, self.entropies = load_vpath(model)
        self.path = viterbi(self.params, self.record, backend=backend)


@functools.cache
def _oracle(model, backend="python"):
    """Decode the 1269-symbol corpus once per model and backend, not once per test."""
    return Oracle(model, backend)


@pytest.fixture(scope="module", params=SAVED_MODELS)
def oracle(request, backend):
    return _oracle(request.param, backend)


@pytest.fixture(scope="module", params=SAVED_MODELS)
def saved(request):
    """The saved files alone, for tests about the oracle rather than a decode."""
    return _oracle(request.param)


def test_the_corpus_reproduces_each_saved_decodes_output_column(saved):
    """Without this the `States` column is aligned to nothing.

    `fprop-all` decoded the 200 `.seq` files as one stream, which is why a
    single `.vpath.xls` covers all of them. Numeric concatenation order is an
    assumption until this passes.
    """
    alphabet = corpus_alphabet()
    expected = [alphabet[int(c)] for c in load_corpus_codes()]
    assert saved.symbols[1:] == expected


def test_the_saved_decode_has_one_more_row_than_it_has_symbols(saved):
    """ADR 0015's N+1 geometry, printed to disk by the original's own author.

    The leading row's `Output` is a literal `-`: no symbol has been emitted yet,
    because emission happens while *crossing* an arc.
    """
    assert saved.symbols[0] == "-"
    assert len(saved.states) == saved.record.length + 1


def test_our_path_has_the_same_shape_as_the_saved_one(oracle):
    assert oracle.path.states.shape == oracle.states.shape
    assert oracle.path.n_symbols == oracle.record.length


def test_reproduces_the_originals_own_decode_after_the_seed(oracle):
    """Every position the seeding defect cannot reach, on all three models.

    1268 of 1269 positions, and the exception is position 0 by construction --
    the seed is the only thing the defect touches.
    """
    assert np.array_equal(oracle.path.states[1:], oracle.states[1:])


def test_reproduces_the_originals_own_path_entropies(oracle):
    """The `Entropy` column, derived rather than stored.

    `ViterbiPath` carries no per-position entropy (ADR 0017: derived quantities
    are computed), so this both checks the decode and demonstrates that the
    omitted field is one fancy-index away.
    """
    derived = oracle.params.state_entropies[oracle.path.states]
    # Position 0 may differ on m008, where the decoded state itself differs.
    assert derived[1:] == pytest.approx(oracle.entropies[1:], abs=ENTROPY_TOL)


def test_the_total_is_the_sum_of_the_arcs_the_path_crossed(oracle):
    """Internal consistency: the reported cost is the cost of the reported path."""
    expected = path_cost(oracle.params, oracle.record.codes, oracle.path.states)
    assert oracle.path.total_bits == pytest.approx(expected, rel=1e-12)


def test_the_decoded_path_is_at_least_as_good_as_the_originals(oracle):
    """The correction can only improve the cost, never worsen it.

    A sharper statement than "we agree at 1268/1269": it says the one
    disagreement is not a coin toss but a strictly better path under the
    objective both programs claim to minimize.
    """
    theirs = path_cost(oracle.params, oracle.record.codes, oracle.states)
    assert oracle.path.total_bits <= theirs + 1e-9


def test_every_backend_decodes_the_corpus_exactly_as_the_reference_does(oracle):
    """The oracle's positions agree above; this pins the whole path and total.

    The `.vpath.xls` files are the one evidence here not downstream of code
    written in this repository, and the formalization was *recovered* from the
    phase-1 kernel, so running the oracle on every backend is what stops a
    compiled one inheriting that circularity unexamined. Exact: all four
    backends take their logarithms through `bits` and only add and compare.
    """
    reference = _oracle(oracle.model).path
    assert np.array_equal(oracle.path.states, reference.states)
    assert oracle.path.total_bits == reference.total_bits


# --- the seeding defect, in both directions ---------------------------------


def test_the_seed_agrees_wherever_the_defect_cannot_bite(backend):
    """Two of the three models start in the same state either way.

    Recorded so that the divergence below reads as one specific measured
    disagreement rather than as a general disclaimer.
    """
    for model in ("m001_0001_001.hmm", "m001_0005_005.hmm"):
        oracle = _oracle(model, backend)
        assert oracle.path.states[0] == oracle.states[0]


def test_the_seeding_fix_diverges_at_exactly_one_position(backend):
    """`m008_0001_008` position 0, and nothing else in 3807 positions.

    Pinned rather than widened on every backend: the oracle is *wrong* at
    exactly one position by construction, so an allowance for "mostly agrees"
    would let a genuine backend defect hide inside the one made for a known one.
    """
    oracle = _oracle("m008_0001_008.hmm", backend)
    disagreements = np.flatnonzero(oracle.path.states != oracle.states)
    assert disagreements.tolist() == [0]


def test_the_divergence_prefers_the_more_probable_start_state(backend):
    """The defect decides *backwards*, and this is the assertion that says so.

    The original seeds delta with a raw probability into a bit-domain
    accumulator, where smaller is better -- so it prefers the *less* likely
    start whenever the seed is what decides. Here the two live starts are 0.3665
    and 0.6335 and their best outgoing arcs differ by 0.004 bits, so the seed
    decides it alone.
    """
    oracle = _oracle("m008_0001_008.hmm", backend)
    ours = int(oracle.path.states[0])
    theirs = int(oracle.states[0])
    init_p = oracle.params.init_state_p
    assert ours != theirs
    assert init_p[ours] > init_p[theirs]


# --- min-sum, not max-product -----------------------------------------------


def test_the_decode_returns_the_cheapest_path_of_all(backend):
    """Brute force over every path, which a max-product port fails immediately.

    Three states and four symbols is 3**5 = 243 paths, so the optimum can be
    found by enumeration rather than argued for.
    """
    rng = np.random.default_rng(20260903)
    transition_p = rng.random((3, 3)) + 0.1
    transition_p /= transition_p.sum(axis=1, keepdims=True)
    user_output = rng.random((3, 3, 2)) + 0.1
    user_output /= user_output.sum(axis=2, keepdims=True)
    init_p = np.array([0.2, 0.3, 0.5])
    params = build(init_p, transition_p, user_output)

    codes = [code(0), code(1), code(1), code(0)]
    decoded = viterbi(params, SequenceRecord(np.asarray(codes, dtype=np.int32)), backend=backend)

    best = min(
        path_cost(params, codes, states)
        for states in itertools.product(range(3), repeat=len(codes) + 1)
    )
    assert decoded.total_bits == pytest.approx(best)


def test_ties_are_broken_toward_the_lower_state_index(backend):
    """The original's tie-break, pinned by construction because the fixtures
    cannot exercise it.

    Lush guards its update with a *strict* "cand is better than current"
    (`safe->--log`), so iterating predecessors in ascending order keeps the
    first minimal one; `np.argmin` does the same. The three tracked models
    produce **0 exact ties in 3804 positions** -- learned float parameters do
    not collide -- so a differential test alone would pass against a last-wins
    port. It is not a hypothetical case either: an exactly uniform model ties at
    every position, and `rand_p_vector(size, noise_width=0)` returns exactly
    that, which is how revision 03 initialises.

    Run on every backend, this is also the test that catches a `<=` in a
    compiled reduction, or a CPU-parallel `prange` over the predecessors `i`
    rather than the states `j`: a parallel reduction resolves ties in whatever
    order the partials combined. Five states and 24 positions give each backend
    120 ties to get wrong.
    """
    params = _uniform_model(5, 3)
    decoded = viterbi(
        params, record(*(i % 3 for i in range(24))), backend=backend
    )
    assert decoded.states.tolist() == [0] * 25


def test_the_most_probable_path_is_the_one_with_fewest_bits(backend):
    """The orientation itself: higher probability must mean *lower* total.

    Stated as a test because the original's naming actively misleads -- its
    `data-p` and `result-p` hold bits despite a suffix meaning probability.
    """
    transition_p = np.array([[0.9, 0.1], [0.1, 0.9]])
    user_output = np.zeros((2, 2, 2))
    user_output[:, :, 0] = 1.0
    params = build(np.array([0.5, 0.5]), transition_p, user_output)

    staying = path_cost(params, [code(0)], [0, 0])
    switching = path_cost(params, [code(0)], [0, 1])
    assert staying < switching  # 0.9 is more probable, so it costs fewer bits
    assert viterbi(params, record(0), backend=backend).total_bits == pytest.approx(staying)


# --- arc emission ------------------------------------------------------------


def test_the_emission_depends_on_the_destination_as_well_as_the_source(backend):
    """A model no `B[state, symbol]` can represent, decoded correctly.

    Both arcs leave state 0, and they emit *different* symbols with certainty.
    Under the state-emission formulation every textbook uses, the two would be
    indistinguishable and the decode could not recover which arc was taken.
    """
    transition_p = np.array([[0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    user_output = np.zeros((3, 3, 2))
    user_output[0, 1, 0] = 1.0  # 0 -> 1 emits "a"
    user_output[0, 2, 1] = 1.0  # 0 -> 2 emits "b"
    user_output[1, 1, 0] = 1.0
    user_output[2, 2, 1] = 1.0
    params = build(np.array([1.0, 0.0, 0.0]), transition_p, user_output)

    assert viterbi(params, record(0), backend=backend).states.tolist() == [0, 1]
    assert viterbi(params, record(1), backend=backend).states.tolist() == [0, 2]


# --- the N+1 geometry --------------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, 2, 5, 17])
def test_a_path_over_n_symbols_visits_n_plus_one_states(length, backend):
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, record(*([0] * length)), backend=backend)
    assert decoded.states.shape == (length + 1,)
    assert decoded.n_symbols == length


def test_an_empty_record_still_visits_one_state(backend):
    """`N = 0` is a path of one state and no arcs, not an error.

    The cost is then the seed alone, which is the cleanest available statement
    that `delta[0]` is in the bit domain.
    """
    params = build(
        np.array([0.25, 0.75]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, SequenceRecord(np.array([], dtype=np.int32)), backend=backend)
    assert decoded.states.tolist() == [1]
    assert decoded.total_bits == pytest.approx(float(bits(0.75)))


# --- impossibility -----------------------------------------------------------


def test_a_record_carrying_unk_is_impossible(backend):
    """The case goal 1's `A == vocab.size` decision was chosen to make loud.

    `encode(..., on_unknown="unk")` is a documented `dataseq` path, and
    `HMMParams` requires the reserved fibres to be exactly zero -- so `UNK`
    reaches a zero, `bits(0)` is `+inf`, and the path is reported impossible
    rather than built from some other symbol's emissions.
    """
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    codes = np.array([code(0), UNK, code(1)], dtype=np.int32)
    with pytest.raises(ImpossibleSequenceError) as excinfo:
        viterbi(params, SequenceRecord(codes), backend=backend)
    assert "symbol 1 of the record" in str(excinfo.value)
    assert f"code {UNK}" in str(excinfo.value)


def test_a_record_carrying_pad_is_impossible(backend):
    """`PAD` is code 0, and a record never holds padding -- but nothing stops a
    caller building one by hand, and zero is the value an uninitialized buffer
    has."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ImpossibleSequenceError):
        viterbi(params, SequenceRecord(np.array([code(0), PAD], dtype=np.int32)), backend=backend)


def test_an_unseen_bigram_is_impossible(backend):
    """The other route to impossibility: a live symbol on a dead arc."""
    transition_p = np.array([[1.0, 0.0], [0.0, 1.0]])  # no arc between them
    user_output = np.zeros((2, 2, 2))
    user_output[0, 0, 0] = 1.0  # state 0 loops emitting "a"
    user_output[1, 1, 1] = 1.0  # state 1 loops emitting "b"
    params = build(np.array([1.0, 0.0]), transition_p, user_output)

    assert viterbi(params, record(0, 0), backend=backend).states.tolist() == [0, 0, 0]
    with pytest.raises(ImpossibleSequenceError) as excinfo:
        viterbi(params, record(0, 1), backend=backend)
    assert "symbol 1 of the record" in str(excinfo.value)


def test_the_impossible_error_is_a_valueerror():
    """`except ValueError` keeps working; the distinct type is for revision 04.

    Its topology search decodes many sequences against many candidate
    topologies, where an impossible sequence is an ordinary search outcome
    rather than a malformed input -- so it must be catchable by type and not by
    message.
    """
    assert issubclass(ImpossibleSequenceError, ValueError)


def test_no_path_with_an_infinite_total_ever_reaches_a_caller(backend):
    """The guarantee `ViterbiPath.total_bits` documents."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ImpossibleSequenceError):
        viterbi(params, SequenceRecord(np.array([UNK], dtype=np.int32)), backend=backend)


# --- the symbol-axis guard ---------------------------------------------------


def test_a_code_past_the_end_of_the_symbol_axis_is_refused(backend):
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ValueError, match="outside the model's symbol axis"):
        viterbi(params, SequenceRecord(np.array([code(0), 99], dtype=np.int32)), backend=backend)


def test_a_negative_code_is_refused(backend):
    """Not reachable through `encode`, and exactly what the rejected
    user-symbols-only axis would have produced: `1 - USER_BASE` is `-5`, which
    numpy indexes without complaint."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ValueError, match="outside the model's symbol axis"):
        viterbi(params, SequenceRecord(np.array([-5], dtype=np.int32)), backend=backend)


def test_a_code_exactly_at_the_end_of_the_symbol_axis_is_the_boundary(backend):
    """The `>=` in the range guard, pinned. TC-19 of the formalization.

    The two range tests above use `99` and `-5`, both far from the edge, so a
    guard written `>` where it should be `>=` passes the entire suite while
    letting code `A` through to `output_p[:, :, A]` -- an `IndexError` from
    numpy rather than the `ValueError` this promises.

    Both halves are asserted against *this* model deliberately. Whether `A - 1`
    decodes is a property of the model, not of the guard: here `code(1)` is
    `A - 1` and `np.full` makes it emittable, so the contrast is
    decode-versus-refuse, which discriminates far better than
    refuse-versus-refuse. A model that cannot emit `A - 1` raises
    `ImpossibleSequenceError` there instead, which would pass a carelessly
    written version of this test for an unrelated reason.
    """
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    axis = params.n_symbols
    assert code(1) == axis - 1, "the last user symbol must be the last valid code"

    # `A - 1` is in range and emittable, so it decodes.
    assert viterbi(params, SequenceRecord(np.array([axis - 1], dtype=np.int32)), backend=backend)

    # `A` is out of range by exactly one.
    with pytest.raises(ValueError, match="outside the model's symbol axis") as excinfo:
        viterbi(params, SequenceRecord(np.array([axis], dtype=np.int32)), backend=backend)
    assert f"[0, {axis})" in str(excinfo.value)


def test_impossibility_is_reported_at_position_zero(backend):
    """The lower boundary of the position report. TC-20 of the formalization.

    Every other impossibility test places the dead symbol at position 1 or
    later, so `_dead_symbol`'s `enumerate` has never been checked at its own
    first iteration -- an off-by-one there would report symbol 1, or fall
    through to the `AssertionError` its docstring calls unreachable.

    The record is impossible on its *first* symbol: only state 0 is reachable,
    and state 0 emits nothing but "a".
    """
    transition_p = np.array([[1.0, 0.0], [0.0, 1.0]])
    user_output = np.zeros((2, 2, 2))
    user_output[0, 0, 0] = 1.0  # state 0 loops emitting "a"
    user_output[1, 1, 1] = 1.0  # state 1 loops emitting "b"
    params = build(np.array([1.0, 0.0]), transition_p, user_output)

    with pytest.raises(ImpossibleSequenceError) as excinfo:
        viterbi(params, record(1), backend=backend)
    assert "symbol 0 of the record" in str(excinfo.value)
    assert "reaches path position 1" in str(excinfo.value)


def test_the_range_error_names_the_likely_cause(backend):
    """A vocabulary mismatch, which is what this almost always is."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ValueError) as excinfo:
        viterbi(params, SequenceRecord(np.array([99], dtype=np.int32)), backend=backend)
    assert "different vocabulary" in str(excinfo.value)


# --- ViterbiPath -------------------------------------------------------------


def test_the_states_array_is_read_only(backend):
    """Same reasoning as `HMMParams`: a result handed out mutable is a result
    that can be corrupted after the fact."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, record(0, 1), backend=backend)
    assert not decoded.states.flags.writeable
    with pytest.raises(ValueError):
        decoded.states[0] = 1


def test_the_states_are_integers_not_floats(backend):
    """`HMMLIB-ACCOUNT.md` section 7's second defect, not reproduced.

    The original declares `psi` a `float-matrix` and round-trips state indices
    through it, which is exact only below 2**24 states.

    **This asserts the output dtype, not the backtrace's internal one, and that
    is the strongest true statement available.** Making `psi` float64 again
    changes no observable behaviour at any testable scale -- every index below
    2**24 is exactly representable -- so no test can distinguish it, which is
    precisely why the master plan judged it harmless rather than a bug to fix.
    Measured: the mutation leaves this whole module green. What is worth pinning
    is that a *caller* receives integers, since `states` is an index array and a
    float one would silently work until it was used as one.
    """
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    assert viterbi(params, record(0), backend=backend).states.dtype == STATE_DTYPE
    assert np.issubdtype(STATE_DTYPE, np.integer)


def test_the_label_is_carried_through(backend):
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    assert viterbi(params, record(0, label="utterance-7"), backend=backend).label == "utterance-7"
    assert viterbi(params, record(0), backend=backend).label is None


def test_a_two_dimensional_state_array_is_refused():
    with pytest.raises(ValueError, match="states must be 1-D"):
        ViterbiPath(states=np.zeros((2, 2)), total_bits=0.0)


def test_the_path_copies_the_array_it_is_given():
    """Freezing the caller's own array would mutate an object we do not own."""
    states = np.array([0, 1, 0])
    path = ViterbiPath(states=states, total_bits=1.0)
    assert states.flags.writeable
    states[0] = 9
    assert path.states[0] == 0


def test_the_repr_names_the_symbol_count_not_the_state_count():
    """`n_symbols`, because a reader comparing it to their input sequence should
    see the length they passed in rather than one more."""
    path = ViterbiPath(states=np.array([0, 1, 0]), total_bits=2.5, label="x")
    assert "n_symbols=2" in repr(path)
    assert "label='x'" in repr(path)


# --- how the fixtures' own boundary markers are mapped ----------------------


def test_the_corpus_names_its_first_two_codes_begin_and_end():
    """The premise of the decision below, read from the corpus rather than assumed.

    These names exist nowhere in a saved model -- its `_alphabet` holds Lush
    pointer addresses -- so the mapping question is only answerable because the
    corpus was tracked beside the models.
    """
    alphabet = corpus_alphabet()
    assert alphabet[0] == "begin"
    assert alphabet[1] == "end"
    assert set(alphabet.values()) == {"begin", "end", "a", "b", "c", "d"}


def test_mapping_begin_and_end_onto_bos_and_eos_is_refused():
    """The obvious mapping is wrong, and this is the assertion that it fails loudly.

    Lush codes 0 and 1 are the corpus's own boundary markers, so putting them at
    ADR 0011's `BOS` and `EOS` looks like the faithful translation. It is not:
    they carry real emission mass, the reserved block must be exactly zero, and
    `HMMParams` refuses at construction. `load_params` therefore maps every Lush
    code through `USER_BASE + c`, making `begin` and `end` ordinary user symbols
    -- which is what they are to this model, since it emits them and scores them.

    The test lives beside the decode because the decode is what would have been
    quietly wrong: a reserved fibre is zero, `bits(0)` is `+inf`, and a corpus
    whose every sequence opens with `begin` would have been reported impossible.
    Whether a *library* should treat boundary markers as `BOS`/`EOS` is a
    separate question, and it belongs to `dataseq`'s encoder.
    """
    directory = FIXTURES / "m001_0005_005.hmm"
    init_state_p = read_ascii_matrix(directory / "init_state_p")
    transition_p = read_ascii_matrix(directory / "transition_p")
    saved_output_p = read_ascii_matrix(directory / "output_p")
    init_state_p = init_state_p / init_state_p.sum()
    transition_p = transition_p / transition_p.sum(axis=1, keepdims=True)

    size, _, alphabet_size = saved_output_p.shape
    # begin -> BOS, end -> EOS, and the four real symbols from USER_BASE.
    destinations = [BOS, EOS] + list(
        range(USER_BASE, USER_BASE + alphabet_size - 2)
    )
    output_p = np.zeros((size, size, max(destinations) + 1), dtype=np.float64)
    for lush_code, destination in enumerate(destinations):
        output_p[:, :, destination] = saved_output_p[:, :, lush_code]
    vocabulary = SymbolTable([f"s{i}" for i in range(alphabet_size - 2)])

    with pytest.raises(ValueError, match="reserved symbol"):
        HMMParams(
            init_state_p=init_state_p,
            transition_p=transition_p,
            output_p=output_p,
            vocabulary=vocabulary,
        )


def test_the_mapping_actually_used_loads_and_decodes(backend):
    """The counterpart: `USER_BASE + c` is not merely permitted, it is correct.

    Every sequence in the corpus opens with `begin` and closes with `end`, so a
    mapping that put them on reserved fibres would make the whole corpus
    impossible rather than subtly mis-scored.
    """
    codes = load_corpus_codes()
    alphabet = corpus_alphabet()
    assert alphabet[int(codes[0])] == "begin"
    assert alphabet[int(codes[-1])] == "end"

    params = load_params(FIXTURES / "m001_0005_005.hmm")
    decoded = viterbi(params, load_corpus_record(), backend=backend)
    assert np.isfinite(decoded.total_bits)


# --- the public surface ------------------------------------------------------


def test_the_decode_is_exported_from_the_package():
    for name in ("viterbi", "ViterbiPath", "ImpossibleSequenceError"):
        assert name in pfsmgraph.hmm.__all__
        assert getattr(pfsmgraph.hmm, name) is globals()[name]


def test_the_decode_is_a_function_not_a_method():
    """ADR 0017: algorithms take parameters rather than owning them.

    The Lush decode reads no forward variable, so its placement on
    `hmm-trainer` was an artefact of where the corpus lived. Asserted so that
    "add a `.viterbi(, backend=backend)` convenience method" is a visible decision rather than a
    drift.
    """
    assert not hasattr(HMMParams, "viterbi")
    assert callable(viterbi)




# --- every backend against the reference -------------------------------------
#
# Above the line, and parameterised, because they call only the public API. The
# oracle and the constructed cases above say what a decode must be; these say
# that every backend is *the same function*, over shapes the fixtures do not
# take, which is where a translation defect lives.
#
# **Equality is exact, and that is a claim rather than a hope.** Every backend
# takes its logarithms on the host through `bits` -- numpy's `log2` -- and then
# only adds and compares, so candidates are identical bit patterns and so are the
# minima. An `abs=` tolerance would hide precisely the defect these exist to find:
# a reassociated expression that is close everywhere and wrong at a tie. **Do not
# copy this assertion into the forward pass's suite**, whose reduction is a sum
# (ADR 0020).
#
# Seeded, so a failure is reproducible from the seed on the line.


def test_every_backend_agrees_with_the_reference_on_generated_models(backend):
    """200 shapes up to nine states, empty records included, exact on every backend.

    Model 82 of this seed is the one that needed a one-ulp bound while phases 2
    and 3 still called libm's `log2`; it passes exactly now.
    """
    rng = np.random.default_rng(20260913)
    for params, rec in _generated_models(rng, 200, 9):
        ours = viterbi(params, rec, backend=backend)
        reference = viterbi(params, rec, backend="python")
        assert np.array_equal(ours.states, reference.states)
        assert ours.total_bits == reference.total_bits


@pytest.mark.parametrize("size", [1, 63, 64, 65, 130])
def test_every_backend_agrees_across_block_and_shape_boundaries(size, backend):
    """`S = 1`, and state counts either side of one CUDA block of 64 threads.

    A launch is rounded up to whole blocks, so past the last state there are
    threads the bounds check must stop: 63, 64 and 65 put the last state just
    inside, exactly at and just past a block edge, and 130 spans three. `S = 1`
    is where an off-by-one in a compiled loop shows first. Every backend runs
    them, since a boundary is where any translation goes wrong.
    """
    rng = np.random.default_rng(size)
    params = _random_model(rng, size, 4)
    rec = SequenceRecord(
        np.asarray([code(int(i)) for i in rng.integers(0, 4, 50)], dtype=np.int32)
    )
    ours = viterbi(params, rec, backend=backend)
    reference = viterbi(params, rec, backend="python")
    assert np.array_equal(ours.states, reference.states)
    assert ours.total_bits == reference.total_bits


def test_an_impossible_start_state_is_never_chosen(backend):
    """The degenerate half of the seeding defect, which the fixtures cannot show.

    Every `init_p == 0` state in the saved models also cannot emit `begin` on
    any outgoing arc, so `+inf` absorbs before the buggy `delta = 0.0` seed can
    win -- the learned topology masks it. Revision 04's `split-state` halves
    initial probabilities without halving a topology, which decouples the two,
    so the case is constructed here instead of waited for.

    State 0 is unreachable as a start (`init_p == 0`) and is otherwise the
    cheapest state to be in. Under the original's seeding it starts there,
    because a raw `init_p` of 0.0 is the *best* possible value in a domain where
    smaller is better. The kernel section below shows the same model does
    reproduce the defect when handed the original's seed, which is what makes
    this assertion mean anything.
    """
    params = _seeding_model()
    assert viterbi(params, record(0, 0, 0), backend=backend).states[0] == 1


def _seeding_model():
    init_p = np.array([0.0, 1.0])
    transition_p = np.array([[1.0, 0.0], [0.5, 0.5]])
    user_output = np.zeros((2, 2, 2))
    user_output[:, :, 0] = 1.0  # every live arc emits "a" with certainty
    return build(init_p, transition_p, user_output)


# --- kernel-level: deliberately NOT public-API -------------------------------
#
# Everything above this line is written against the public API and runs on every
# backend through the `backend` fixture. The tests below reach a kernel through
# the private `_resolve`, because what they assert is invisible from the public
# call: ADR 0003 names this case and asks for exactly this, "a separate,
# explicitly non-shared home". They are still parameterised over the table,
# because every kernel owes the contract.


@pytest.fixture(scope="module")
def kernel(backend):
    """The raw kernel function for one backend, resolved through the table."""
    return _resolve("viterbi", backend)


def test_every_kernel_reports_impossibility_numerically(kernel):
    """The backend contract: kernels do not raise, they return `+inf`.

    This is what lets ADR 0002's later phases be transliterations -- a CUDA
    device function cannot raise a Python exception, so impossibility has to
    survive the trip out as a number, and the public wrapper is what turns it
    into `ImpossibleSequenceError`. From the public call the two are
    indistinguishable, which is why this lives below the line.
    """
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    states, total = kernel(
        params.init_state_p,
        params.transition_p,
        params.output_p,
        np.array([code(0), UNK], dtype=np.int32),
    )
    assert np.isinf(total)
    assert states.shape == (3,)


def test_every_kernel_reads_the_frozen_parameter_arrays_directly(kernel):
    """ADR 0017's read-only arrays reach each kernel, and it returns integer states.

    The failure this rules out is silent in review and loud at runtime: Cython 3
    refuses to bind a mutable typed memoryview to a read-only buffer, so a
    `double[:, ::1]` where a `const double[:, ::1]` was needed compiles cleanly
    and raises "buffer source array is read-only" on every call. Asserting the
    inputs really are read-only is what stops this decaying into a restatement of
    the tests above if `HMMParams` ever stopped freezing them.
    """
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    assert not params.init_state_p.flags.writeable
    assert not params.transition_p.flags.writeable
    assert not params.output_p.flags.writeable
    states, total = kernel(
        params.init_state_p, params.transition_p, params.output_p, record(0, 1, 0).codes
    )
    assert states.dtype == STATE_DTYPE
    assert np.isfinite(total)


def test_every_kernel_reproduces_the_seeding_defect_given_the_originals_seed(kernel):
    """The counterpart of `test_an_impossible_start_state_is_never_chosen`.

    `2 ** -init_p` has `bits()` equal to `init_p` itself, so passing it to a
    kernel reproduces the original's seeding exactly -- no flag, and no second
    code path to keep alive (`HMMLIB-ACCOUNT.md` section 7). The model does
    start in state 0 then, so the public test above is ruling out a defect this
    model really exhibits.
    """
    params = _seeding_model()
    theirs, _ = kernel(
        2.0**-params.init_state_p,
        params.transition_p,
        params.output_p,
        np.array([code(0)] * 3, dtype=np.int32),
    )
    assert theirs[0] == 0, "the defect this test exists to rule out did not reproduce"


# --- phase 3 only: the CPU-parallel backend under real threads ----------------
#
# **The phase-3 failure class is a race, and a race is a property of the
# interleaving**, so it can pass on every input above and fail in production.
# The only way to sample interleavings is to run threads, which is ADR 0016's
# argument for putting this phase before CUDA. Neither test has a counterpart on
# another backend, since only this one has a thread count.


def _require(name):
    """Skip, with `backends()`'s reason, unless backend `name` can run here."""
    status = {s.name: s for s in backends("viterbi")}[name]
    if not status.available:
        pytest.skip(f"backend {name!r} unavailable: {status.reason}")


@pytest.fixture
def numba_threads():
    """`(max_threads, set_num_threads)`, restoring the thread count afterwards."""
    _require("cpu_parallel")
    numba = importlib.import_module("numba")
    original = numba.get_num_threads()
    yield numba.config.NUMBA_NUM_THREADS, numba.set_num_threads
    numba.set_num_threads(original)


def test_the_cpu_parallel_backend_is_invariant_to_thread_count(numba_threads):
    """A correct parallel kernel gives one answer; a racy one gives two.

    This is the check phase 3 exists for, and it is not a tolerance question: a
    disagreement between thread counts is a race. A racy kernel very often
    passes at one thread and fails at many, which is why this runs over
    generated models rather than one.

    Skipped rather than silently vacuous on a single-core runner: with one
    thread there is no second count to compare against.
    """
    max_threads, set_num_threads = numba_threads
    if max_threads < 2:
        pytest.skip(f"needs >= 2 threads to compare; this process has {max_threads}")

    rng = np.random.default_rng(20260910)
    models = list(_generated_models(rng, 40, 7, max_length=60))
    results = {}
    for threads in (1, max_threads):
        set_num_threads(threads)
        results[threads] = [
            viterbi(p, rec, backend="cpu_parallel") for p, rec in models
        ]
    for one, many in zip(results[1], results[max_threads]):
        assert np.array_equal(one.states, many.states)
        assert one.total_bits == many.total_bits


def test_the_cpu_parallel_backend_breaks_ties_to_the_smallest_index_at_every_thread_count(
    numba_threads,
):
    """The constructed tie, at one thread and at many.

    **This is the test that catches a `prange` on the wrong axis** at the thread
    count where it shows. Parallelising the reduction over `i` instead of the
    states `j` resolves ties in whatever order the partials combined, and a bad
    axis can still look first-wins on one thread.
    """
    max_threads, set_num_threads = numba_threads
    params = _uniform_model(5, 3)
    rec = record(*(i % 3 for i in range(24)))
    for threads in {1, max_threads}:
        set_num_threads(threads)
        decoded = viterbi(params, rec, backend="cpu_parallel")
        assert decoded.states.tolist() == [0] * 25, f"at {threads} thread(s)"


# --- phase 4 only: the CUDA backend's launch geometry -------------------------
#
# Only this backend partitions work into blocks, so only it can be wrong about
# them. The kernel module refuses to import without a device, so it is reached
# through the table, after `_require` has skipped with the reason.


@pytest.fixture
def cuda_module():
    _require("cuda")
    return importlib.import_module("pfsmgraph.hmm._viterbi_cuda")


@pytest.mark.parametrize("threads_per_block", [1, 7, 64])
def test_the_cuda_backend_is_invariant_to_block_size(monkeypatch, cuda_module, threads_per_block):
    """Phase 3's thread-count invariance, restated for a grid.

    A correct kernel gives one answer however its threads are partitioned into
    blocks. One thread per block makes every state its own block; 7 leaves a
    ragged last block for most sizes; 64 is the default. A missing or wrong
    bounds check, or a read that crosses into another thread's cell, shows up as
    a disagreement between these.
    """
    monkeypatch.setattr(cuda_module, "_THREADS_PER_BLOCK", threads_per_block)
    rng = np.random.default_rng(20260913)
    for params, rec in _generated_models(rng, 40, 12):
        ours = viterbi(params, rec, backend="cuda")
        reference = viterbi(params, rec, backend="python")
        assert np.array_equal(ours.states, reference.states)
        assert ours.total_bits == reference.total_bits


def test_the_cuda_backend_decodes_an_empty_record_without_a_launch(monkeypatch, cuda_module):
    """`N = 0` visits one state and crosses no arc, so no kernel is launched.

    Asserted by making any launch fail: the step function is replaced with one
    that cannot be indexed, so a decode that reached the device would raise.
    """
    monkeypatch.setattr(cuda_module, "_step", None)
    params = build(
        np.array([0.25, 0.75]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, SequenceRecord(np.array([], dtype=np.int32)), backend="cuda")
    assert decoded.states.tolist() == [1]
    assert decoded.total_bits == float(bits(0.75))


# --- one logarithm for every backend -----------------------------------------
#
# **The contract is numpy's `log2`, taken on the host through `bits`, for every
# arc cost and the seed; a kernel performs only `+` and `<`.** Until 2026-09-14
# phases 2 and 3 called libm's scalar `log2` instead, one ulp away from numpy's
# in roughly 0.07-0.19% of probability-shaped inputs on an AVX-512 host. Every
# test above passed against those kernels, because a one-ulp difference in an
# arc cost is usually absorbed when it is added to `delta`: 1 seeded model in
# 200 reached it, and only in `total_bits`.
#
# **What it can break is a path, not a score, and only at a tie.** `1 + bits(p)`
# rounds to steps of 2.2e-16 while `bits(p)` near zero moves in steps of about
# 1.4e-17, so two adjacent probabilities can produce *exactly* equal candidates
# in the sum. The first-wins rule then picks state 0. A kernel whose logarithm
# rounds one of them up by an ulp sees no tie and picks state 1, with the same
# `total_bits` -- which is why no bound on the total could have caught it.
#
# The pair is found at test time rather than written down, because it is a fact
# about this host's numpy. Finding an exact tie in bits is asserted: the
# addition makes one exist under any IEEE-754 double. Whether libm breaks it is
# not guaranteed, so the test that shows the fixture discriminates skips, and
# says so, on a host where the two logarithms agree on every tied candidate.
#
# Checked 2026-09-14 against the pre-fix phase-3 kernel from `10c442e`, which
# decoded the tie model to `[1, 0]` where every current backend decodes `[0, 0]`.



def _adjacent_tie_in_bits(rng, draws=20_000):
    """Two adjacent probabilities whose candidates `1 + bits(p)` tie exactly.

    Returns `(p_first, p_second, libm_breaks)`. When some drawn tie is one that
    libm's `log2` does not reproduce, that pair is returned with `libm_breaks`
    true, ordered so that libm's candidate for `p_first` is the larger -- which
    is the order in which a libm-logarithm kernel abandons state 0. Otherwise
    the first tie found is returned with `libm_breaks` false. `None` means no
    tie at all, which the addition's rounding should make impossible.
    """
    first = 0.9 + 0.1 * rng.random(draws)
    second = np.nextafter(first, 1.0)
    tied = (1.0 + bits(first)) == (1.0 + bits(second))
    fallback = None
    for a, b in zip(first[tied].tolist(), second[tied].tolist()):
        libm_a, libm_b = 1.0 + -math.log2(a), 1.0 + -math.log2(b)
        if libm_a != libm_b:
            return (a, b, True) if libm_a > libm_b else (b, a, True)
        if fallback is None:
            fallback = (a, b, False)
    return fallback


def _tie_model(p_first, p_second):
    """Two states, one symbol, one step; the candidates for state 0 are the tie.

    A uniform start makes both seeds exactly `bits(0.5) = 1.0`, under either
    logarithm. Arriving in state 0 then costs `1 + bits(p_first)` from state 0
    and `1 + bits(p_second)` from state 1. Arriving in state 1 costs more than
    4 bits either way, so the decode always ends in state 0, and the tie decides
    only where it starts.
    """
    params = build(
        [0.5, 0.5],
        [[p_first, 1.0 - p_first], [p_second, 1.0 - p_second]],
        np.ones((2, 2, 1)),
        symbols=("a",),
    )
    return params, np.array([code(0)], dtype=np.int32)


def _libm_decode(params, codes):
    """The pre-2026-09-14 arithmetic of phases 2 and 3, in plain Python.

    A decode whose every logarithm is libm's scalar `log2` (`math.log2`, the
    libc function the old Cython kernel called and numba lowered to), taken
    inside the recurrence. It exists only to show that the tie model tells the
    two logarithms apart, and is not a backend.
    """
    init, trans, out = params.init_state_p, params.transition_p, params.output_p
    size = trans.shape[0]
    delta = [-math.log2(float(p)) if p > 0 else math.inf for p in init]
    psi = []
    for symbol in codes.tolist():
        row_delta, row_psi = [], []
        for j in range(size):
            best, best_i = math.inf, 0
            for i in range(size):
                p = float(trans[i, j] * out[i, j, symbol])
                cand = delta[i] + (-math.log2(p) if p > 0 else math.inf)
                if cand < best:
                    best, best_i = cand, i
            row_delta.append(best)
            row_psi.append(best_i)
        delta, psi = row_delta, psi + [row_psi]
    states = [min(range(size), key=lambda j: (delta[j], j))]
    for row in reversed(psi):
        states.append(row[states[-1]])
    return states[::-1]


def test_the_tie_model_really_is_an_exact_tie_in_bits():
    p_first, p_second, _ = _adjacent_tie_in_bits(np.random.default_rng(20260914))
    assert p_first != p_second
    assert 1.0 + bits(p_first) == 1.0 + bits(p_second)


def test_every_backend_breaks_an_exact_tie_in_bits_to_state_zero(backend):
    """The case a backend with its own logarithm gets wrong."""
    p_first, p_second, _ = _adjacent_tie_in_bits(np.random.default_rng(20260914))
    params, codes = _tie_model(p_first, p_second)
    decoded = viterbi(params, SequenceRecord(codes), backend=backend)
    assert decoded.states.tolist() == [0, 0]
    assert decoded.total_bits == 1.0 + bits(p_first)


def test_the_tie_would_catch_a_backend_taking_its_logarithms_from_libm():
    """Without this, a green tie test could mean only that the model is easy."""
    p_first, p_second, libm_breaks = _adjacent_tie_in_bits(
        np.random.default_rng(20260914)
    )
    if not libm_breaks:
        pytest.skip(
            "numpy's and libm's log2 agree on every tied candidate drawn on this "
            f"host (numpy {np.__version__}), so there is no split for the tie to catch"
        )
    params, codes = _tie_model(p_first, p_second)
    assert _libm_decode(params, codes) == [1, 0]
    assert viterbi(params, SequenceRecord(codes), backend="python").states.tolist() == [0, 0]
