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
"""

from __future__ import annotations

import itertools

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
    viterbi,
)
from pfsmgraph.hmm._numeric import bits
from pfsmgraph.hmm._viterbi import STATE_DTYPE, _viterbi

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


# --- the differential decode ------------------------------------------------


class Oracle:
    """One saved model, its saved decode, and our decode of the same corpus."""

    def __init__(self, model):
        self.model = model
        self.params = load_params(FIXTURES / model)
        self.record = load_corpus_record()
        self.symbols, self.states, self.entropies = load_vpath(model)
        self.path = viterbi(self.params, self.record)


@pytest.fixture(scope="module", params=SAVED_MODELS)
def oracle(request):
    """Decode the 1269-symbol corpus once per model, not once per test."""
    return Oracle(request.param)


def test_the_corpus_reproduces_each_saved_decodes_output_column(oracle):
    """Without this the `States` column is aligned to nothing.

    `fprop-all` decoded the 200 `.seq` files as one stream, which is why a
    single `.vpath.xls` covers all of them. Numeric concatenation order is an
    assumption until this passes.
    """
    alphabet = corpus_alphabet()
    expected = [alphabet[int(c)] for c in load_corpus_codes()]
    assert oracle.symbols[1:] == expected


def test_the_saved_decode_has_one_more_row_than_it_has_symbols(oracle):
    """ADR 0015's N+1 geometry, printed to disk by the original's own author.

    The leading row's `Output` is a literal `-`: no symbol has been emitted yet,
    because emission happens while *crossing* an arc.
    """
    assert oracle.symbols[0] == "-"
    assert len(oracle.states) == oracle.record.length + 1


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


# --- the seeding defect, in both directions ---------------------------------


def test_the_seed_agrees_wherever_the_defect_cannot_bite():
    """Two of the three models start in the same state either way.

    Recorded so that the divergence below reads as one specific measured
    disagreement rather than as a general disclaimer.
    """
    for model in ("m001_0001_001.hmm", "m001_0005_005.hmm"):
        oracle = Oracle(model)
        assert oracle.path.states[0] == oracle.states[0]


def test_the_seeding_fix_diverges_at_exactly_one_position():
    """`m008_0001_008` position 0, and nothing else in 3807 positions."""
    oracle = Oracle("m008_0001_008.hmm")
    disagreements = np.flatnonzero(oracle.path.states != oracle.states)
    assert disagreements.tolist() == [0]


def test_the_divergence_prefers_the_more_probable_start_state():
    """The defect decides *backwards*, and this is the assertion that says so.

    The original seeds delta with a raw probability into a bit-domain
    accumulator, where smaller is better -- so it prefers the *less* likely
    start whenever the seed is what decides. Here the two live starts are 0.3665
    and 0.6335 and their best outgoing arcs differ by 0.004 bits, so the seed
    decides it alone.
    """
    oracle = Oracle("m008_0001_008.hmm")
    ours = int(oracle.path.states[0])
    theirs = int(oracle.states[0])
    init_p = oracle.params.init_state_p
    assert ours != theirs
    assert init_p[ours] > init_p[theirs]


def test_an_impossible_start_state_is_never_chosen():
    """The degenerate half of the defect, which the tracked fixtures cannot show.

    Every `init_p == 0` state in the saved models also cannot emit `begin` on
    any outgoing arc, so `+inf` absorbs before the buggy `delta = 0.0` seed can
    win -- the learned topology masks it. Revision 04's `split-state` halves
    initial probabilities without halving a topology, which decouples the two,
    so the case is constructed here instead of waited for.

    State 0 is unreachable as a start (`init_p == 0`) and is otherwise the
    cheapest state to be in. Under the original's seeding it starts there,
    because a raw `init_p` of 0.0 is the *best* possible value in a domain where
    smaller is better.
    """
    init_p = np.array([0.0, 1.0])
    transition_p = np.array([[1.0, 0.0], [0.5, 0.5]])
    user_output = np.zeros((2, 2, 2))
    user_output[:, :, 0] = 1.0  # every live arc emits "a" with certainty
    params = build(init_p, transition_p, user_output)

    ours = viterbi(params, record(0, 0, 0))
    assert ours.states[0] == 1

    # 2 ** -init_p has bits() equal to init_p itself, so passing it to the
    # kernel reproduces the original's seeding exactly -- no flag, and no second
    # code path to keep alive. See HMMLIB-ACCOUNT.md section 7.
    theirs, _ = _viterbi(
        2.0**-init_p, params.transition_p, params.output_p, ours.states.dtype.type(
            [code(0)] * 3
        ),
    )
    assert theirs[0] == 0, "the defect this test exists to rule out did not reproduce"


# --- min-sum, not max-product -----------------------------------------------


def test_the_decode_returns_the_cheapest_path_of_all():
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
    decoded = viterbi(params, SequenceRecord(np.asarray(codes, dtype=np.int32)))

    best = min(
        path_cost(params, codes, states)
        for states in itertools.product(range(3), repeat=len(codes) + 1)
    )
    assert decoded.total_bits == pytest.approx(best)


def test_ties_are_broken_toward_the_lower_state_index():
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
    """
    size = 3
    uniform_transition = np.full((size, size), 1.0 / size)
    uniform_output = np.full((size, size, 2), 0.5)
    params = build(np.full(size, 1.0 / size), uniform_transition, uniform_output)

    decoded = viterbi(params, record(0, 1, 0, 1))
    assert decoded.states.tolist() == [0, 0, 0, 0, 0]


def test_the_most_probable_path_is_the_one_with_fewest_bits():
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
    assert viterbi(params, record(0)).total_bits == pytest.approx(staying)


# --- arc emission ------------------------------------------------------------


def test_the_emission_depends_on_the_destination_as_well_as_the_source():
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

    assert viterbi(params, record(0)).states.tolist() == [0, 1]
    assert viterbi(params, record(1)).states.tolist() == [0, 2]


# --- the N+1 geometry --------------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, 2, 5, 17])
def test_a_path_over_n_symbols_visits_n_plus_one_states(length):
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, record(*([0] * length)))
    assert decoded.states.shape == (length + 1,)
    assert decoded.n_symbols == length


def test_an_empty_record_still_visits_one_state():
    """`N = 0` is a path of one state and no arcs, not an error.

    The cost is then the seed alone, which is the cleanest available statement
    that `delta[0]` is in the bit domain.
    """
    params = build(
        np.array([0.25, 0.75]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, SequenceRecord(np.array([], dtype=np.int32)))
    assert decoded.states.tolist() == [1]
    assert decoded.total_bits == pytest.approx(float(bits(0.75)))


# --- impossibility -----------------------------------------------------------


def test_a_record_carrying_unk_is_impossible():
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
        viterbi(params, SequenceRecord(codes))
    assert "symbol 1 of the record" in str(excinfo.value)
    assert f"code {UNK}" in str(excinfo.value)


def test_a_record_carrying_pad_is_impossible():
    """`PAD` is code 0, and a record never holds padding -- but nothing stops a
    caller building one by hand, and zero is the value an uninitialized buffer
    has."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ImpossibleSequenceError):
        viterbi(params, SequenceRecord(np.array([code(0), PAD], dtype=np.int32)))


def test_an_unseen_bigram_is_impossible():
    """The other route to impossibility: a live symbol on a dead arc."""
    transition_p = np.array([[1.0, 0.0], [0.0, 1.0]])  # no arc between them
    user_output = np.zeros((2, 2, 2))
    user_output[0, 0, 0] = 1.0  # state 0 loops emitting "a"
    user_output[1, 1, 1] = 1.0  # state 1 loops emitting "b"
    params = build(np.array([1.0, 0.0]), transition_p, user_output)

    assert viterbi(params, record(0, 0)).states.tolist() == [0, 0, 0]
    with pytest.raises(ImpossibleSequenceError) as excinfo:
        viterbi(params, record(0, 1))
    assert "symbol 1 of the record" in str(excinfo.value)


def test_the_impossible_error_is_a_valueerror():
    """`except ValueError` keeps working; the distinct type is for revision 04.

    Its topology search decodes many sequences against many candidate
    topologies, where an impossible sequence is an ordinary search outcome
    rather than a malformed input -- so it must be catchable by type and not by
    message.
    """
    assert issubclass(ImpossibleSequenceError, ValueError)


def test_no_path_with_an_infinite_total_ever_reaches_a_caller():
    """The guarantee `ViterbiPath.total_bits` documents."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ImpossibleSequenceError):
        viterbi(params, SequenceRecord(np.array([UNK], dtype=np.int32)))


def test_the_kernel_itself_reports_impossibility_numerically():
    """The backend contract: kernels do not raise, they return `+inf`.

    Asserted directly because it is what lets ADR 0002's later phases be
    transliterations -- a CUDA device function cannot raise a Python exception.
    """
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    states, total = _viterbi(
        params.init_state_p,
        params.transition_p,
        params.output_p,
        np.array([UNK], dtype=np.int32),
    )
    assert np.isinf(total)
    assert states.shape == (2,)


# --- the symbol-axis guard ---------------------------------------------------


def test_a_code_past_the_end_of_the_symbol_axis_is_refused():
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ValueError, match="outside the model's symbol axis"):
        viterbi(params, SequenceRecord(np.array([code(0), 99], dtype=np.int32)))


def test_a_negative_code_is_refused():
    """Not reachable through `encode`, and exactly what the rejected
    user-symbols-only axis would have produced: `1 - USER_BASE` is `-5`, which
    numpy indexes without complaint."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ValueError, match="outside the model's symbol axis"):
        viterbi(params, SequenceRecord(np.array([-5], dtype=np.int32)))


def test_the_range_error_names_the_likely_cause():
    """A vocabulary mismatch, which is what this almost always is."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    with pytest.raises(ValueError) as excinfo:
        viterbi(params, SequenceRecord(np.array([99], dtype=np.int32)))
    assert "different vocabulary" in str(excinfo.value)


# --- ViterbiPath -------------------------------------------------------------


def test_the_states_array_is_read_only():
    """Same reasoning as `HMMParams`: a result handed out mutable is a result
    that can be corrupted after the fact."""
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    decoded = viterbi(params, record(0, 1))
    assert not decoded.states.flags.writeable
    with pytest.raises(ValueError):
        decoded.states[0] = 1


def test_the_states_are_integers_not_floats():
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
    assert viterbi(params, record(0)).states.dtype == STATE_DTYPE
    assert np.issubdtype(STATE_DTYPE, np.integer)


def test_the_label_is_carried_through():
    params = build(
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.full((2, 2, 2), 0.5),
    )
    assert viterbi(params, record(0, label="utterance-7")).label == "utterance-7"
    assert viterbi(params, record(0)).label is None


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


def test_the_mapping_actually_used_loads_and_decodes():
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
    decoded = viterbi(params, load_corpus_record())
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
    "add a `.viterbi()` convenience method" is a visible decision rather than a
    drift.
    """
    assert not hasattr(HMMParams, "viterbi")
    assert callable(viterbi)
