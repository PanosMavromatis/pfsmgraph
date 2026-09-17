"""Tests for ``_search``: the topology search walk (ADR 0023).

**Every test here reaches a private function**, labelled rather than hidden as ADR 0003
asks: nothing in ``_search`` is exported yet.

The walk has no oracle for its bits, since ADR 0022 changed the split and ADR 0023 the
choice of ``d``, so most of this file tests properties and constructed cases:

- **The walk's rules, against a scripted round.** ``_suggest_move`` is replaced by a
  script of ranked moves, so acceptance, patience, the dead end and the round cap are
  checked exactly and in isolation from EM. Randomised scripts check the properties
  that must hold for every sequence of totals: the best never rises, a round that does
  not improve leaves the best as the same object, the walk always follows its winner,
  and no search runs past ``max_rounds``.
- **The draw contract.** Every generator is keyed from the search's seed by role, so
  the start and each round are rebuilt from an independent ``SeedSequence`` and
  compared.
- **The start.** The one-state model's draws, and that its converged parameters do not
  depend on them.
- **One real search**, which is also the one oracle there is: its first three rounds
  choose the same moves as the Lush training log of ``m001_0005_005``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, baum_welch
from pfsmgraph.hmm import _search as search_module
from pfsmgraph.hmm._mdl import _model_description_length, _scan_d
from pfsmgraph.hmm._numeric import rand_p_vector
from pfsmgraph.hmm._search import (
    SPLIT_MIN_CYCLES,
    START_NOISE_WIDTH,
    _COLUMNS,
    Round,
    SearchResult,
    _log_start_row,
    _marker,
    _one_state_model,
    _search,
)
from pfsmgraph.hmm._trials import Move, TrialResult

from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params

SEED = 20260917
N_USER = 3
VOCABULARY = SymbolTable([f"s{i}" for i in range(N_USER)])


def _corpus(seed=SEED):
    rng = np.random.default_rng(seed)
    return [SequenceRecord(rng.integers(USER_BASE, USER_BASE + N_USER, n)) for n in (40, 25)]


# ---------------------------------------------------------------------------
# The walk's rules, against a scripted round.


class _Script:
    """Stands in for ``_suggest_move``: returns scripted ranked moves and records each call.

    ``rounds[r]`` is a list of totals for round ``r``; ``None`` is a move with no result.
    Each move with a result carries a fresh sentinel as its params, so identity shows
    which model the walk moved to.
    """

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    def __call__(self, params, records, *, seed, **keywords):
        index = len(self.calls)
        self.calls.append((params, seed, keywords))
        moves = []
        for trial, total in enumerate(self.rounds[index]):
            if total is None:
                moves.append(Move("split", (0,), trial, None, np.inf))
            else:
                result = TrialResult(object(), 1.0, total, total, 0, True)
                moves.append(Move("split", (0,), trial, result, total))
        return sorted(moves, key=lambda move: move.total_bits)


@pytest.fixture(scope="module")
def start_total():
    records = _corpus()
    return _search(records, start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=0, patience=0)


def _run(monkeypatch, rounds, *, max_rounds, patience):
    script = _Script(rounds)
    monkeypatch.setattr(search_module, "_suggest_move", script)
    # Cython for speed alone: its scores are bit-identical to numpy's (ADR 0023 section 7).
    result = _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED),
        max_rounds=max_rounds, patience=patience, backend="cython", score_backend="cython",
    )
    return result, script


def test_max_rounds_zero_scores_the_start_alone(start_total):
    assert start_total.rounds == ()
    assert start_total.stop == "max_rounds"
    assert start_total.best is start_total.start
    assert start_total.start.params.n_states == 1


def test_the_walk_follows_each_winner_even_when_it_is_worse(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, script = _run(monkeypatch, [[s - 10], [s - 5], [s - 20]], max_rounds=3, patience=5)
    winners = [round_.move.result for round_ in result.rounds]
    # Round r + 1 is proposed from round r's winner, whatever its total was.
    assert [call[0] for call in script.calls[1:]] == [winner.params for winner in winners[:2]]
    assert script.calls[0][0] is result.start.params
    assert [round_.improved for round_ in result.rounds] == [True, False, True]
    assert result.best is winners[2]
    assert result.stop == "max_rounds"


def test_the_winner_is_the_first_move_with_a_finite_total(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[None, None, s - 1, s - 2]], max_rounds=1, patience=0)
    (round_,) = result.rounds
    assert round_.move.total_bits == s - 2
    assert round_.move.result is result.best


def test_a_result_scoring_inf_is_not_a_winner(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[np.inf, None, s + 3]], max_rounds=1, patience=0)
    assert result.rounds[0].move.total_bits == s + 3


@pytest.mark.parametrize("round_", [[], [None], [None, np.inf], [np.inf]])
def test_a_round_with_no_finite_move_is_a_dead_end_and_records_no_round(monkeypatch, start_total, round_):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[s - 1], round_], max_rounds=5, patience=5)
    assert result.stop == "dead_end"
    assert len(result.rounds) == 1
    assert result.best is result.rounds[0].move.result


def test_a_tie_with_the_best_is_not_an_improvement(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, _ = _run(monkeypatch, [[s]], max_rounds=1, patience=0)
    assert result.rounds[0].improved is False
    assert result.best is result.start
    assert result.stop == "patience"


def test_patience_zero_is_greedy_and_stops_at_the_first_round_that_does_not_improve(monkeypatch, start_total):
    s = start_total.start.total_bits
    result, script = _run(monkeypatch, [[s - 1], [s - 2], [s + 1], [s - 9]], max_rounds=10, patience=0)
    assert len(script.calls) == 3
    assert result.stop == "patience"
    assert result.best.total_bits == s - 2


@pytest.mark.parametrize("patience", [1, 2, 3])
def test_patience_tolerates_that_many_rounds_without_a_new_best(monkeypatch, start_total, patience):
    s = start_total.start.total_bits
    rounds = [[s - 1]] + [[s + k] for k in range(1, 10)]
    result, script = _run(monkeypatch, rounds, max_rounds=10, patience=patience)
    assert len(script.calls) == 1 + patience + 1
    assert result.stop == "patience"


def test_a_new_best_resets_patience(monkeypatch, start_total):
    s = start_total.start.total_bits
    rounds = [[s + 1], [s - 1], [s + 1], [s - 2], [s + 1], [s + 1]]
    result, script = _run(monkeypatch, rounds, max_rounds=6, patience=1)
    assert len(script.calls) == 6
    assert result.stop == "patience"
    assert result.best.total_bits == s - 2


def test_randomised_walks_keep_every_invariant(monkeypatch, start_total):
    s = start_total.start.total_bits
    rng = np.random.default_rng(SEED)
    for _ in range(300):
        max_rounds = int(rng.integers(0, 12))
        patience = int(rng.integers(0, 4))
        rounds = []
        for _ in range(max_rounds):
            size = int(rng.integers(0, 4))
            totals = [
                None if rng.random() < 0.15 else float(s + rng.integers(-6, 7)) for _ in range(size)
            ]
            rounds.append(totals)
        result, script = _run(monkeypatch, rounds, max_rounds=max_rounds, patience=patience)

        assert isinstance(result, SearchResult)
        assert len(result.rounds) <= max_rounds
        assert [r.index for r in result.rounds] == list(range(len(result.rounds)))
        best = result.start
        without = 0
        for round_ in result.rounds:
            assert isinstance(round_, Round)
            ranked = sorted(t for t in rounds[round_.index] if t is not None)
            assert round_.move.total_bits == ranked[0]
            assert round_.improved == (round_.move.total_bits < best.total_bits)
            if round_.improved:
                best, without = round_.move.result, 0
            else:
                without += 1
            assert without <= patience or round_ is result.rounds[-1]
        # The best never rises, and a round that did not improve left it the same object.
        assert result.best is best
        assert result.best.total_bits <= result.start.total_bits
        if result.stop == "dead_end":
            assert not any(t is not None for t in rounds[len(result.rounds)])
        elif result.stop == "patience":
            assert without == patience + 1
        else:
            assert len(result.rounds) == max_rounds


def test_every_option_reaches_every_round(monkeypatch, start_total):
    s = start_total.start.total_bits
    script = _Script([[s - 1], [s - 2]])
    monkeypatch.setattr(search_module, "_suggest_move", script)
    _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=2, patience=0,
        split_min_cycles=7, merge_min_cycles=3, min_split_trials=5, split_trials_per_state_p=1.5,
        backend="python", score_backend="python", batch_size=4,
    )
    expected = dict(
        split_min_cycles=7, merge_min_cycles=3, min_split_trials=5, split_trials_per_state_p=1.5,
        backend="python", score_backend="python", batch_size=4,
    )
    assert [call[2] for call in script.calls] == [expected, expected]


def test_the_default_split_floor_is_adr_0023s(monkeypatch, start_total):
    s = start_total.start.total_bits
    script = _Script([[s - 1]])
    monkeypatch.setattr(search_module, "_suggest_move", script)
    _search(_corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=1, patience=0)
    assert SPLIT_MIN_CYCLES == 200
    assert script.calls[0][2]["split_min_cycles"] == 200
    assert script.calls[0][2]["merge_min_cycles"] == 0


# ---------------------------------------------------------------------------
# The draw contract.


@pytest.mark.parametrize("spawn_key", [(), (4,), (2, 9)])
def test_round_r_is_seeded_from_the_role_key_under_the_searchs_own(monkeypatch, start_total, spawn_key):
    s = start_total.start.total_bits
    script = _Script([[s - 1], [s + 1], [s + 2]])
    monkeypatch.setattr(search_module, "_suggest_move", script)
    seed = np.random.SeedSequence(SEED, spawn_key=spawn_key)
    _search(_corpus(), start=VOCABULARY, seed=seed, max_rounds=3, patience=5)
    for index, (_, round_seed, _) in enumerate(script.calls):
        assert round_seed.entropy == SEED
        assert round_seed.spawn_key == spawn_key + (1, index)


def test_the_one_state_start_draws_from_the_start_key_in_init_random_order():
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,)))
    params = _one_state_model(VOCABULARY, np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,))))

    init = rand_p_vector(1, START_NOISE_WIDTH, rng)
    transition = rand_p_vector(1, START_NOISE_WIDTH, rng)
    fibre = rand_p_vector(N_USER, START_NOISE_WIDTH, rng)
    assert params.init_state_p.tobytes() == init.tobytes()
    assert params.transition_p.tobytes() == transition.reshape(1, 1).tobytes()
    assert params.output_p[0, 0, USER_BASE:].tobytes() == fibre.tobytes()
    assert not params.output_p[0, 0, :USER_BASE].any()
    assert params.vocabulary is VOCABULARY
    # The one-element vectors still draw: skipping them would shift the fibre.
    skipped = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(0,)))
    assert rand_p_vector(N_USER, START_NOISE_WIDTH, skipped).tobytes() != fibre.tobytes()


def test_a_search_is_reproducible_from_its_seed():
    records = _corpus()
    runs = [
        _search(
            records, start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=2, patience=2,
            backend="cython", score_backend="cython",
        )
        for _ in range(2)
    ]
    for a, b in zip(runs[0].rounds, runs[1].rounds):
        assert (a.move.kind, a.move.states, a.move.trial, a.move.total_bits) == (
            b.move.kind, b.move.states, b.move.trial, b.move.total_bits,
        )
        assert a.move.result.params.transition_p.tobytes() == b.move.result.params.transition_p.tobytes()
    assert runs[0].best.total_bits == runs[1].best.total_bits


# ---------------------------------------------------------------------------
# The start.


def test_the_one_state_start_converges_to_the_same_bits_from_any_draw():
    """With one state the E-step counts do not depend on the parameters (ADR 0023)."""
    records = _corpus()
    starts = [
        _search(records, start=VOCABULARY, seed=np.random.SeedSequence(entropy), max_rounds=0, patience=0).start
        for entropy in (1, 2, 3)
    ]
    first = starts[0].params
    assert not np.array_equal(
        _one_state_model(VOCABULARY, np.random.default_rng(1)).output_p,
        _one_state_model(VOCABULARY, np.random.default_rng(2)).output_p,
    )
    for other in starts[1:]:
        assert other.params.output_p.tobytes() == first.output_p.tobytes()
        assert other.total_bits == starts[0].total_bits


def test_a_given_start_is_converged_under_the_default_rule_and_scored_by_the_scan():
    rng = np.random.default_rng(SEED)
    output = np.zeros((2, 2, USER_BASE + N_USER))
    output[:, :, USER_BASE:] = rng.dirichlet(np.ones(N_USER), size=(2, 2))
    params = HMMParams(rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(2), size=2), output, VOCABULARY)
    records = _corpus()

    result = _search(records, start=params, seed=np.random.SeedSequence(SEED), max_rounds=0, patience=0)
    expected = baum_welch(params, records)
    assert result.start.params.output_p.tobytes() == expected.params.output_p.tobytes()
    assert result.start.cycles == expected.cycles
    assert result.start.data_bits == expected.description_lengths[-1]
    assert (result.start.d, result.start.total_bits) == _scan_d(
        expected.params, records, expected.description_lengths[-1]
    )


@pytest.mark.parametrize(
    ("keywords", "error", "match"),
    [
        (dict(start="a vocabulary"), TypeError, "start"),
        (dict(seed=7), TypeError, "SeedSequence"),
        (dict(max_rounds=2.0), TypeError, "max_rounds"),
        (dict(patience=True), TypeError, "patience"),
        (dict(max_rounds=-1), ValueError, "max_rounds"),
        (dict(patience=-1), ValueError, "patience"),
    ],
)
def test_the_search_rejects_bad_arguments_before_any_work(keywords, error, match):
    arguments = dict(start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=1, patience=0)
    arguments.update(keywords)
    with pytest.raises(error, match=match):
        _search(_corpus(), **arguments)


def test_max_rounds_and_patience_are_required():
    with pytest.raises(TypeError):
        _search(_corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), patience=0)
    with pytest.raises(TypeError):
        _search(_corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=0)


# ---------------------------------------------------------------------------
# One real search.


def test_the_first_rounds_choose_the_moves_the_lush_training_log_records():
    """The only oracle for a trajectory, and an oracle for moves rather than bits.

    `m001_0005_005`'s log (`_training_log`) shows the original growing the one-state
    model by splitting state 0, then 0, then 2, to totals printed at `%g` as 2433.99,
    2131.1 and 1774.03. ADR 0022 changed the split's draws and ADR 0023 the choice of
    `d`, so the candidates' bits cannot match; the winning split in each round beats its
    runners-up by more than those changes move a total, so the moves do, and the totals
    land within a bit of the log.
    """
    records = [load_corpus_record()]
    vocabulary = load_params(FIXTURES / "m001_0001_001.hmm").vocabulary
    result = _search(
        records, start=vocabulary, seed=np.random.SeedSequence(SEED), max_rounds=3, patience=0,
        backend="cython", score_backend="cython",
    )
    assert [(r.move.kind, r.move.states) for r in result.rounds] == [
        ("split", (0,)), ("split", (0,)), ("split", (2,)),
    ]
    logged = [2433.99, 2131.1, 1774.03]
    for round_, total in zip(result.rounds, logged):
        assert abs(round_.move.total_bits - total) < 1.0
        assert round_.improved
    assert result.start.d == 13.0
    assert result.best is result.rounds[-1].move.result
    assert result.best.params.n_states == 4


# ---------------------------------------------------------------------------
# The check's phase (ADR 0024 section 2).


def _assert_same_trial(ours, reference):
    for got, want in zip(
        (ours.params.init_state_p, ours.params.transition_p, ours.params.output_p),
        (reference.params.init_state_p, reference.params.transition_p, reference.params.output_p),
    ):
        assert got.tobytes() == want.tobytes()
    assert (ours.d, ours.total_bits, ours.data_bits, ours.cycles, ours.converged) == (
        reference.d, reference.total_bits, reference.data_bits, reference.cycles,
        reference.converged,
    )


def _two_rounds(score_backend):
    return _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=2, patience=2,
        backend="cython", score_backend=score_backend,
    )


@pytest.fixture(scope="module")
def reference_search():
    return _two_rounds("python")


def test_the_check_phase_changes_no_search(reference_search, score_backend):
    ours = _two_rounds(score_backend)
    assert ours.stop == reference_search.stop
    _assert_same_trial(ours.start, reference_search.start)
    _assert_same_trial(ours.best, reference_search.best)
    assert len(ours.rounds) == len(reference_search.rounds) >= 1
    for a, b in zip(ours.rounds, reference_search.rounds):
        assert (a.index, a.improved, a.move.kind, a.move.states, a.move.trial, a.move.total_bits) == (
            b.index, b.improved, b.move.kind, b.move.states, b.move.trial, b.move.total_bits,
        )
        _assert_same_trial(a.move.result, b.move.result)


def test_the_start_passes_score_backend_to_baum_welch(monkeypatch):
    # Equality above cannot see a dropped keyword, since every phase gives the same bits.
    seen = []

    def spy(params, records, **keywords):
        seen.append(keywords)
        return baum_welch(params, records, **keywords)

    monkeypatch.setattr(search_module, "baum_welch", spy)
    _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED), max_rounds=0, patience=0,
        score_backend="cython",
    )
    assert [keywords.get("score_backend") for keywords in seen] == ["cython"]


# ---------------------------------------------------------------------------
# The log's format.
#
# **The oracle is the layout, never the values.** Line 1 of all three tracked
# ``_training_log`` files carries ``d = 29``, Brent's local minimum, where ADR 0023
# section 5's bounded scan returns 13 on the same one-state model -- 10.46 bits cheaper.
# A test pinning printed numbers would pin a defect this project deliberately fixed.


def _oracle_rows():
    """Every line of the three tracked ``_training_log`` files, parsed.

    Yields ``(name, raw, size, marker, numbers)``, the marker rendered in the form
    :func:`~._search._marker` produces so the two compare directly.
    """
    for name in SAVED_MODELS:
        for raw in (FIXTURES / name / "_training_log").read_text().splitlines():
            fields = raw.split()
            size = int(fields[0])
            if fields[1] == "-":
                yield name, raw, size, "-", fields[2:]
            elif fields[2] == "^":
                yield name, raw, size, f"{int(fields[1])} ^", fields[3:]
            else:
                yield name, raw, size, f"{int(fields[1])} v {int(fields[3])}", fields[4:]


def _log_of(**keywords):
    """Run a search with a log attached, and return ``(result, text)``."""
    log = io.StringIO()
    result = _search(
        _corpus(), seed=np.random.SeedSequence(SEED), backend="cython",
        score_backend="cython", log=log, **keywords,
    )
    return result, log.getvalue()


def _data_rows(text):
    """The rows between the header and the closing sentence."""
    lines = text.splitlines()
    assert lines[-1].lstrip().startswith("stopped:"), lines[-1]
    return lines[1:-1]


@pytest.fixture(scope="module")
def two_state_params():
    """A real two-state model, so a constructed merge has something coherent to merge."""
    result = _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED),
        max_rounds=1, patience=1, backend="cython", score_backend="cython",
    )
    assert result.rounds[0].move.kind == "split"
    return result.rounds[0].move.result.params


def test_the_oracle_identifies_its_own_column_order():
    """The original's field order is recoverable from the files, not only from its source.

    ``data-dl + model-dl == total-dl`` says which three of the numeric fields are which
    and in what order, which is what makes the column order an *oracle* rather than a
    transcription of ``training-log-line``. **The tolerance is measured, not guessed**:
    ``%g`` rounds each field to six significant digits independently, so the printed
    halves miss their printed total by up to 3.4e-6 relative, and ``rel=1e-6`` fails on
    four of the seven lines. Same shape as the four-decimal model fixtures, where
    ``5e-5`` is exactly attainable and fails by less than an ulp.
    """
    rows = list(_oracle_rows())
    assert len(rows) == 7
    for name, raw, size, _marker_text, numbers in rows:
        data, model, total, d, test_data = (float(value) for value in numbers)
        assert size >= 1, raw
        assert data + model == pytest.approx(total, rel=1e-5), (name, raw)
        assert d == int(d) and d >= 1, raw
        # ``test-data-dl``, the column the port drops: never assigned by the original.
        assert test_data == 0.0, raw


def test_the_ported_columns_follow_the_original_order():
    """``_COLUMNS`` carries ``training-log-line``'s fields in its order, less the dropped one."""
    labels = [label for label, _, _ in _COLUMNS]
    assert labels[1:7] == ["Size", "Move", "Data DL", "Model DL", "Total DL", "d"]
    # The four with no counterpart: the original's user ran one trial at a time by
    # hand, so no round of it ranked candidates or lost to a runner-up.
    assert labels[0] == "Round"
    assert labels[7:] == ["Candidates", "Runner-Up", "Comments"]


def test_the_marker_forms_are_the_original_s_three():
    """``n ^``, ``i v j``, ``-`` -- and the oracles exhibit only two of them."""
    assert _marker(Move("split", (7,), 0, None, 0.0)) == "7 ^"
    assert _marker(Move("merge", (2, 5), 0, None, 0.0)) == "2 v 5"
    # No merge was ever logged, which is why the merge row below is constructed.
    assert {row[3] for row in _oracle_rows()} == {"-", "0 ^", "2 ^"}


def test_a_start_row_precedes_every_move():
    result, text = _log_of(start=VOCABULARY, max_rounds=2, patience=2)
    rows = _data_rows(text)
    assert len(rows) == len(result.rounds) + 1
    start, *moves = rows
    assert start.split()[2] == "-"
    for row in moves:
        # A split row reads ``n ^`` and a merge ``i v j``, so field 3 is the operator.
        assert row.split()[3] in {"^", "v"}, row


def test_the_round_column_is_the_row_number_not_the_round_index():
    """Deliberately out of step, and pinned so it stays deliberate.

    The column counts the walk's rows, of which 0 is the starting model.
    ``SearchResult.rounds[r].index`` stays ``r`` because it is also the round's seed
    component -- round ``r`` draws from ``spawn_key + (1, r)`` (ADR 0023 section 2) --
    so renumbering it to match the column would silently re-seed every search.
    """
    result, text = _log_of(start=VOCABULARY, max_rounds=2, patience=2)
    columns = [int(row.split()[0]) for row in _data_rows(text)]
    assert columns == [0, *[round_.index + 1 for round_ in result.rounds]]
    assert [round_.index for round_ in result.rounds] == list(range(len(result.rounds)))


def test_a_merge_row_uses_the_original_s_i_v_j_form(monkeypatch, start_total, two_state_params):
    """The case no oracle has: every logged move in all three files is a split.

    The merge is scripted rather than searched for, because whether this corpus ever
    accepts one is a fact about the corpus and this is a test about the format.
    """
    one_state = start_total.start
    monkeypatch.setattr(
        search_module, "_suggest_move",
        lambda params, records, **keywords: [
            Move("merge", (0, 1), 0, one_state, one_state.total_bits)
        ],
    )
    _, text = _log_of(start=two_state_params, max_rounds=1, patience=1)
    rows = _data_rows(text)
    assert len(rows) == 2
    assert rows[0].split()[2] == "-"
    # Round 1, one state after the merge, and the original's ``i v j``.
    assert rows[1].split()[:5] == ["1", "1", "0", "v", "1"]


def test_a_move_row_places_its_columns_in_the_original_s_order(
    monkeypatch, start_total, two_state_params
):
    """The move row's own columns, pinned separately from the start row's.

    **Found by mutation.** Swapping ``Data DL`` and ``Model DL`` inside
    ``_log_move_row`` survived all nine tests here, because the start row and the move
    row are laid out by two separate calls and only the first was pinned. Pinning one
    row says nothing about the other.
    """
    one_state = start_total.start
    monkeypatch.setattr(
        search_module, "_suggest_move",
        lambda params, records, **keywords: [
            Move("merge", (0, 1), 0, one_state, one_state.total_bits)
        ],
    )
    _, text = _log_of(start=two_state_params, max_rounds=1, patience=1)
    fields = _data_rows(text)[1].split()
    model_bits = _model_description_length(one_state.params, one_state.d)
    assert fields[5:9] == [
        f"{one_state.total_bits - model_bits:g}",
        f"{model_bits:g}",
        f"{one_state.total_bits:g}",
        f"{one_state.d:g}",
    ]
    # Candidates, then a runner-up column reading ``-`` because the round had only one
    # candidate, then an empty Comments the row's rstrip removes.
    assert fields[9:] == ["1", "-"]


def test_the_runner_up_column_is_what_the_round_gave_up(monkeypatch, start_total, two_state_params):
    """The margin is the runner-up's total *minus* the winner's, so it is never negative.

    **Found by mutation.** Negating the subtraction survived every other test here,
    because the only scripted round had a single candidate and so printed ``-`` rather
    than a number. A sign is only pinned by a case that has one.
    """
    winner = start_total.start
    runner_up = TrialResult(
        winner.params, winner.d, winner.total_bits + 12.5,
        winner.data_bits, winner.cycles, winner.converged,
    )
    monkeypatch.setattr(
        search_module, "_suggest_move",
        lambda params, records, **keywords: [  # ranked, as ``_suggest_move`` returns them
            Move("merge", (0, 1), 0, winner, winner.total_bits),
            Move("split", (0,), 0, runner_up, runner_up.total_bits),
        ],
    )
    _, text = _log_of(start=two_state_params, max_rounds=1, patience=1)
    fields = _data_rows(text)[1].split()
    assert fields[9:] == ["2", "+12.5"]


def test_the_printed_halves_are_taken_at_d_and_not_from_data_bits(start_total):
    """The decomposition trap, pinned on a case where it is 2.5 bits wide.

    ``TrialResult.data_bits`` is the corpus length of the *unrounded* parameters, which
    ``d`` never touched, so printing it beside ``total - data_bits`` would push the
    whole quantization gap into the model column.
    """
    result = start_total.start
    log = io.StringIO()
    _log_start_row(log, result)
    data_text, model_text, total_text = log.getvalue().split()[3:6]
    model_bits = _model_description_length(result.params, result.d)
    # Compared as *text*, so there is no tolerance to get wrong. The oracle test above
    # needs one because it compares the original's independently rounded fields to each
    # other; here the source values are known, so ``%g`` of them is the exact expectation.
    assert model_text == f"{model_bits:g}"
    assert data_text == f"{result.total_bits - model_bits:g}"
    assert total_text == f"{result.total_bits:g}"
    # The trap: the unrounded corpus length is 2.57 bits away on this corpus, and 1.5 to
    # 29 bits away on the three tracked models.
    assert abs(float(data_text) - result.data_bits) > 1.0


def test_the_nested_convergence_log_is_silenced():
    """``baum_welch``'s ``log=`` is never passed down, so no EM block reaches this stream.

    Checked on the text rather than by a spy, because the EM runs that matter are
    ``_trials``', not the one call this module makes.
    """
    _, text = _log_of(start=VOCABULARY, max_rounds=1, patience=1)
    for fragment in ("cycle", "change", "quiet", "converged after", "max_cycles"):
        assert fragment not in text, fragment


def test_no_log_writes_nothing():
    """``log=None`` is the default and reports nothing at all."""
    result = _search(
        _corpus(), start=VOCABULARY, seed=np.random.SeedSequence(SEED),
        max_rounds=1, patience=1, backend="cython", score_backend="cython",
    )
    assert len(result.rounds) == 1
