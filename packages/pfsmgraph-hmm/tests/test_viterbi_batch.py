"""`viterbi_batch`: many records in one padded decode, held to `viterbi` bit for bit.

The contract is short and exact. Row `i` of the result is `viterbi(params,
records[i])` -- the same states, the same `total_bits` to the last bit, the same
label -- at every `batch_size` and on every backend the call has. A min-sum performs
only `+` and `<`, so unlike the E-step, where regrouping a sum moves the last bits,
batching a decode has no tolerance to hide in. Every equivalence below is therefore
`array_equal` on the states and a byte comparison on the total.

`viterbi` is the oracle here, and it is a fair one: `test_viterbi.py` holds it to
the original's own saved decodes. The batch is checked against it rather than
against the saved files directly, because a batch of the corpus split into its 200
`.seq` files is not the original's one flat stream.

**The padding cases are constructed, not waited for.** On a valid model `PAD`'s
emission fibre is zero, so a padded step that ran would turn a short record's final
δ infinite, and every ragged batch below would catch it. The kernel-level test goes
one step further and makes `PAD` emittable in raw arrays, where a padded step would
be finite and merely wrong.

The `backend` fixture is this call's own, over `_TABLE["viterbi_batch"]`, so a
lifecycle phase that batches is added by a row in `_backends.py`, not by a test.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from pfsmgraph.dataseq import PAD, UNK, USER_BASE, SequenceRecord, SymbolTable, pad_collate

import pfsmgraph.hmm
from pfsmgraph.hmm import (
    HMMParams,
    ImpossibleSequenceError,
    ViterbiPath,
    backends,
    viterbi,
    viterbi_batch,
)
from pfsmgraph.hmm._backends import _TABLE, _resolve
from pfsmgraph.hmm._viterbi import _viterbi

from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params


@pytest.fixture(scope="module", params=[row.name for row in _TABLE["viterbi_batch"]])
def backend(request):
    """One `viterbi_batch` backend name, or a skip naming why it cannot run here."""
    status = {s.name: s for s in backends("viterbi_batch")}[request.param]
    if not status.available:
        pytest.skip(f"backend {status.name!r} unavailable: {status.reason}")
    return request.param


# --- models and records ------------------------------------------------------


def _params(init_p, transition_p, user_output):
    """An `HMMParams` from user-symbol emissions, reserved block zero-padded."""
    user_output = np.asarray(user_output, dtype=np.float64)
    size, _, n_symbols = user_output.shape
    vocabulary = SymbolTable([f"s{i}" for i in range(n_symbols)])
    output_p = np.zeros((size, size, vocabulary.size), dtype=np.float64)
    output_p[:, :, USER_BASE:] = user_output
    return HMMParams(
        init_state_p=np.asarray(init_p, dtype=np.float64),
        transition_p=np.asarray(transition_p, dtype=np.float64),
        output_p=output_p,
        vocabulary=vocabulary,
    )


def _random_model(rng, size, n_symbols):
    transition_p = rng.random((size, size))
    transition_p /= transition_p.sum(axis=1, keepdims=True)
    user_output = rng.random((size, size, n_symbols))
    user_output /= user_output.sum(axis=2, keepdims=True)
    init_p = rng.random(size)
    init_p /= init_p.sum()
    return _params(init_p, transition_p, user_output)


def _uniform_model(size, n_symbols):
    """Exactly uniform everywhere, so every candidate at every position ties."""
    return _params(
        np.full(size, 1.0 / size),
        np.full((size, size), 1.0 / size),
        np.full((size, size, n_symbols), 1.0 / n_symbols),
    )


def _record(indices, label=None):
    return SequenceRecord(np.asarray([USER_BASE + i for i in indices], dtype=np.int32), label=label)


def _random_records(rng, n_symbols, count, max_length=40):
    """Ragged records, labelled by position, with an empty and a length-1 one."""
    lengths = [0, 1, *rng.integers(0, max_length, count)]
    rng.shuffle(lengths)
    return [
        _record(rng.integers(0, n_symbols, int(n)), label=f"r{i}")
        for i, n in enumerate(lengths)
    ]


def _assert_same_path(batched, alone):
    assert isinstance(batched, ViterbiPath)
    assert np.array_equal(batched.states, alone.states)
    assert batched.states.dtype == alone.states.dtype
    assert np.float64(batched.total_bits).tobytes() == np.float64(alone.total_bits).tobytes()
    assert batched.label == alone.label


def _assert_rows_are_viterbi(params, records, paths, backend):
    assert len(paths) == len(records)
    for record, path in zip(records, paths):
        _assert_same_path(path, viterbi(params, record, backend=backend))


# --- every row is the decode of that record alone -----------------------------


def test_every_row_is_viterbi_on_that_record_alone(backend):
    """200 shapes up to 40 states, each a ragged batch with empty and length-1 records."""
    rng = np.random.default_rng(20260914)
    for _ in range(200):
        n_symbols = int(rng.integers(1, 6))
        params = _random_model(rng, int(rng.integers(1, 41)), n_symbols)
        records = _random_records(rng, n_symbols, int(rng.integers(0, 8)))
        _assert_rows_are_viterbi(params, records, viterbi_batch(params, records, backend=backend), backend)


def test_every_backend_agrees_with_the_batched_reference(backend):
    """Each backend's batch against `backend="python"`'s batch, exactly.

    The tests around this one hold a backend's batch to its own per-record decode;
    this holds it to the reference, which is what a lifecycle phase promises.
    """
    rng = np.random.default_rng(20260915)
    for _ in range(200):
        n_symbols = int(rng.integers(1, 6))
        params = _random_model(rng, int(rng.integers(1, 41)), n_symbols)
        records = _random_records(rng, n_symbols, int(rng.integers(0, 8)))
        reference = viterbi_batch(params, records, backend="python")
        for path, expected in zip(viterbi_batch(params, records, backend=backend), reference):
            _assert_same_path(path, expected)


@pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
def test_the_result_ignores_batch_size(batch_size, backend):
    rng = np.random.default_rng(7)
    params = _random_model(rng, 9, 4)
    records = _random_records(rng, 4, 11)
    paths = viterbi_batch(params, records, backend=backend, batch_size=batch_size)
    _assert_rows_are_viterbi(params, records, paths, backend)


def test_ties_are_broken_toward_the_lower_state_in_every_row(backend):
    """The first-wins tie-break is contract, and a uniform model ties everywhere.

    `rand_p_vector(size, noise_width=0)` returns exactly this, which is how
    training initialises, so the case is not academic.
    """
    params = _uniform_model(4, 3)
    records = [_record([0, 1, 2, 1]), _record([]), _record([2]), _record([1, 1, 0, 2, 2, 0, 1])]
    paths = viterbi_batch(params, records, backend=backend)
    _assert_rows_are_viterbi(params, records, paths, backend)
    for path in paths:
        assert not path.states.any()


def test_the_corpus_decodes_beside_short_records_exactly_as_alone(backend):
    """The 1269-symbol corpus pads every other row to its own length."""
    params = load_params(FIXTURES / SAVED_MODELS[0])
    corpus = load_corpus_record()
    records = [
        SequenceRecord(corpus.codes[:3]),
        corpus,
        SequenceRecord(corpus.codes[:0]),
        SequenceRecord(corpus.codes[100:117]),
    ]
    paths = viterbi_batch(params, records, backend=backend)
    _assert_rows_are_viterbi(params, records, paths, backend)


# --- impossible records --------------------------------------------------------


def _impossible_beside_possible():
    rng = np.random.default_rng(3)
    params = _random_model(rng, 5, 3)
    unk = SequenceRecord(np.asarray([USER_BASE, UNK, USER_BASE + 1], dtype=np.int32))
    records = [_record([0, 1, 2]), unk, _record([2, 2]), _record([1])]
    return params, records


def test_an_impossible_record_raises_naming_its_index_and_dead_symbol(backend):
    params, records = _impossible_beside_possible()
    with pytest.raises(ImpossibleSequenceError, match=r"^record 1: no path over these 3 symbols") as excinfo:
        viterbi_batch(params, records, backend=backend)
    # The same diagnosis viterbi gives the record alone, prefixed by its index.
    with pytest.raises(ImpossibleSequenceError) as alone:
        viterbi(params, records[1], backend=backend)
    assert str(excinfo.value) == f"record 1: {alone.value}"


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_the_first_impossible_record_in_record_order_is_reported(batch_size, backend):
    params, records = _impossible_beside_possible()
    records = [*records, SequenceRecord(np.asarray([PAD], dtype=np.int32))]
    with pytest.raises(ImpossibleSequenceError, match=r"^record 1: "):
        viterbi_batch(params, records, backend=backend, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_on_impossible_none_leaves_a_gap_and_decodes_the_rest(batch_size, backend):
    params, records = _impossible_beside_possible()
    paths = viterbi_batch(params, records, backend=backend, batch_size=batch_size, on_impossible="none")
    assert paths[1] is None
    for index in (0, 2, 3):
        _assert_same_path(paths[index], viterbi(params, records[index], backend=backend))


# --- validation, all before any work ---------------------------------------------


def test_no_records_decode_to_an_empty_list(backend):
    params = _uniform_model(2, 2)
    assert viterbi_batch(params, [], backend=backend) == []
    assert viterbi_batch(params, iter(()), backend=backend, batch_size=3) == []


def test_any_iterable_of_records_is_accepted(backend):
    params = _uniform_model(3, 2)
    records = [_record([0, 1]), _record([1])]
    paths = viterbi_batch(params, (record for record in records), backend=backend)
    _assert_rows_are_viterbi(params, records, paths, backend)


@pytest.mark.parametrize("policy", ["skip", "None", None, "RAISE"])
def test_an_unknown_policy_is_refused_before_any_work(policy):
    params = _uniform_model(2, 2)
    with pytest.raises(ValueError, match=r"on_impossible must be one of \['raise', 'none'\]"):
        viterbi_batch(params, [], on_impossible=policy)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_batch_size_below_one_is_refused(batch_size):
    params = _uniform_model(2, 2)
    with pytest.raises(ValueError, match="batch_size must be at least 1 or None"):
        viterbi_batch(params, [], batch_size=batch_size)


def test_a_backend_this_call_does_not_have_is_refused_even_with_no_records():
    params = _uniform_model(2, 2)
    with pytest.raises(ValueError, match="viterbi_batch has no 'torch' backend"):
        viterbi_batch(params, [], backend="torch")
    with pytest.raises(ValueError, match="backend must be one of"):
        viterbi_batch(params, [], backend="numpy")


def test_an_out_of_range_code_is_refused_naming_its_record(backend):
    """Every range check runs before any decode, so an impossible record before
    the bad one does not get to raise first."""
    params, records = _impossible_beside_possible()
    bad = SequenceRecord(np.asarray([params.n_symbols], dtype=np.int32))
    with pytest.raises(ValueError, match=r"^record 4: record holds code\(s\) outside") as excinfo:
        viterbi_batch(params, [*records, bad], backend=backend)
    assert not isinstance(excinfo.value, ImpossibleSequenceError)


# --- the public surface ----------------------------------------------------------


def test_the_batched_decode_is_exported_and_enumerable():
    assert "viterbi_batch" in pfsmgraph.hmm.__all__
    assert pfsmgraph.hmm.viterbi_batch is viterbi_batch
    assert [s.name for s in backends("viterbi_batch")] == [row.name for row in _TABLE["viterbi_batch"]]


# --- kernel-level: deliberately NOT public-API -------------------------------------
#
# What the wrapper hides: the kernel's padded return shape, its numeric report of
# impossibility, and inputs no valid HMMParams can hold.


@pytest.fixture(scope="module")
def kernel(backend):
    return _resolve("viterbi_batch", backend)


def test_the_kernel_reports_impossibility_numerically(kernel):
    params, records = _impossible_beside_possible()
    batch = pad_collate(records)
    states, total_bits = kernel(
        params.init_state_p, params.transition_p, params.output_p, batch["codes"], batch["lengths"]
    )
    assert np.isinf(total_bits[1])
    assert np.isfinite(total_bits[[0, 2, 3]]).all()


def test_the_kernel_zeroes_the_states_past_each_records_end(kernel):
    rng = np.random.default_rng(11)
    params = _random_model(rng, 6, 3)
    records = _random_records(rng, 3, 6)
    batch = pad_collate(records)
    states, _ = kernel(
        params.init_state_p, params.transition_p, params.output_p, batch["codes"], batch["lengths"]
    )
    assert states.shape == (len(records), batch["codes"].shape[1] + 1)
    for row, record in enumerate(records):
        assert not states[row, record.length + 1 :].any()


def test_padding_stays_inert_where_pad_is_emittable(kernel):
    """Raw arrays with `PAD` emittable, so a padded step would be finite and wrong.

    On a valid model a padded step goes infinite and every ragged batch catches it.
    Here it would not, so this is the case that pins the carry itself: each row must
    still equal the per-record kernel over the record's own codes.
    """
    rng = np.random.default_rng(5)
    size, width = 6, USER_BASE + 3
    init_state_p = rng.random(size)
    init_state_p /= init_state_p.sum()
    transition_p = rng.random((size, size))
    transition_p /= transition_p.sum(axis=1, keepdims=True)
    output_p = rng.random((size, size, width))
    output_p[:, :, PAD] *= 50.0  # the cheapest symbol of all
    output_p /= output_p.sum(axis=2, keepdims=True)

    records = _random_records(rng, 3, 7, max_length=25)
    batch = pad_collate(records)
    states, total_bits = kernel(init_state_p, transition_p, output_p, batch["codes"], batch["lengths"])
    assert np.isfinite(total_bits).all()
    for row, record in enumerate(records):
        alone_states, alone_bits = _viterbi(init_state_p, transition_p, output_p, record.codes)
        assert np.array_equal(states[row, : record.length + 1], alone_states)
        assert np.float64(total_bits[row]).tobytes() == np.float64(alone_bits).tobytes()


# --- phase 3 only: the CPU-parallel batch under real threads -----------------------
#
# A race is a property of the interleaving, so these run the batch at one thread and
# at the most this process has. `test_viterbi.py` holds the same pair for the
# per-record kernel; the batch parallelises a different loop, so it needs its own.


@pytest.fixture
def numba_threads():
    """`(max_threads, set_num_threads)`, restoring the thread count afterwards."""
    status = {s.name: s for s in backends("viterbi_batch")}["cpu_parallel"]
    if not status.available:
        pytest.skip(f"backend 'cpu_parallel' unavailable: {status.reason}")
    numba = importlib.import_module("numba")
    original = numba.get_num_threads()
    yield numba.config.NUMBA_NUM_THREADS, numba.set_num_threads
    numba.set_num_threads(original)


def test_the_cpu_parallel_batch_is_invariant_to_thread_count(numba_threads):
    """One answer at every thread count, over 40 ragged batches; two would be a race."""
    max_threads, set_num_threads = numba_threads
    if max_threads < 2:
        pytest.skip(f"needs >= 2 threads to compare; this process has {max_threads}")

    rng = np.random.default_rng(20260915)
    cases = []
    for _ in range(40):
        n_symbols = int(rng.integers(1, 6))
        params = _random_model(rng, int(rng.integers(1, 17)), n_symbols)
        cases.append((params, _random_records(rng, n_symbols, int(rng.integers(0, 12)), max_length=60)))
    results = {}
    for threads in (1, max_threads):
        set_num_threads(threads)
        results[threads] = [viterbi_batch(p, recs, backend="cpu_parallel") for p, recs in cases]
    for one, many in zip(results[1], results[max_threads]):
        for a, b in zip(one, many):
            _assert_same_path(a, b)


def test_the_cpu_parallel_batch_breaks_ties_to_the_smallest_index_at_every_thread_count(
    numba_threads,
):
    """A uniform model ties at every cell; lifting the `i` reduction into the
    parallel loop would resolve those ties in whatever order threads combined."""
    max_threads, set_num_threads = numba_threads
    params = _uniform_model(5, 3)
    records = [_record([i % 3 for i in range(n)]) for n in (24, 0, 1, 17, 24, 9)]
    for threads in {1, max_threads}:
        set_num_threads(threads)
        for path in viterbi_batch(params, records, backend="cpu_parallel"):
            assert not path.states.any(), f"at {threads} thread(s)"
