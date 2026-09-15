"""Forward-backward's lifecycle phases against the numpy reference, byte for byte.

**The compiled phases are held to equality, not to a tolerance.** ADR 0020 fixes the
evaluation order of every sum and forbids fused multiply-add precisely so that phases 2-4
can agree with the reference to the bit on one host. `test_baum_welch_backends.py` holds
`torch` to `N · eps · max(1, x)` because torch cannot order its reductions; nothing here
may borrow that bound. Comparisons are on `tobytes()`, so a `-0.0` where the reference
has `0.0`, or a NaN payload, is a failure too.

**Every test reaches a private kernel**, labelled rather than hidden, as ADR 0003 asks:
`forward_backward` has no public call, and the counts are invisible from `baum_welch`. The
kernels come from the backend table through `_resolve`, so a phase registers here by
gaining a row, with no edit to this file. `torch` is excluded by name, for the reason
above.

The cases are the formalization's (`docs/design/algorithms/forward_backward/`):

- **TC-22**, every phase's `_forward_backward` and `_e_step_batch` equal to the
  reference's, on generated models and ragged batches, the Lush fixtures, an impossible
  record, and raw arrays where `PAD` is emittable, so the padding rule is load-bearing;
- **TC-20**, an empty record costs exactly 0 bits whatever its seed sums to;
- **TC-21**, an input on which a fused multiply-add rounds differently, found at test
  time, so a contracting build fails here rather than passing every other case;
- **TC-23**, no result depends on the memory layout of the parameters.

Phase 3 adds one section of its own at the end: its kernels at one thread and at the
most this process has, which is the check that sees a race or a sum numba turned into a
parallel reduction.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pfsmgraph.dataseq import PAD, UNK, USER_BASE, SequenceRecord, pad_collate
from pfsmgraph.hmm._backends import _TABLE, _resolve, _status

from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params

SEED = 20260916

#: The algorithm keys whose rows implement this formalization's two signatures.
ALGORITHMS = ("forward_backward", "baum_welch")

#: Held to a tolerance by its own suite, never to equality.
UNORDERED = frozenset({"torch"})


def _ordered_rows(algorithm):
    return [row.name for row in _TABLE[algorithm] if row.name not in UNORDERED]


def _kernel(algorithm, name):
    status = {s.name: s for s in _status(algorithm)}[name]
    if not status.available:
        pytest.skip(f"backend {name!r} unavailable: {status.reason}")
    return _resolve(algorithm, name)


def test_every_phase_registers_both_signatures():
    # The formalization holds every phase to both kernels, and the fixtures below take
    # their parameters from one table on that basis, so a phase registered under one key
    # alone would silently lose half its cases.
    assert _ordered_rows("forward_backward") == _ordered_rows("baum_welch")


@pytest.fixture(params=_ordered_rows("baum_welch"))
def phase(request):
    """One lifecycle phase's name. Both kernel fixtures follow it, so a test that takes
    both runs once per phase rather than once per pairing of phases."""
    return request.param


@pytest.fixture
def recurrences(phase):
    """That phase's `_forward_backward`, or a skip naming why it cannot run here."""
    return _kernel("forward_backward", phase)


@pytest.fixture
def e_step(phase):
    """That phase's `_e_step_batch`, or a skip naming why it cannot run here."""
    return _kernel("baum_welch", phase)


def _reference(algorithm):
    return _resolve(algorithm, "python")


def _assert_same_bytes(ours, theirs, label):
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    assert ours.dtype == theirs.dtype == np.float64, label
    assert ours.shape == theirs.shape, label
    if ours.tobytes() != theirs.tobytes():
        differ = np.flatnonzero(ours.ravel().view(np.uint64) != theirs.ravel().view(np.uint64))
        pytest.fail(
            f"{label}: {differ.size} of {ours.size} elements differ; first at flat index "
            f"{differ[0]}: {ours.ravel()[differ[0]]!r} against {theirs.ravel()[differ[0]]!r}"
        )


def _assert_recurrences_match(kernel, init, transition, output, codes, label=""):
    for name, ours, theirs in zip(
        ("alpha", "beta", "scale"),
        kernel(init, transition, output, codes),
        _reference("forward_backward")(init, transition, output, codes),
    ):
        _assert_same_bytes(ours, theirs, f"{label} {name}")


def _assert_e_step_matches(kernel, init, transition, output, codes, lengths, label=""):
    (ours_counts, ours_bits) = kernel(init, transition, output, codes, lengths)
    (their_counts, their_bits) = _reference("baum_welch")(init, transition, output, codes, lengths)
    for name, ours, theirs in zip(("init", "transition", "emission"), ours_counts, their_counts):
        _assert_same_bytes(ours, theirs, f"{label} {name} counts")
    _assert_same_bytes(ours_bits, their_bits, f"{label} bits")


# --- models ------------------------------------------------------------------------


def _random_model(rng, size, n_user, dead_fraction):
    """Raw arrays over the whole ADR 0011 symbol axis, some arcs exactly dead."""
    init = rng.dirichlet(np.ones(size))
    transition = rng.dirichlet(np.ones(size), size=size)
    dead = rng.random((size, size)) < dead_fraction
    dead[np.arange(size), np.arange(size)] = False
    transition = np.where(dead, 0.0, transition)
    transition /= transition.sum(axis=1, keepdims=True)
    output = np.zeros((size, size, USER_BASE + n_user))
    output[:, :, USER_BASE:] = rng.dirichlet(np.ones(n_user), size=(size, size))
    return init, transition, output


def _ragged(rng, n_user, lengths):
    return [SequenceRecord(rng.integers(USER_BASE, USER_BASE + n_user, n)) for n in lengths]


# --- TC-22: every phase is the reference -------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 16, 32])
@pytest.mark.parametrize("length", [0, 1, 7, 50, 300])
def test_every_phases_recurrences_are_the_references_on_generated_models(size, length, recurrences):
    rng = np.random.default_rng([SEED, size, length])
    for trial in range(3):
        init, transition, output = _random_model(rng, size, 6, 0.3)
        codes = rng.integers(USER_BASE, USER_BASE + 6, length)
        _assert_recurrences_match(recurrences, init, transition, output, codes, f"trial {trial}")


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 16, 32])
def test_every_phases_e_step_is_the_references_on_ragged_batches(size, e_step):
    rng = np.random.default_rng([SEED, size])
    for trial in range(6):
        init, transition, output = _random_model(rng, size, 5, 0.3)
        lengths = [int(n) for n in rng.integers(0, 120, int(rng.integers(1, 9)))] + [0, 1]
        batch = pad_collate(_ragged(rng, 5, lengths))
        _assert_e_step_matches(
            e_step, init, transition, output, batch["codes"], batch["lengths"], f"trial {trial}"
        )


@pytest.mark.parametrize("model", SAVED_MODELS)
def test_every_phase_is_the_reference_on_the_lush_fixtures(model, recurrences, e_step):
    params, record = load_params(FIXTURES / model), load_corpus_record()
    arrays = (params.init_state_p, params.transition_p, params.output_p)
    # Not a formality: fed un-renumbered codes every fixture is impossible, and an
    # all-zero comparison would pass having checked nothing.
    assert np.isfinite(_reference("baum_welch")(*arrays, record.codes[np.newaxis], [record.length])[1][0])
    # The parameters are frozen, so this is also the read-only-buffer case.
    assert not params.transition_p.flags.writeable
    _assert_recurrences_match(recurrences, *arrays, record.codes, model)
    _assert_e_step_matches(e_step, *arrays, record.codes[np.newaxis], [record.length], model)


def test_every_phase_is_the_reference_on_an_impossible_record(recurrences, e_step):
    rng = np.random.default_rng(SEED + 1)
    init, transition, output = _random_model(rng, 4, 5, 0.0)
    # UNK's fibres are zero, so the record dies at position 1 and every later column
    # is zero: the `x / 0 = 0` path of every division, forward and backward.
    codes = np.array([USER_BASE, UNK, USER_BASE, USER_BASE + 1])
    _assert_recurrences_match(recurrences, init, transition, output, codes, "impossible")
    records = [SequenceRecord(codes), *_ragged(rng, 5, [9, 0, 2])]
    batch = pad_collate(records)
    _assert_e_step_matches(e_step, init, transition, output, batch["codes"], batch["lengths"], "impossible")


def test_every_phase_keeps_padding_inert_where_pad_is_emittable(e_step):
    # PAD's fibre is zero in every valid model, so padding already weighs 0 and a kernel
    # that stepped over it would still pass on HMMParams. With PAD the likeliest symbol,
    # only the padding rule stands between the padded positions and the counts.
    rng = np.random.default_rng(SEED + 2)
    init, transition, output = _random_model(rng, 5, 3, 0.3)
    output[:, :, PAD] = 0.5
    output /= output.sum(axis=2, keepdims=True)
    batch = pad_collate(_ragged(rng, 3, [200, 1, 0, 37, 1]))
    assert not batch["mask"].all()
    _assert_e_step_matches(e_step, init, transition, output, batch["codes"], batch["lengths"], "pad-emittable")


# --- TC-20: an empty record costs nothing ------------------------------------------


@pytest.mark.parametrize("total", [1.0 + 1e-6, 1.0 - 1e-6], ids=["above-one", "below-one"])
def test_an_empty_record_costs_zero_bits_whatever_its_seed_sums_to(total, recurrences, e_step):
    # Enumeration gives -log2(sum(init)) here, and so did the original's trailing term;
    # the formalization specifies 0 (deviation 5). A seed summing to 1 within rounding
    # cannot tell the two apart, which is why the seed is pushed off 1 on purpose.
    rng = np.random.default_rng(SEED + 3)
    init, transition, output = _random_model(rng, 3, 4, 0.0)
    init = init * total
    empty = np.array([], dtype=np.int64)

    alpha, beta, scale = recurrences(init, transition, output, empty)
    np.testing.assert_array_equal(scale, [1.0])
    _assert_same_bytes(alpha, init[np.newaxis, :], "alpha")

    (init_counts, transition_counts, emission_counts), bits = e_step(
        init, transition, output, np.zeros((1, 0), dtype=np.int64), np.array([0])
    )
    assert bits[0] == 0.0
    assert not init_counts.any() and not transition_counts.any() and not emission_counts.any()


# --- TC-21: no fused multiply-add ---------------------------------------------------


def _fused(a, b, c):
    """`fma(a, b, c)`: `a * b + c` computed exactly, then rounded once.

    `float(Fraction)` is correctly rounded, since CPython's integer true division is.
    """
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _contraction_witness(rng):
    """A 2-state, 1-symbol model on which contracting `c + alpha * w` changes Q[1].

    With one symbol and every emission 1, `w` is the transition matrix, so the first
    forward column is `init[0] * t[0, j] + init[1] * t[1, j]`, and a contracting
    compiler computes `fma(init[1], t[1, j], init[0] * t[0, j])` instead. Raw arrays: the
    kernel contract does not ask for a valid model.
    """
    for _ in range(10_000):
        init = rng.random(2)
        transition = rng.random((2, 2))
        term = float(init[0] * transition[0, 0])
        if float(term + init[1] * transition[1, 0]) != _fused(float(init[1]), float(transition[1, 0]), term):
            return init, transition, np.ones((2, 2, 1))
    raise AssertionError("no contraction witness in 10 000 draws; the search is broken")


def test_no_phase_fuses_multiply_add(recurrences, e_step):
    init, transition, output = _contraction_witness(np.random.default_rng(SEED + 4))
    codes = np.zeros(6, dtype=np.int64)

    # The witness really discriminates: the reference's Q[1] is the unfused sum.
    _, _, scale = _reference("forward_backward")(init, transition, output, codes)
    column_0 = float(init[0] * transition[0, 0]) + float(init[1] * transition[1, 0])
    column_1 = float(init[0] * transition[0, 1]) + float(init[1] * transition[1, 1])
    assert scale[1] == column_0 + column_1

    _assert_recurrences_match(recurrences, init, transition, output, codes, "contraction")
    _assert_e_step_matches(e_step, init, transition, output, codes[np.newaxis], np.array([6]), "contraction")


# --- TC-23: memory layout is not a parameter ----------------------------------------


def test_no_phase_depends_on_the_memory_layout_of_the_parameters(recurrences, e_step):
    # terms.sum(axis=0) agrees with an ascending loop on a C-ordered array and disagreed
    # in 50 of 50 trials on a Fortran-ordered copy at S >= 16, so a reduction that passes
    # every C-ordered case can still follow memory layout.
    rng = np.random.default_rng(SEED + 5)
    for size in (16, 33):
        init, transition, output = _random_model(rng, size, 7, 0.2)
        fortran = (init, np.asfortranarray(transition), np.asfortranarray(output))
        assert not fortran[1].flags.c_contiguous
        codes = rng.integers(USER_BASE, USER_BASE + 7, 60)
        batch = pad_collate(_ragged(rng, 7, [60, 13, 0]))

        for name, ours, theirs in zip(
            ("alpha", "beta", "scale"),
            recurrences(*fortran, codes),
            _reference("forward_backward")(init, transition, output, codes),
        ):
            _assert_same_bytes(ours, theirs, f"S={size} fortran {name}")
        ours_counts, ours_bits = e_step(*fortran, batch["codes"], batch["lengths"])
        their_counts, their_bits = _reference("baum_welch")(init, transition, output, batch["codes"], batch["lengths"])
        for name, ours, theirs in zip(("init", "transition", "emission"), ours_counts, their_counts):
            _assert_same_bytes(ours, theirs, f"S={size} fortran {name} counts")
        _assert_same_bytes(ours_bits, their_bits, f"S={size} fortran bits")


# --- phase 3 only: the CPU-parallel kernels under real threads ----------------------
#
# A race, or a sum numba turned into a parallel reduction, is a property of the
# interleaving, so these run the cpu_parallel kernels at one thread and at the most this
# process has, and require one answer, the reference's. The shared cases above run at
# whatever thread count the session has, which cannot tell one thread from many.


@pytest.fixture
def numba_threads():
    """`(max_threads, set_num_threads)`, restoring the thread count afterwards."""
    status = {s.name: s for s in _status("baum_welch")}["cpu_parallel"]
    if not status.available:
        pytest.skip(f"backend 'cpu_parallel' unavailable: {status.reason}")
    numba = pytest.importorskip("numba")
    original = numba.get_num_threads()
    yield numba.config.NUMBA_NUM_THREADS, numba.set_num_threads
    numba.set_num_threads(original)


def _at_each_thread_count(numba_threads, run):
    max_threads, set_num_threads = numba_threads
    if max_threads < 2:
        pytest.skip(f"needs >= 2 threads to compare; this process has {max_threads}")
    results = {}
    for threads in (1, max_threads):
        set_num_threads(threads)
        results[threads] = run()
    return results[1], results[max_threads]


def test_the_cpu_parallel_kernels_are_invariant_to_thread_count(numba_threads):
    """One answer at every thread count, and it is the reference's; two would be a race."""
    recurrences = _resolve("forward_backward", "cpu_parallel")
    e_step = _resolve("baum_welch", "cpu_parallel")
    rng = np.random.default_rng(SEED + 6)
    cases = []
    for _ in range(40):
        size = int(rng.integers(1, 24))
        init, transition, output = _random_model(rng, size, 5, 0.3)
        codes = rng.integers(USER_BASE, USER_BASE + 5, int(rng.integers(0, 150)))
        lengths = [int(n) for n in rng.integers(0, 150, int(rng.integers(1, 12)))] + [0, 1]
        batch = pad_collate(_ragged(rng, 5, lengths))
        cases.append((init, transition, output, codes, batch["codes"], batch["lengths"]))

    def run():
        return [
            (recurrences(i, t, o, c), e_step(i, t, o, bc, bl)) for i, t, o, c, bc, bl in cases
        ]

    one, many = _at_each_thread_count(numba_threads, run)
    for index, ((r1, (c1, b1)), (rn, (cn, bn)), case) in enumerate(zip(one, many, cases)):
        init, transition, output, codes, batch_codes, batch_lengths = case
        reference_recurrences = _reference("forward_backward")(init, transition, output, codes)
        reference_counts, reference_bits = _reference("baum_welch")(
            init, transition, output, batch_codes, batch_lengths
        )
        for name, a, b, want in zip(("alpha", "beta", "scale"), r1, rn, reference_recurrences):
            _assert_same_bytes(a, b, f"case {index} {name}, 1 thread against many")
            _assert_same_bytes(b, want, f"case {index} {name}, many threads against python")
        for name, a, b, want in zip(("init", "transition", "emission"), c1, cn, reference_counts):
            _assert_same_bytes(a, b, f"case {index} {name} counts, 1 thread against many")
            _assert_same_bytes(b, want, f"case {index} {name} counts, many threads against python")
        _assert_same_bytes(b1, bn, f"case {index} bits, 1 thread against many")
        _assert_same_bytes(bn, reference_bits, f"case {index} bits, many threads against python")


@pytest.mark.parametrize("length", [1, 7, 40])
def test_the_cpu_parallel_kernels_compute_a_uniform_model_exactly_at_every_thread_count(
    length, numba_threads
):
    """A sum has no tie to break, so this is the constructed case in its place.

    Every probability is 1/4 and every intermediate dyadic, so nothing rounds and the
    answer is known in closed form: Q = 1/4 at every step, 2 bits per symbol, a quarter of
    each count. Any thread count must produce exactly that, record by record.
    """
    e_step = _resolve("baum_welch", "cpu_parallel")
    size = n_user = 4
    init = np.full(size, 0.25)
    transition = np.full((size, size), 0.25)
    output = np.zeros((size, size, USER_BASE + n_user))
    output[:, :, USER_BASE:] = 0.25
    rng = np.random.default_rng(SEED + 7)
    records = _ragged(rng, n_user, [length, 0, 1, length])
    batch = pad_collate(records)

    one, many = _at_each_thread_count(
        numba_threads, lambda: e_step(init, transition, output, batch["codes"], batch["lengths"])
    )
    for (counts, bits) in (one, many):
        for b, record in enumerate(records):
            n = record.length
            np.testing.assert_array_equal(bits[b], 2.0 * n)
            np.testing.assert_array_equal(counts[0][b], 0.25 if n else 0.0)
            np.testing.assert_array_equal(counts[1][b], n / 16)
            for symbol in range(n_user):
                expected = np.count_nonzero(record.codes == USER_BASE + symbol) / 16
                np.testing.assert_array_equal(counts[2][b][:, :, USER_BASE + symbol], expected)
