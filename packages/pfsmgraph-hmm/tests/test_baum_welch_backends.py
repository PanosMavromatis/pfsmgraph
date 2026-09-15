"""Every `baum_welch` backend against the numpy reference: the count identity.

The torch backend and the reference share no code. The reference writes α, β and
ξ out and adds ξ into the counts; torch writes only the forward pass and takes
the counts as gradients, `x · ∂ln P/∂x` for every weight `x` in it (Eisner 2016).
That the two agree is the check the backend exists to provide, and it is not a
check `hmmlearn` could make, since autograd and a hand-written β are two
derivations of the same model and could share a misreading of it -- which is why
`test_hmmlearn_oracle.py` validates the reference first.

**Agreement is within a tolerance, and the tolerance was measured, not chosen**
(ADR 0020: torch's reductions have no order a caller can fix). Every count, per
element, agrees within `N · eps · max(1, count)` and every description length
within `N · eps · max(1, bits)`, measured worst 1.00 and 0.21 over random models,
EM trajectories and the Lush fixtures. `KERNEL_K` holds both to four times that.
A relative bound is the wrong shape: counts span 1e-300 to 1e3, and near
underflow a relative error of `5e57` is still an absolute one of rounding.

**Two layers, as `test_viterbi.py` has.** Above the line the public
`baum_welch` runs on each backend through this module's `backend` fixture and must
train like `backend="python"`: the same number of cycles, the same stopping
verdict, the same degenerate states, and parameters within `PARAMS_ATOL`. Below
it the kernels are reached through the private `_resolve`, because counts are
invisible from the public call.

**The EM trajectories are the load-bearing cases.** Random models cannot exhibit
either defect this suite was written after finding, since their Dirichlet
parameters are never near zero. EM drives parameters to `1e-50` and below within
a few dozen cycles, and there the first torch kernel returned negative counts
(differentiating through the scale factors). Longer runs reach the subnormal
range, where the second returned fibres summing to 1.002 (differentiating the
transition and emission arrays separately, a subnormal unit apart). Both crashed
`HMMParams`. The kernel tests therefore walk EM trajectories for the first and
construct subnormal arcs for the second, and re-estimate from every state, so a
green run means every kernel's counts make a valid model where it matters.
"""

from __future__ import annotations

import functools
import io

import numpy as np
import pytest

from pfsmgraph.dataseq import UNK, USER_BASE, SequenceRecord, SymbolTable, pad_collate
from pfsmgraph.hmm import (
    BackendUnavailableError,
    HMMParams,
    ImpossibleSequenceError,
    backends,
    baum_welch,
)
from pfsmgraph.hmm._backends import _TABLE, _resolve
from pfsmgraph.hmm._baum_welch import _corpus_step, _re_estimate

from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params

SEED = 20260915
EPS = np.finfo(np.float64).eps

#: Multiple of the measured worst for `|Δ| / (N · eps · max(1, x))`, which was
#: 1.00 for counts and 0.21 for description lengths.
KERNEL_K = 4.0

#: Fitted parameters after a whole run; measured worst 3.4e-14 over 30 runs.
PARAMS_ATOL = 1e-12


@pytest.fixture(scope="module", params=[row.name for row in _TABLE["baum_welch"]])
def backend(request):
    """One `baum_welch` backend name, or a skip naming why it cannot run here.

    Overrides `conftest.py`'s fixture, which is keyed to `viterbi`'s table.
    """
    status = {s.name: s for s in backends("baum_welch")}[request.param]
    if not status.available:
        pytest.skip(f"backend {status.name!r} unavailable: {status.reason}")
    return request.param


# --- building models ------------------------------------------------------------


def _vocabulary(n_user):
    return SymbolTable([f"s{i}" for i in range(n_user)])


def _random_params(rng, size, n_user, dead_fraction):
    """A model with some arcs exactly dead, the diagonal kept live."""
    transition = rng.dirichlet(np.ones(size), size=size)
    dead = rng.random((size, size)) < dead_fraction
    dead[np.arange(size), np.arange(size)] = False
    transition = np.where(dead, 0.0, transition)
    transition /= transition.sum(axis=1, keepdims=True)
    output = np.zeros((size, size, USER_BASE + n_user))
    output[:, :, USER_BASE:] = rng.dirichlet(np.ones(n_user), size=(size, size))
    return HMMParams(rng.dirichlet(np.ones(size)), transition, output, _vocabulary(n_user))


def _random_corpus(rng, n_user, n_records, max_length):
    return [
        SequenceRecord(rng.integers(USER_BASE, USER_BASE + n_user, int(rng.integers(5, max_length))))
        for _ in range(n_records)
    ]


def _trainings(seed, count):
    """`count` small random trainings, each a starting model and its corpus."""
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(count):
        size, n_user = int(rng.integers(2, 7)), int(rng.integers(2, 6))
        params = _random_params(rng, size, n_user, 0.2)
        runs.append((params, _random_corpus(rng, n_user, int(rng.integers(1, 6)), 60)))
    return runs


def _arrays(params):
    return params.init_state_p, params.transition_p, params.output_p


def _within(ours, reference, n):
    """`|ours - reference| <= KERNEL_K · n · eps · max(1, |reference|)`, elementwise."""
    bound = KERNEL_K * max(n, 1) * EPS * np.maximum(1.0, np.abs(reference))
    return bool((np.abs(np.asarray(ours) - reference) <= bound).all())


# === public API: every backend trains like the reference =========================


@functools.cache
def _reference_run(key):
    if key == "fixture":
        params, records = load_params(FIXTURES / "m008_0001_008.hmm"), [load_corpus_record()]
    else:
        params, records = _trainings(SEED, 8)[key]
    return params, records, baum_welch(params, records)


def _assert_trains_like_the_reference(key, backend):
    params, records, reference = _reference_run(key)
    # The reference is compared with its own cached run rather than trained twice.
    ours = reference if backend == "python" else baum_welch(params, records, backend=backend)
    assert ours.cycles == reference.cycles
    assert ours.converged == reference.converged
    assert ours.degenerate_states == reference.degenerate_states
    for ours_array, reference_array in zip(_arrays(ours.params), _arrays(reference.params)):
        np.testing.assert_allclose(ours_array, reference_array, rtol=0, atol=PARAMS_ATOL)
    total = sum(record.length for record in records)
    assert _within(ours.description_lengths, np.array(reference.description_lengths), total)


@pytest.mark.parametrize("index", range(8))
def test_every_backend_trains_like_the_reference_on_generated_corpora(index, backend):
    _assert_trains_like_the_reference(index, backend)


def test_every_backend_trains_like_the_reference_on_a_lush_fixture(backend):
    """`m008_0001_008` over the corpus as one record, as the original trained it.

    One fixture rather than three, for time (about 10 s on torch's CPU): the kernel
    section below compares all three, and this is the one whose description
    length differed most there, by 3.1 eps.
    """
    _assert_trains_like_the_reference("fixture", backend)


def test_every_backend_refuses_an_impossible_record_before_training(backend):
    rng = np.random.default_rng(SEED)
    params = _random_params(rng, 2, 3, 0.0)
    records = [SequenceRecord(np.array([USER_BASE, UNK]))]
    with pytest.raises(ImpossibleSequenceError, match="record 0"):
        baum_welch(params, records, backend=backend)


@pytest.mark.parametrize("max_cycles", [None, 7])
def test_the_progress_log_changes_no_result_on_any_backend(max_cycles, backend):
    # Compared with ==, not within a tolerance: the log only formats values the
    # loop has already computed, so any difference at all is a defect.
    params, records = _trainings(SEED + 1, 1)[0]
    silent = baum_welch(params, records, backend=backend, max_cycles=max_cycles)
    log = io.StringIO()
    logged = baum_welch(params, records, backend=backend, max_cycles=max_cycles, log=log)

    assert (logged.cycles, logged.converged, logged.degenerate_states) == (
        silent.cycles, silent.converged, silent.degenerate_states,
    )
    assert logged.description_lengths == silent.description_lengths
    for logged_array, silent_array in zip(_arrays(logged.params), _arrays(silent.params)):
        assert np.array_equal(logged_array, silent_array)
    stop = "converged after" if max_cycles is None else "stopped at max_cycles after"
    assert log.getvalue().endswith(f"  {stop} {silent.cycles} cycles\n")


# --- kernel-level: deliberately NOT public-API -------------------------------
#
# Everything above this line calls only `baum_welch`. The tests below reach each
# kernel through the private `_resolve`, because the counts are what they assert
# and the public call returns only what the M-step made of them (ADR 0003's
# "separate, explicitly non-shared home"). They still run on every backend,
# the reference included, since every kernel owes the contract.


def _one_record(batched, device=None):
    """A batched kernel on a batch of one record, unwrapped: the `B = 1` case.

    Every count assertion below takes one unpadded record, so it goes through
    this; `.batched` is the kernel bound to its device, for `_corpus_step`.
    """
    bound = functools.partial(batched, device=device)

    def step(init, transition, output, codes):
        codes = np.asarray(codes)
        counts, bits = bound(init, transition, output, codes[np.newaxis], np.array([codes.size]))
        return tuple(array[0] for array in counts), float(bits[0])

    # `_corpus_step` passes its own `device`, None here, positionally: ignore it.
    step.batched = lambda init, transition, output, codes, lengths, _=None: bound(
        init, transition, output, codes, lengths
    )
    return step


def _cuda_for_torch():
    """Why torch cannot run on a CUDA device here, or `None` when it can."""
    status = {s.name: s for s in backends("baum_welch")}["torch"]
    if not status.available:
        return f"backend 'torch' unavailable: {status.reason}"
    import torch

    return None if torch.cuda.is_available() else "no CUDA device for torch"


#: Every (backend, device) a kernel runs on. The reference and its compiled phases
#: are CPU-only. The phases also meet the far tighter bit-exact bar of
#: `test_forward_backward_backends.py`; they run here too so the EM-trajectory and
#: subnormal-arc cases, which only this module constructs, reach them.
KERNEL_TARGETS = [("python", None), ("cython", None), ("cpu_parallel", None), ("cuda", None), ("torch", None), ("torch", "cuda")]


@pytest.fixture(scope="module", params=KERNEL_TARGETS, ids=["python", "cython", "cpu_parallel", "cuda", "torch", "torch-cuda"])
def target(request):
    name, device = request.param
    status = {s.name: s for s in backends("baum_welch")}[name]
    if not status.available:
        pytest.skip(f"backend {name!r} unavailable: {status.reason}")
    if device == "cuda" and (reason := _cuda_for_torch()):
        pytest.skip(reason)
    return request.param


@pytest.fixture(scope="module")
def kernel(target):
    name, device = target
    return _one_record(_resolve("baum_welch", name), device)


@pytest.fixture(scope="module")
def reference():
    return _one_record(_resolve("baum_welch", "python"))


def _assert_counts_match(kernel, reference, arrays, codes):
    (ours, ours_bits), (theirs, their_bits) = kernel(*arrays, codes), reference(*arrays, codes)
    n = len(codes)
    for name, a, b in zip(("init", "transition", "emission"), ours, theirs):
        assert a.shape == b.shape and a.dtype == np.float64, name
        assert (a >= 0).all(), f"{name} counts went negative: {a.min()!r}"
        assert _within(a, b, n), f"{name} counts differ by {np.abs(a - b).max()!r}"
    if np.isfinite(their_bits):
        assert _within(ours_bits, their_bits, n)
    else:
        assert ours_bits == their_bits
    return ours


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 16, 32])
@pytest.mark.parametrize("length", [1, 7, 50, 400])
def test_every_kernel_matches_the_reference_counts_on_generated_models(size, length, kernel, reference):
    rng = np.random.default_rng([SEED, size, length])
    for _ in range(4):
        params = _random_params(rng, size, 6, 0.3)
        codes = rng.integers(USER_BASE, USER_BASE + 6, length)
        counts = _assert_counts_match(kernel, reference, _arrays(params), codes)

        # The zeros that are structural are exact on every backend: a dead arc
        # carries no count, and neither does a reserved symbol or one absent from
        # the record. Zeros made by underflow are not asserted; a count can land a
        # subnormal unit either side of zero.
        dead = params.transition_p == 0
        assert not counts[1][dead].any() and not counts[2][dead].any()
        absent = np.setdiff1d(np.arange(params.n_symbols), codes)
        assert not counts[2][:, :, absent].any()


@pytest.mark.parametrize("model", SAVED_MODELS)
def test_every_kernel_matches_the_reference_counts_on_the_lush_fixtures(model, kernel, reference):
    params, record = load_params(FIXTURES / model), load_corpus_record()
    # Not a formality: fed un-renumbered codes, every fixture is impossible, every
    # count is zero on both sides, and the comparison passes having checked nothing.
    assert np.isfinite(reference(*_arrays(params), record.codes)[1])
    _assert_counts_match(kernel, reference, _arrays(params), record.codes)


@pytest.mark.parametrize("index", range(6))
def test_every_kernels_counts_re_estimate_to_a_valid_model_along_em_trajectories(index, kernel, reference):
    """Thirty cycles of the reference's training, re-estimated from each state.

    `_re_estimate` constructs an `HMMParams`, so a negative count or a fibre not
    summing to 1 raises here, as it did inside `baum_welch` for both defects the
    module docstring names. The trajectory is the reference's, so every kernel is
    tested on the same states.
    """
    params, records = _trainings(SEED + 1, 6)[index]
    for _ in range(30):
        for record in records:
            _assert_counts_match(kernel, reference, _arrays(params), record.codes)
        counts, _, _ = _corpus_step(params, records, kernel.batched)
        _re_estimate(params, counts)
        params = baum_welch(params, records, max_cycles=1).params


#: One subnormal unit. An arc carrying tens to hundreds of them is where two
#: gradients computed separately round a whole unit apart, relative to the value.
SUBNORMAL = np.nextafter(0.0, 1.0)


@pytest.mark.parametrize("index", range(12))
def test_every_kernels_counts_re_estimate_to_a_valid_model_on_a_subnormal_arc(index, kernel, reference):
    """One arc of probability 1e-322 to 2e-321, its mass moved to the diagonal.

    `output_p`'s fibre on that arc is `emission_counts / transition_counts`, so it
    sums to 1 only if the emission counts sum to the transition count. Measured
    with the transition and emission gradients taken separately, 59 of 117 such
    models re-estimated to a fibre off by up to 0.2% and `HMMParams` refused it.
    """
    rng = np.random.default_rng([SEED, index])
    size, n_user = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    base = _random_params(rng, size, n_user, 0.0)
    i, j = rng.choice(size, 2, replace=False)
    transition = base.transition_p.copy()
    transition[i, i] += transition[i, j]
    transition[i, j] = int(rng.integers(20, 400)) * SUBNORMAL
    params = HMMParams(base.init_state_p, transition, base.output_p, base.vocabulary)
    records = _random_corpus(rng, n_user, 3, 40)

    for record in records:
        _assert_counts_match(kernel, reference, _arrays(params), record.codes)
    counts, _, _ = _corpus_step(params, records, kernel.batched)
    _re_estimate(params, counts)


@pytest.mark.parametrize("codes", [np.array([], dtype=np.int64), np.array([USER_BASE, UNK])], ids=["empty", "impossible"])
def test_every_kernel_returns_zero_counts_where_there_is_nothing_to_count(codes, kernel):
    rng = np.random.default_rng(SEED)
    params = _random_params(rng, 3, 4, 0.0)
    counts, total = kernel(*_arrays(params), codes)
    assert all(not array.any() for array in counts)
    assert total == (0.0 if codes.size == 0 else np.inf)


def test_every_kernel_matches_the_reference_row_by_row_on_a_ragged_batch(kernel, reference):
    """A real padded batch, not a batch of one: empty, length-1 and impossible rows
    beside long ones, so most of the batch is padding."""
    rng = np.random.default_rng(SEED + 7)
    params = _random_params(rng, 4, 5, 0.25)
    lengths = [200, 0, 1, 1, 37]
    records = [SequenceRecord(rng.integers(USER_BASE, USER_BASE + 5, n)) for n in lengths]
    records.append(SequenceRecord(np.array([USER_BASE, UNK, USER_BASE])))
    batch = pad_collate(records)
    assert not batch["mask"].all()

    counts, bits = kernel.batched(*_arrays(params), batch["codes"], batch["lengths"])
    for b, record in enumerate(records):
        theirs, their_bits = reference(*_arrays(params), record.codes)
        n = record.length
        for name, ours, ref in zip(("init", "transition", "emission"), (c[b] for c in counts), theirs):
            assert (ours >= 0).all(), f"row {b} {name} counts went negative"
            assert _within(ours, ref, n), f"row {b} {name} counts differ by {np.abs(ours - ref).max()!r}"
        if np.isfinite(their_bits):
            assert _within(bits[b], their_bits, n)
        else:
            assert bits[b] == their_bits


# --- device= on the public call ---------------------------------------------------


def test_torch_trains_like_the_reference_on_a_cuda_device():
    if reason := _cuda_for_torch():
        pytest.skip(reason)
    # A generated corpus rather than the Lush fixture: one 1268-position record
    # takes about 30 s on the device, and the kernel tests above already compare
    # the fixtures' counts there.
    params, records, reference = _reference_run(0)
    ours = baum_welch(params, records, backend="torch", device="cuda")
    assert (ours.cycles, ours.converged, ours.degenerate_states) == (
        reference.cycles,
        reference.converged,
        reference.degenerate_states,
    )
    for ours_array, reference_array in zip(_arrays(ours.params), _arrays(reference.params)):
        np.testing.assert_allclose(ours_array, reference_array, rtol=0, atol=PARAMS_ATOL)


def _one_symbol_training():
    rng = np.random.default_rng(SEED)
    return _random_params(rng, 2, 3, 0.0), [SequenceRecord(np.array([USER_BASE, USER_BASE + 1]))]


@pytest.mark.parametrize(
    "backend_name, device, error, message",
    [
        ("python", "cuda", ValueError, "runs on the CPU only"),
        ("cuda", "cuda", ValueError, "runs on numba-cuda's current device"),
        ("cuda", "cpu", ValueError, "applies only to backend='torch'"),
        ("python", 0, TypeError, "device must be a device name"),
        ("torch", "bogus", ValueError, "not a torch device name"),
        ("torch", "cuda:99", BackendUnavailableError, "cannot allocate on device 'cuda:99'"),
    ],
    ids=["python-on-cuda", "cuda-given-cuda", "cuda-given-cpu", "not-a-string", "torch-bogus", "torch-missing-ordinal"],
)
def test_a_device_the_backend_cannot_use_is_refused_before_training(backend_name, device, error, message):
    status = {s.name: s for s in backends("baum_welch")}[backend_name]
    if not status.available:
        pytest.skip(f"backend {backend_name!r} unavailable: {status.reason}")
    params, records = _one_symbol_training()
    with pytest.raises(error, match=message):
        baum_welch(params, records, backend=backend_name, device=device)


def test_the_cpu_is_a_device_the_reference_accepts():
    params, records = _one_symbol_training()
    assert baum_welch(params, records, device="cpu", max_cycles=2).cycles == 2
