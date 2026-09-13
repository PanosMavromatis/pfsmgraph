"""Tests for the ADR 0003 backend matrix and its session header.

These cover the reporting mechanism itself, which nothing else touches: every
other suite here tests library behaviour, and a broken header would fail none of
them. ADR 0003's premise is that an unexercised backend must say so, which makes
the thing doing the saying worth testing directly.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from _backends import (
    BACKENDS,
    EMPTY_HEADER,
    REQUIRE_ENV,
    Availability,
    Backend,
    BackendError,
    check_required,
    detect,
    format_header,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _device_present() -> bool:
    """Ask numba-cuda directly, never through `_viterbi_cuda`.

    Deriving the expectation from the module under test would make the `detect()`
    assertion a tautology: a kernel module that forgot its device check would
    import, report available, and agree with itself. This asks the question the
    module's guard asks, from outside it.
    """
    try:
        from numba import cuda
    except ImportError:
        return False
    return bool(cuda.is_available())


_EXPECTED_CUDA = (
    Availability("cuda", True, None)
    if _device_present()
    else Availability("cuda", False, "no CUDA device detected")
)


# --- the matrix as it stands -------------------------------------------------

def test_the_matrix_is_python_then_cython_then_cpu_parallel_then_cuda():
    # Filled 2026-09-04 by pfsmgraph.hmm._viterbi, the first DP kernel to reach
    # ADR 0002 phase 1. This was `BACKENDS == ()` until then, and its comment
    # said it would fail when align or hmm added the first row -- which is what
    # happened. The previous version of this test then said "adding the *second*
    # row should break this one the same way", and on 2026-09-09 it did: phase 2
    # landed as _viterbi_cython. On 2026-09-10 it broke a third time, for phase
    # 3 -- _viterbi_cpu_parallel. Order is asserted, not just membership, because
    # format_header prints the rows in this order and ADR 0003's header is a
    # specified string. Rows are in lifecycle order. It broke a fourth time on
    # 2026-09-13 for phase 4's _viterbi_cuda, the first row to carry a real
    # hardware absence, and that completes the Viterbi lifecycle -- so the next
    # break is a new algorithm rather than a new phase.
    assert BACKENDS == (
        Backend("python", "pfsmgraph.hmm._viterbi", None),
        Backend("cython", "pfsmgraph.hmm._viterbi_cython", None),
        Backend("cpu_parallel", "pfsmgraph.hmm._viterbi_cpu_parallel", "numba"),
        Backend("cuda", "pfsmgraph.hmm._viterbi_cuda", "CUDA device"),
    )


def test_only_the_numba_rows_may_be_skipped():
    # optional_on is the whole claim, and it means something different on each
    # row. For `python`, nothing external is needed to run pure Python at all.
    # For `cython`, the source is committed but the extension exists only if it
    # was built -- so ADR 0003's "implemented but not importable (missing or
    # stale Cython build) is a hard failure" is the clause doing the work, and a
    # missing compiler is a broken working copy rather than a legitimate
    # absence.
    #
    # `cpu_parallel` is the first row where the absence *is* legitimate, and the
    # reason it is not a broken working copy is a packaging decision rather than
    # a property of the kernel: numba is not a dependency of pfsmgraph-hmm (ADR
    # 0004 -- acceleration is opt-in; the extra is withheld in 0.1.0 and optional
    # when restored), so an install without it is entitled to lack this backend. Pin the value, not just
    # its truthiness: making it None would silently escalate a legitimate skip
    # into a startup failure for every lean install.
    #
    # `cuda` is the absence the field was first named for, and the only row whose
    # probe can fail with every package installed: `_viterbi_cuda` raises
    # ImportError when numba-cuda reports no device, because `@cuda.jit`
    # decorates lazily and a plain import would otherwise succeed on a machine
    # with no GPU.
    assert [b.optional_on for b in BACKENDS] == [None, None, "numba", "CUDA device"]


def test_every_registered_module_is_a_kernel_not_a_package():
    # `import pfsmgraph.hmm` succeeds whether or not a decode exists in it, so
    # the package would be a row that cannot fail. Each row is a claim about one
    # lifecycle phase of one algorithm, so it names the module carrying it.
    assert [b.module.rsplit(".", 1)[-1] for b in BACKENDS] == [
        "_viterbi",
        "_viterbi_cython",
        "_viterbi_cpu_parallel",
        "_viterbi_cuda",
    ]


def test_the_registered_backends_actually_resolve():
    # The rows are not aspirational: this is the import probe running against
    # the real matrix rather than a synthetic one. For `cython` it is also the
    # only assertion in the suite that the extension was actually built -- if it
    # was not, detect() raises BackendError here rather than returning a skip.
    #
    # `cpu_parallel` resolves here because numba is in the root `dev` group, not
    # because it is required: no install of the package promises numba, and the
    # dev group is what makes this repository's own suite exercise the backend. A parallel
    # backend nobody runs is worse than none, which is why both halves exist.
    #
    # `cuda` is different in kind: it resolves only where a device exists, and
    # asserting it unconditionally would make this repository's suite fail on
    # every machine without a GPU, which is the one absence ADR 0003 calls
    # legitimate. So the expected row follows the device. What stays pinned is
    # the *shape* of an absence -- a skip with the reason, never an escalation.
    assert detect() == (
        Availability("python", True, None),
        Availability("cython", True, None),
        Availability("cpu_parallel", True, None),
        _EXPECTED_CUDA,
    )


def test_the_header_names_every_registered_backend():
    cuda_cell = "cuda ✓" if _EXPECTED_CUDA.available else "cuda ✗ (no CUDA device detected)"
    assert format_header(detect()) == (
        "backends: python ✓ · cython ✓ · cpu_parallel ✓ · " + cuda_cell
    )


def test_an_empty_matrix_would_still_print_explicitly():
    # EMPTY_HEADER is no longer what a run prints, but the branch is still live:
    # ADR 0003 requires that a matrix with nothing in it says so in as many
    # words, because a missing line is indistinguishable from a hook that was
    # never registered. Kept under test so the message cannot rot unnoticed.
    assert format_header(()) == EMPTY_HEADER
    assert "none registered" in EMPTY_HEADER


# --- format_header -----------------------------------------------------------

def test_header_format_matches_adr_0003():
    states = (
        Availability("python", True),
        Availability("cython", True),
        Availability("cuda", False, "no CUDA device detected"),
    )
    assert format_header(states) == (
        "backends: python ✓ · cython ✓ · cuda ✗ (no CUDA device detected)"
    )


# --- detect ------------------------------------------------------------------

def test_importable_backend_is_available():
    (state,) = detect([Backend("python", "os")])
    assert state == Availability("python", True, None)


def test_an_unbuilt_compiled_backend_escalates_rather_than_skipping():
    # The failure mode the `cython` row exists to catch, exercised against a
    # synthetic row so the real one stays untouched: a backend whose source is
    # committed but whose extension was never built. ADR 0003 makes this a hard
    # failure precisely because a skip would render it as "not available here",
    # which is indistinguishable from a phase that was never written.
    with pytest.raises(BackendError, match="implemented but"):
        detect([Backend("cython", "pfsmgraph.hmm._viterbi_never_built")])


def test_backend_missing_its_optional_dependency_is_a_reported_skip():
    (state,) = detect([Backend("cuda", "pfsmgraph._absent", optional_on="CUDA device")])
    assert state.available is False
    assert state.reason == "no CUDA device detected"


def test_unimportable_backend_without_optional_on_is_a_hard_failure():
    # The stale-or-missing Cython build. ADR 0003 forbids skipping this.
    with pytest.raises(BackendError, match="hard failure"):
        detect([Backend("cython", "pfsmgraph._absent")])


# --- check_required ----------------------------------------------------------

STATES = (
    Availability("python", True),
    Availability("cuda", False, "no CUDA device detected"),
)


def test_no_requirement_is_a_no_op():
    assert check_required(STATES, {}) is None
    assert check_required(STATES, {REQUIRE_ENV: "  ,  "}) is None


def test_required_and_available_passes():
    assert check_required(STATES, {REQUIRE_ENV: "python"}) is None


def test_requiring_python_passes_against_the_real_matrix():
    # The CI escalation, resolved against BACKENDS rather than a fixture: this
    # is what PFSMGRAPH_REQUIRE_BACKENDS=python does on a runner today.
    assert check_required(detect(), {REQUIRE_ENV: "python"}) is None


def test_required_but_skipped_is_escalated():
    with pytest.raises(BackendError, match="would have skipped"):
        check_required(STATES, {REQUIRE_ENV: "python,cuda"})


def test_unknown_required_name_is_rejected_before_availability():
    # A CI believing in a backend this working copy never had must fail, not
    # pass because nothing by that name was missing.
    with pytest.raises(BackendError, match="not in the matrix"):
        check_required(STATES, {REQUIRE_ENV: "cudaa"})


def test_unknown_name_rejected_even_against_an_empty_matrix():
    with pytest.raises(BackendError, match="registered: none"):
        check_required((), {REQUIRE_ENV: "cuda"})


# --- wiring ------------------------------------------------------------------

def test_conftest_is_at_the_repo_root():
    # pytest_report_header is a startup hook; a conftest under packages/*/tests/
    # is loaded during collection and its hook is discarded silently.
    assert (REPO_ROOT / "conftest.py").is_file()
    assert (REPO_ROOT / "_backends.py").is_file()


def test_header_actually_reaches_the_session_output(pytester):
    # End to end in a subprocess, so it exercises the real startup path rather
    # than calling the hook by hand. The copied _backends.py resolves
    # pfsmgraph.hmm._viterbi out of the same venv, which is also a check that
    # the row survives being probed from outside the repo root.
    for name in ("conftest.py", "_backends.py"):
        shutil.copy(REPO_ROOT / name, pytester.path / name)
    pytester.makepyfile(test_trivial="def test_trivial(): pass")
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(["backends: python*"])
    result.assert_outcomes(passed=1)
