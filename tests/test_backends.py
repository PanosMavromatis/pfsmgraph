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


# --- the matrix as it stands -------------------------------------------------

def test_the_python_backend_is_the_whole_matrix():
    # Filled 2026-09-04 by pfsmgraph.hmm._viterbi, the first DP kernel to reach
    # ADR 0002 phase 1. This was `BACKENDS == ()` until then, and its comment
    # said it would fail when align or hmm added the first row -- which is what
    # happened, and is why the surrounding docs were revisited in the same
    # commit. Adding the *second* row should break this one the same way.
    assert BACKENDS == (Backend("python", "pfsmgraph.hmm._viterbi", None),)


def test_the_python_backend_escalates_rather_than_skips():
    # hardware=None is the whole claim: nothing external is needed to run pure
    # Python, so a failed import is a broken working copy and never a skip.
    (python,) = BACKENDS
    assert python.hardware is None


def test_the_registered_module_is_the_kernel_not_the_package():
    # `import pfsmgraph.hmm` succeeds whether or not a decode exists in it, so
    # the package would be a row that cannot fail. The row is a claim about one
    # lifecycle phase of one algorithm, so it names the module carrying it.
    (python,) = BACKENDS
    assert python.module.rsplit(".", 1)[-1] == "_viterbi"


def test_the_registered_backend_actually_resolves():
    # The row is not aspirational: this is the import probe running against the
    # real matrix rather than a synthetic one.
    assert detect() == (Availability("python", True, None),)


def test_the_header_names_the_registered_backend():
    assert format_header(detect()) == "backends: python ✓"


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


def test_missing_hardware_backend_is_a_reported_skip():
    (state,) = detect([Backend("cuda", "pfsmgraph._absent", hardware="CUDA device")])
    assert state.available is False
    assert state.reason == "no CUDA device detected"


def test_unimportable_backend_without_hardware_is_a_hard_failure():
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
