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

def test_registry_is_empty_until_a_dp_kernel_lands():
    # Not a placeholder: ADR 0002 scopes backends to dynamic programming and
    # dataseq has none. This fails when align or hmm adds the first row, which
    # is exactly when the surrounding docs need revisiting.
    assert BACKENDS == ()


def test_header_reports_an_empty_matrix_explicitly():
    assert format_header(detect()) == EMPTY_HEADER
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
    for name in ("conftest.py", "_backends.py"):
        shutil.copy(REPO_ROOT / name, pytester.path / name)
    pytester.makepyfile(test_trivial="def test_trivial(): pass")
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(["backends: none registered*"])
    result.assert_outcomes(passed=1)
