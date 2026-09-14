"""Tests for the ADR 0003 backend matrix and its session header.

These cover the reporting mechanism itself, which nothing else touches: every
other suite here tests library behaviour, and a broken header would fail none of
them. ADR 0003's premise is that an unexercised backend must say so, which makes
the thing doing the saying worth testing directly.

Since 2026-09-14 the table is ``pfsmgraph/hmm/_backends.py``'s (ADR 0021) and the
repo-root module is the policy over it, so the tests about which rows exist read
that table, and the synthetic cases swap rows into it -- which also exercises the
real probe and the real reason text rather than a stand-in. The package's own
tests of ``backends()`` are in ``packages/pfsmgraph-hmm/tests/test_backend_selection.py``.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

import pfsmgraph.hmm._backends as hmm_backends
from _backends import (
    EMPTY_HEADER,
    ESCALATED_NEEDS,
    REQUIRE_ENV,
    TABLES,
    Availability,
    BackendError,
    check_required,
    detect,
    format_header,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
Row = hmm_backends._Row


def _expected_cuda_reason() -> str | None:
    """Ask numba-cuda directly, never through `_viterbi_cuda`.

    Deriving the expectation from the module under test would make the `detect()`
    assertion a tautology: a kernel module that forgot its device check would
    import, report available, and agree with itself. This asks the question the
    module's guard asks, from outside it.
    """
    try:
        from numba import cuda
    except ImportError:
        return "numba-cuda is not installed: pip install 'pfsmgraph-hmm[gpu]'"
    return None if cuda.is_available() else "no CUDA device detected"


_CUDA_REASON = _expected_cuda_reason()


@pytest.fixture
def table(monkeypatch):
    """Swap a synthetic table into `pfsmgraph.hmm._backends`, with an empty cache.

    Returns a setter; the real table and cache are restored afterwards.
    """
    monkeypatch.setattr(hmm_backends, "_cache", {})

    def use(rows_by_algorithm):
        monkeypatch.setattr(hmm_backends, "_TABLE", rows_by_algorithm)
        return [hmm_backends]

    return use


# --- the matrix as it stands -------------------------------------------------

def test_the_policy_reads_the_hmm_table_and_holds_none_of_its_own():
    # ADR 0021 section 3: the table ships with the kernels; this module is policy.
    # A second table here would be two lists of the same backends that can disagree.
    import _backends

    assert TABLES == ("pfsmgraph.hmm._backends",)
    assert not hasattr(_backends, "BACKENDS")


def test_viterbi_has_all_four_phases_in_lifecycle_order_and_forward_backward_one():
    # Order is asserted, not just membership, because format_header prints the rows
    # in this order and ADR 0003's header is a specified string. Viterbi completed its
    # lifecycle on 2026-09-13; forward_backward is at phase 1 with no public call, and
    # is here so the header can say so. baum_welch is public, and its rows are
    # E-steps: the reference and torch's gradients.
    assert {a: [r.name for r in rows] for a, rows in hmm_backends._TABLE.items()} == {
        "viterbi": ["python", "cython", "cpu_parallel", "cuda"],
        "viterbi_batch": ["python"],
        "forward_backward": ["python"],
        "baum_welch": ["python", "torch"],
    }


def test_only_the_numba_and_cuda_rows_may_be_skipped():
    # `needs` is the whole claim, and it means something different on each row.
    # `python` needs nothing external. `cython`'s source is committed but the
    # extension exists only if it was built, so in a checkout its absence is a
    # missing or stale build -- escalated -- while at runtime it is a pure wheel.
    # `cpu_parallel` needs numba, which no install of pfsmgraph-hmm is promised
    # (ADR 0004), and `cuda` a device. Pin the values, not their truthiness: moving
    # a row into ESCALATED_NEEDS would silently turn a legitimate skip into a
    # startup failure for every lean install.
    assert [(r.name, r.needs) for r in hmm_backends._TABLE["viterbi"]] == [
        ("python", None),
        ("cython", "compiled extension"),
        ("cpu_parallel", "numba"),
        ("cuda", "CUDA device"),
    ]
    assert [(r.name, r.needs) for r in hmm_backends._TABLE["viterbi_batch"]] == [
        ("python", None),
    ]
    # torch, like numba, is promised to no install of pfsmgraph-hmm: an extra.
    assert [(r.name, r.needs) for r in hmm_backends._TABLE["baum_welch"]] == [
        ("python", None),
        ("torch", "torch"),
    ]
    assert ESCALATED_NEEDS == {None, "compiled extension"}


def test_every_registered_module_is_a_kernel_not_a_package():
    # `import pfsmgraph.hmm` succeeds whether or not a decode exists in it, so the
    # package would be a row that cannot fail. Each row is a claim about one
    # lifecycle phase of one algorithm, so it names the module carrying it.
    modules = [r.module.rsplit(".", 1)[-1] for rows in hmm_backends._TABLE.values() for r in rows]
    assert modules == [
        "_viterbi",
        "_viterbi_cython",
        "_viterbi_cpu_parallel",
        "_viterbi_cuda",
        "_viterbi",
        "_forward_backward",
        "_forward_backward",
        "_baum_welch_torch",
    ]


def test_the_registered_backends_actually_resolve():
    # The rows are not aspirational: this is the probe running against the real
    # table. For `cython` it is also the assertion that the extension was built --
    # if not, detect() raises BackendError here rather than returning a skip.
    # `cpu_parallel` resolves because numba is in the root `dev` group. `cuda`
    # resolves only where a device exists, so its expectation follows the device;
    # what stays pinned is the *shape* of an absence -- a skip with the reason.
    assert detect() == (
        Availability("viterbi", "python", True, None),
        Availability("viterbi", "cython", True, None),
        Availability("viterbi", "cpu_parallel", True, None),
        Availability("viterbi", "cuda", _CUDA_REASON is None, _CUDA_REASON),
        Availability("viterbi_batch", "python", True, None),
        Availability("forward_backward", "python", True, None),
        Availability("baum_welch", "python", True, None),
        Availability("baum_welch", "torch", True, None),
    )


def test_the_header_names_every_registered_backend():
    cuda_cell = "cuda ✓" if _CUDA_REASON is None else f"cuda ✗ ({_CUDA_REASON})"
    assert format_header(detect()) == (
        "backends: viterbi python ✓ · cython ✓ · cpu_parallel ✓ · "
        + cuda_cell
        + " | viterbi_batch python ✓"
        + " | forward_backward python ✓ | baum_welch python ✓ · torch ✓"
    )


def test_the_header_and_backends_report_the_same_reason():
    # One probe per process: the header is built from the same cached status the
    # public call returns, so a user and the test run cannot be told different things.
    public = {s.name: s.reason for s in hmm_backends.backends("viterbi")}
    header = {s.name: s.reason for s in detect() if s.algorithm == "viterbi"}
    assert header == public


def test_an_empty_matrix_would_still_print_explicitly():
    # Not what a run prints, but ADR 0003 requires that a matrix with nothing in it
    # says so in as many words, because a missing line is indistinguishable from a
    # hook that was never registered. Kept under test so the message cannot rot.
    assert format_header(()) == EMPTY_HEADER
    assert "none registered" in EMPTY_HEADER


# --- format_header -----------------------------------------------------------

def test_header_format_groups_by_algorithm_on_one_line():
    states = (
        Availability("viterbi", "python", True),
        Availability("viterbi", "cython", True),
        Availability("viterbi", "cuda", False, "no CUDA device detected"),
        Availability("forward_backward", "python", True),
    )
    assert format_header(states) == (
        "backends: viterbi python ✓ · cython ✓ · cuda ✗ (no CUDA device detected)"
        " | forward_backward python ✓"
    )
    assert "\n" not in format_header(states)


# --- detect ------------------------------------------------------------------

def test_importable_backend_is_available(table):
    (state,) = detect(table({"demo": (Row("python", "os", "getcwd"),)}))
    assert state == Availability("demo", "python", True, None)


def test_an_unbuilt_compiled_backend_escalates_rather_than_skipping(table):
    # The failure mode the `cython` row exists to catch: a backend whose source is
    # committed but whose extension was never built. At runtime the same absence is
    # a pure wheel and reports unavailable; in a checkout a skip would render it as
    # "not available here", indistinguishable from a phase never written.
    rows = {"demo": (Row("cython", "pfsmgraph.hmm._never_built", "_demo", needs="compiled extension"),)}
    with pytest.raises(BackendError, match="implemented but"):
        detect(table(rows))


def test_backend_missing_numba_is_a_reported_skip_naming_the_extra(table):
    rows = {"demo": (Row("cpu_parallel", "numba_is_absent_here", "_demo", needs="numba", extra="cpu-parallel"),)}
    (state,) = detect(table(rows))
    assert state.available is False
    # The probe names the module that was missing; this synthetic module is not
    # numba, so the reason reports the import error rather than the extra.
    assert "did not import" in state.reason


def test_unimportable_backend_needing_nothing_is_a_hard_failure(table):
    rows = {"demo": (Row("python", "pfsmgraph._absent", "_demo"),)}
    with pytest.raises(BackendError, match="hard failure"):
        detect(table(rows))


def test_a_table_module_that_will_not_import_is_a_hard_failure(monkeypatch):
    import _backends

    monkeypatch.setattr(_backends, "TABLES", ("pfsmgraph.hmm._no_such_table",))
    with pytest.raises(BackendError, match="did not import"):
        _backends.detect()


# --- check_required ----------------------------------------------------------

STATES = (
    Availability("viterbi", "python", True),
    Availability("viterbi", "cuda", False, "no CUDA device detected"),
    Availability("forward_backward", "python", True),
)


def test_no_requirement_is_a_no_op():
    assert check_required(STATES, {}) is None
    assert check_required(STATES, {REQUIRE_ENV: "  ,  "}) is None


def test_required_and_available_passes():
    assert check_required(STATES, {REQUIRE_ENV: "python"}) is None


def test_requiring_python_passes_against_the_real_matrix():
    # The CI escalation, resolved against the real table rather than a fixture:
    # this is what PFSMGRAPH_REQUIRE_BACKENDS=python does on a runner today.
    assert check_required(detect(), {REQUIRE_ENV: "python"}) is None


def test_required_but_skipped_is_escalated_naming_the_algorithm():
    with pytest.raises(BackendError, match="would have skipped: viterbi cuda"):
        check_required(STATES, {REQUIRE_ENV: "python,cuda"})


def test_a_required_name_covers_every_algorithm_that_has_it():
    states = STATES + (Availability("forward_backward", "cuda", False, "no CUDA device detected"),)
    with pytest.raises(BackendError) as excinfo:
        check_required(states, {REQUIRE_ENV: "cuda"})
    assert "viterbi cuda" in str(excinfo.value)
    assert "forward_backward cuda" in str(excinfo.value)


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
    # than calling the hook by hand. The copied _backends.py reads the table out
    # of the installed pfsmgraph.hmm, which is also a check that the policy
    # survives being run from outside the repo root.
    for name in ("conftest.py", "_backends.py"):
        shutil.copy(REPO_ROOT / name, pytester.path / name)
    pytester.makepyfile(test_trivial="def test_trivial(): pass")
    result = pytester.runpytest_subprocess()
    result.stdout.fnmatch_lines(["backends: viterbi python ✓*| forward_backward python ✓ | baum_welch python ✓ · torch ✓"])
    result.assert_outcomes(passed=1)
