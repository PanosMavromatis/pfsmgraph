"""ADR 0021's runtime backend selection: `backend=`, `backends()` and their errors.

`test_viterbi.py` runs every decoding case on every backend; this module tests the
selection itself, which that suite can only use. Four properties are contract
rather than mechanics, and each has a test that would fail if it were loosened:

- **validation happens before any work**, so the checks here pass `None` for the
  model and record -- a check that ran after the decode began would fail with an
  `AttributeError` instead of the error asserted;
- **nothing falls back**: an unavailable backend raises `BackendUnavailableError`;
- **asking in advance and asking by calling give one answer**: the error's reason
  is the `reason` `backends()` reports;
- **one probe per process**, shared by both, and nothing is imported until asked.

Absences are simulated by planting `None` in `sys.modules`, which makes an import
raise `ModuleNotFoundError` naming that module -- exactly what a real absence does --
so an environment without numba needs no CI job of its own. Every such test starts
from an empty probe cache and removes any already-imported kernel module, or the
cached module would be returned and nothing would be simulated.

Not `test_backends.py`: that basename is taken by the repo-root suite, and with no
`__init__.py` in either directory pytest's `prepend` import mode refuses the pair.
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
import threading

import pytest

import pfsmgraph.hmm
import pfsmgraph.hmm._backends as hmm_backends
from pfsmgraph.hmm import (
    BackendStatus,
    BackendUnavailableError,
    BaumWelchResult,
    backends,
    baum_welch,
    viterbi,
)


@pytest.fixture
def fresh(monkeypatch):
    """An empty probe cache, restored afterwards; returns a function hiding modules.

    `hide("numba", "pfsmgraph.hmm._viterbi_cpu_parallel")` makes the first import
    fail and forgets the second, so the next probe really re-imports it.
    """
    monkeypatch.setattr(hmm_backends, "_cache", {})

    def hide(missing, *forget):
        monkeypatch.setitem(sys.modules, missing, None)
        for name in forget:
            monkeypatch.delitem(sys.modules, name, raising=False)

    return hide


def _status(name):
    return {s.name: s for s in backends("viterbi")}[name]


# --- backends() ---------------------------------------------------------------


def test_backends_lists_every_viterbi_backend_in_lifecycle_phase_order():
    """Ordered by phase, not by speed: availability is not recommendation."""
    assert [s.name for s in backends("viterbi")] == ["python", "cython", "cpu_parallel", "cuda"]


def test_the_reference_backend_is_always_available():
    status = _status("python")
    assert status == BackendStatus("python", True, None)


def test_a_status_is_a_frozen_value():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _status("python").available = False


def test_backends_refuses_a_name_that_is_not_a_public_call():
    with pytest.raises(ValueError, match="no public call 'decode'"):
        backends("decode")


def test_backends_refuses_a_private_kernel_name():
    """`forward_backward` has a row, for the session header, and no public call.

    ADR 0021 section 4 keys enumeration by the public call so revision 04 can
    rename or split kernels; accepting this name would publish it.
    """
    with pytest.raises(ValueError, match="known: \\['baum_welch', 'viterbi', 'viterbi_batch'\\]"):
        backends("forward_backward")


# --- validation order ------------------------------------------------------------


def test_the_default_backend_is_the_reference():
    parameter = inspect.signature(viterbi).parameters["backend"]
    assert parameter.default == "python"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("bad", ["numba", "CUDA", "cuda12", "", 3, None])
def test_a_name_that_is_no_backend_is_a_valueerror_before_any_work(bad):
    with pytest.raises(ValueError, match="backend must be one of"):
        viterbi(None, None, backend=bad)


def test_an_unimplemented_phase_is_a_valueerror_naming_what_exists():
    """ADR 0003's "contributes no parameter", seen from the API: the same everywhere.

    Reached through `_resolve` on the private algorithm, whose next missing phase is
    `cpu_parallel` since phase 2 landed (2026-09-15); `baum_welch` goes through the
    same function, and `docs/api/hmm/baum_welch.md` shows it refusing the same phase.
    """
    with pytest.raises(ValueError) as excinfo:
        hmm_backends._resolve("forward_backward", "cpu_parallel")
    assert "no 'cpu_parallel' backend" in str(excinfo.value)
    assert "['python', 'cython']" in str(excinfo.value)
    assert not isinstance(excinfo.value, BackendUnavailableError)


# --- unavailable backends --------------------------------------------------------


def test_missing_numba_is_reported_with_the_cpu_parallel_extra(fresh):
    fresh("numba", "pfsmgraph.hmm._viterbi_cpu_parallel")
    status = _status("cpu_parallel")
    assert status.available is False
    assert status.reason == "numba is not installed: pip install 'pfsmgraph-hmm[cpu-parallel]'"


def test_missing_numba_cuda_is_reported_with_the_gpu_extra(fresh):
    fresh("numba", "pfsmgraph.hmm._viterbi_cuda")
    status = _status("cuda")
    assert status.available is False
    assert status.reason == "numba-cuda is not installed: pip install 'pfsmgraph-hmm[gpu]'"


def test_missing_torch_is_reported_with_the_torch_extra(fresh):
    fresh("torch", "pfsmgraph.hmm._baum_welch_torch")
    status = {s.name: s for s in backends("baum_welch")}["torch"]
    assert status.available is False
    assert status.reason == "torch is not installed: pip install 'pfsmgraph-hmm[torch]'"


def test_a_missing_extension_is_reported_as_a_pure_install(fresh):
    """At runtime this is an ordinary pure wheel, not the broken build a checkout
    escalates -- ADR 0021 section 3's one row where the two policies differ."""
    fresh("pfsmgraph.hmm._viterbi_cython")
    status = _status("cython")
    assert status.available is False
    assert "no compiled extension" in status.reason
    assert "platform wheel" in status.reason


def test_an_extension_failing_for_another_reason_reports_the_real_error(fresh, monkeypatch, tmp_path):
    """Only the extension module itself being absent is a pure install.

    A module that exists but raises `ImportError` from inside -- a stale build
    against another numpy, a missing symbol -- must not be described as a pure
    wheel, because installing a platform wheel over it is not the fix.
    """
    (tmp_path / "stale_extension_demo.py").write_text("import a_symbol_the_build_needs\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        hmm_backends,
        "_TABLE",
        {"viterbi": (hmm_backends._Row("cython", "stale_extension_demo", "_viterbi", needs="compiled extension"),)},
    )
    (status,) = hmm_backends._status("viterbi")
    assert status.available is False
    assert status.reason.startswith("stale_extension_demo did not import")
    assert "a_symbol_the_build_needs" in status.reason


def test_an_unavailable_backend_raises_before_any_work_and_does_not_fall_back(fresh):
    fresh("numba", "pfsmgraph.hmm._viterbi_cpu_parallel")
    with pytest.raises(BackendUnavailableError) as excinfo:
        viterbi(None, None, backend="cpu_parallel")
    assert isinstance(excinfo.value, ImportError)


def test_the_error_carries_the_reason_backends_reports(fresh):
    """Asking in advance and asking by calling give one answer."""
    fresh("numba", "pfsmgraph.hmm._viterbi_cpu_parallel")
    reason = _status("cpu_parallel").reason
    with pytest.raises(BackendUnavailableError) as excinfo:
        viterbi(None, None, backend="cpu_parallel")
    assert str(excinfo.value) == f"backend 'cpu_parallel' for viterbi cannot run here: {reason}"


def test_other_backends_still_run_when_one_is_unavailable(fresh):
    fresh("numba", "pfsmgraph.hmm._viterbi_cpu_parallel", "pfsmgraph.hmm._viterbi_cuda")
    available = {s.name for s in backends("viterbi") if s.available}
    assert {"python", "cython"} <= available
    assert not {"cpu_parallel", "cuda"} & available


# --- one probe per process -------------------------------------------------------


class _CountingImport:
    """Stands in for `importlib` inside `_backends`, counting imports per module."""

    def __init__(self, real):
        self.real = real
        self.calls = []
        self.lock = threading.Lock()

    def import_module(self, name):
        with self.lock:
            self.calls.append(name)
        return self.real.import_module(name)


@pytest.fixture
def imports(monkeypatch):
    import importlib

    counter = _CountingImport(importlib)
    monkeypatch.setattr(hmm_backends, "_cache", {})
    monkeypatch.setattr(hmm_backends, "importlib", counter)
    return counter


def test_each_backend_is_probed_once_for_backends_and_calls_together(imports):
    backends("viterbi")
    backends("viterbi")
    hmm_backends._resolve("viterbi", "cython")
    hmm_backends._resolve("viterbi", "python")
    assert sorted(imports.calls) == sorted(r.module for r in hmm_backends._TABLE["viterbi"])


def test_a_call_probes_only_the_backend_it_names(imports):
    """The lazy half: a decode on `"python"` never pays for a numba or CUDA probe."""
    hmm_backends._resolve("viterbi", "python")
    assert imports.calls == ["pfsmgraph.hmm._viterbi"]


def test_concurrent_first_calls_probe_once(imports):
    """The lock: two threads must not both query for a CUDA device."""
    barrier = threading.Barrier(8)

    def ask():
        barrier.wait()
        backends("viterbi")

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(imports.calls) == len(hmm_backends._TABLE["viterbi"])


def test_importing_the_package_loads_no_backend_dependency():
    """Checked in a fresh interpreter, since this one imported numba long ago."""
    code = (
        "import sys, pfsmgraph.hmm; "
        "print(sorted(m for m in ('numba', 'pfsmgraph.hmm._viterbi_cython', "
        "'pfsmgraph.hmm._viterbi_cuda', 'torch', 'pfsmgraph.hmm._baum_welch_torch') "
        "if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- the public surface ----------------------------------------------------------


def test_the_selection_names_are_exported():
    for name in ("backends", "BackendStatus", "BackendUnavailableError", "baum_welch", "BaumWelchResult"):
        assert name in pfsmgraph.hmm.__all__
        assert getattr(pfsmgraph.hmm, name) is globals()[name]
