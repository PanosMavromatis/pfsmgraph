"""Runtime backend selection: the table, the probe, and the public enumeration.

[ADR 0021] settles how a caller chooses among a DP kernel's [ADR 0002] lifecycle
phases, and this module is where that decision lives:

- **A backend is a per-call keyword**, ``backend=``, named by one of
  :data:`BACKEND_NAMES`. There is no module-level default and no setter.
- **The default is ``"python"``**, and a requested backend that cannot run here
  raises :class:`BackendUnavailableError`. **Nothing falls back.**
- **The table is here**, in the distribution that owns the kernels, keyed by
  algorithm and then by backend. The repo-root ``_backends.py`` holds ADR 0003's
  test policy and reads this table rather than keeping one.
- **Enumeration is public**: :func:`backends` reports every backend a call has,
  available or not, with the reason.

**One probe per process.** A row is probed at most once -- by :func:`backends`, by
:func:`_status`, or by the first call naming it -- and the outcome is cached, so
asking in advance and asking by calling give the same answer, and a CUDA device
query runs once rather than on every decode. A device that appears mid-process is
not noticed. The probe is an import, and nothing is imported until something asks:
``import pfsmgraph.hmm`` loads no numba.

**The table carries private algorithms too.** ``forward_backward`` has a row so the
session header can report it, but it is not a public call, so :func:`backends`
refuses its name; :func:`_status` accepts it. ``baum_welch`` is public, and its
rows are per-record E-steps rather than forward-backward kernels, since its
``torch`` backend builds no β. Kernel module names are private and
revision 04 is free to change them, which is why the public key is the call.

.. [ADR 0002] ``docs/design/adr/0002-three-phase-algorithm-lifecycle.md``
.. [ADR 0021] ``docs/design/adr/0021-runtime-backend-selection.md``
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Callable, Final, Literal

__all__ = ["BackendName", "BackendStatus", "BackendUnavailableError", "backends"]

BackendName = Literal["python", "cython", "cpu_parallel", "cuda", "torch"]

#: Every backend name: the ADR 0002 lifecycle phases in order, then ``torch``,
#: which is no phase but a second derivation held to the reference within
#: ADR 0020's tolerance. The same vocabulary as the session header and
#: ``PFSMGRAPH_REQUIRE_BACKENDS``.
BACKEND_NAMES: Final[tuple[str, ...]] = ("python", "cython", "cpu_parallel", "cuda", "torch")

#: What an absence can be attributable to. ``None`` means nothing external: a
#: failed import is a broken install. The repo-root test policy escalates ``None``
#: and ``"compiled extension"`` and skips the other three (ADR 0021 section 3).
Needs = Literal["compiled extension", "numba", "CUDA device", "torch"]


class BackendUnavailableError(ImportError):
    """A requested backend exists for this call but cannot run in this environment.

    An :class:`ImportError` because that is what it is -- a dependency could not
    be imported, or a device could not be found -- so code that already guards an
    optional import catches it. The message names what is missing and what
    supplies it. It is raised before any work, and nothing falls back.
    """


@dataclass(frozen=True)
class BackendStatus:
    """One backend of one public call, as probed in this process.

    :param name: the value to pass as ``backend=``.
    :param available: whether that call would run here. Branch on this.
    :param reason: ``None`` when available; otherwise the text
        :class:`BackendUnavailableError` would carry. It is prose for a person
        and may be reworded, so do not match on it.
    """

    name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class _Row:
    name: str
    module: str
    function: str
    needs: Needs | None = None
    extra: str | None = None


_TABLE: Final[dict[str, tuple[_Row, ...]]] = {
    "viterbi": (
        _Row("python", "pfsmgraph.hmm._viterbi", "_viterbi"),
        _Row("cython", "pfsmgraph.hmm._viterbi_cython", "_viterbi", needs="compiled extension"),
        _Row(
            "cpu_parallel",
            "pfsmgraph.hmm._viterbi_cpu_parallel",
            "_viterbi",
            needs="numba",
            extra="cpu-parallel",
        ),
        _Row("cuda", "pfsmgraph.hmm._viterbi_cuda", "_viterbi", needs="CUDA device", extra="gpu"),
    ),
    # Phase 1 only. Private, so it is here for the session header and absent
    # from _PUBLIC; baum_welch reaches it through its python row's E-step.
    "forward_backward": (
        _Row("python", "pfsmgraph.hmm._forward_backward", "_forward_backward"),
    ),
    # Keyed by the public call (ADR 0021 section 4), and its rows are E-steps,
    # not forward-backward kernels: torch derives the counts as gradients and
    # builds no beta, so the step is the one signature both share.
    "baum_welch": (
        _Row("python", "pfsmgraph.hmm._forward_backward", "_e_step"),
        _Row("torch", "pfsmgraph.hmm._baum_welch_torch", "_e_step", needs="torch", extra="torch"),
    ),
}

#: The names :func:`backends` accepts: public calls, never private kernels.
_PUBLIC: Final = frozenset({"viterbi", "baum_welch"})

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[Callable | None, str | None]] = {}


def _reason(row: _Row, exc: ImportError) -> str:
    """The remedy text for a failed probe, derived from the row."""
    missing = exc.name.split(".", 1)[0] if isinstance(exc, ModuleNotFoundError) and exc.name else None
    if row.needs is None:
        return f"{row.module} did not import, which means a broken install: {exc}"
    if row.needs == "compiled extension":
        # Only the extension module itself being absent is a pure install. Any
        # other ImportError from inside it is a broken or stale build, and says so.
        if exc.name != row.module:
            return f"{row.module} did not import: {exc}"
        return (
            "this install of pfsmgraph-hmm has no compiled extension (a pure wheel, "
            "or a build with -Dcompiled=false): install a platform wheel, or build "
            "from source with a C compiler"
        )
    if row.needs == "numba":
        if missing == "numba":
            return f"numba is not installed: pip install 'pfsmgraph-hmm[{row.extra}]'"
        return f"numba did not import: {exc}"
    if row.needs == "torch":
        if missing == "torch":
            return f"torch is not installed: pip install 'pfsmgraph-hmm[{row.extra}]'"
        return f"torch did not import: {exc}"
    # CUDA device. The kernel module raises ImportError itself when numba-cuda
    # reports no device, because @cuda.jit decorates lazily and an import would
    # otherwise succeed on a machine that cannot run it.
    if missing == "numba":
        return f"numba-cuda is not installed: pip install 'pfsmgraph-hmm[{row.extra}]'"
    if str(exc).startswith("no CUDA device detected"):
        return "no CUDA device detected"
    return f"numba-cuda did not import: {exc}"


def _probe(algorithm: str, row: _Row) -> tuple[Callable | None, str | None]:
    key = (algorithm, row.name)
    with _lock:
        if key not in _cache:
            try:
                module = importlib.import_module(row.module)
            except ImportError as exc:
                _cache[key] = (None, _reason(row, exc))
            else:
                _cache[key] = (getattr(module, row.function), None)
        return _cache[key]


def _rows(algorithm: str) -> tuple[_Row, ...]:
    try:
        return _TABLE[algorithm]
    except KeyError:
        raise ValueError(
            f"no backend table for {algorithm!r}; known: {sorted(_TABLE)}"
        ) from None


def _status(algorithm: str) -> tuple[BackendStatus, ...]:
    """Probe every backend of ``algorithm``, private algorithms included."""
    states = []
    for row in _rows(algorithm):
        _, reason = _probe(algorithm, row)
        states.append(BackendStatus(row.name, reason is None, reason))
    return tuple(states)


def _resolve(algorithm: str, backend: object, call: str | None = None) -> Callable:
    """The kernel function for ``backend``, validated in ADR 0021's order.

    :param call: the public name to use in messages; defaults to ``algorithm``.
    :raises ValueError: ``backend`` is not a backend name, or names a lifecycle
        phase this algorithm has not reached. Both fail identically everywhere.
    :raises BackendUnavailableError: the backend exists but cannot run here.
    """
    call = call or algorithm
    if backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {list(BACKEND_NAMES)}, got {backend!r}")
    rows = {row.name: row for row in _rows(algorithm)}
    if backend not in rows:
        raise ValueError(
            f"{call} has no {backend!r} backend: that lifecycle phase is not "
            f"implemented for it. It has {list(rows)}"
        )
    kernel, reason = _probe(algorithm, rows[backend])
    if kernel is None:
        raise BackendUnavailableError(f"backend {backend!r} for {call} cannot run here: {reason}")
    return kernel


def backends(name: str) -> tuple[BackendStatus, ...]:
    """Report every backend the public call ``name`` has, in lifecycle-phase order.

    Unavailable backends are included, with the reason a call naming them would
    raise. A backend whose phase has not been implemented is absent. The first
    call probes, which may import numba or query for a CUDA device; later calls,
    and later ``backend=`` resolution, read the cached result.

    :param name: a public call, e.g. ``"viterbi"``.
    :raises ValueError: ``name`` is not a public call with backends.
    """
    if name not in _PUBLIC:
        raise ValueError(f"no public call {name!r} has backends; known: {sorted(_PUBLIC)}")
    return _status(name)
