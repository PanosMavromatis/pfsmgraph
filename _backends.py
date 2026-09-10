"""Backend availability for the ADR 0003 parameterized suites.

Test-only infrastructure. Nothing under ``packages/`` imports this module and no
built artifact contains it, which is deliberate: enumerating backends is most of
what a runtime backend-selection API needs, and ADR 0003 leaves that API's
behaviour explicitly open. Shipping the enumeration now would prejudge it.

:data:`BACKENDS` holds two rows as of 2026-09-09. It was empty until 2026-09-04, and
that emptiness was the correct steady state rather than a placeholder: ADR 0002
scopes the lifecycle to "wherever dynamic programming appears", and ``dataseq``
contributes no backend at any maturity -- it is a container and an encoder. The
condition named here for filling the matrix -- ``align`` or ``hmm`` landing a
recurrence -- was met by ``pfsmgraph.hmm._viterbi``, the Viterbi decode at ADR
0002 phase 1. The second row is ``pfsmgraph.hmm._viterbi_cython``, the same
algorithm at phase 2 -- the first compiled backend this repository has had, and
the first row whose absence would mean a *build* is broken rather than a source
tree.

**Two rows, and the algorithm suites are still not parameterized over them.** ADR 0003 asks for one suite per algorithm run against
every available backend, with the backend as a fixture parameter and the tests
"written against the public API only". Both halves cannot hold yet:
``viterbi(params, record)`` has nowhere to put a backend, and giving it one is
the runtime backend-selection API that ADR 0003's own Open section defers --
"settle this when ``align`` acquires a backend-selection API; it warrants its
own record". So the header reports honestly today and the parameterization
arrives with the seam, not before. The intervening cost is named rather than
hidden: until then, this table says which phases *exist*, not which ones the
suite exercises.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

#: Printed when no backend is registered. ADR 0003 requires the header print even
#: then: a missing line is indistinguishable from a hook that is absent, sited in
#: the wrong directory, or that raised and was swallowed.
EMPTY_HEADER: Final = "backends: none registered — no DP kernel has reached ADR 0002 phase 1"

REQUIRE_ENV: Final = "PFSMGRAPH_REQUIRE_BACKENDS"


class BackendError(Exception):
    """A backend problem ADR 0003 requires be a hard failure, never a skip."""


@dataclass(frozen=True)
class Backend:
    """One row of the matrix.

    A backend appears here only once its lifecycle phase is *implemented*. ADR
    0003: a phase not yet reached "contributes no parameter at all", because
    absence is the honest representation of "not written yet" while a skip means
    "written, but not runnable here". Membership of this table is therefore the
    record of which phases exist -- deliberately not derived by probing for build
    products, since a Cython backend whose ``.so`` was never built would then
    read as unwritten instead of broken.

    :param name: what the header calls it -- ``python``, ``cython``, ``cuda``.
    :param module: the import proving it is usable in this environment.
    :param optional_on: what an absence may legitimately be blamed on -- a short
        noun phrase, e.g. ``"CUDA device"`` or ``"numba"``. ``None`` means
        nothing external is required, so a failed import is a broken working
        copy and is escalated.

    Called ``hardware`` until 2026-09-10. Phase 3 produced the first backend
    whose absence is legitimate but is not a *device*: an environment that
    declined the ``cpu-parallel`` extra is entitled to lack numba exactly as one
    without a GPU is entitled to lack a CUDA device. The semantics never
    changed -- the name had been taken from the first instance rather than from
    the concept, and this module is test-only, so correcting it costs nothing a
    consumer can see.
    """

    name: str
    module: str
    optional_on: str | None = None


#: Adding a backend is adding a row. See the module docstring.
#:
#: ``python`` carries no ``optional_on``, so a failed import is escalated rather
#: than skipped -- nothing external is required to run pure Python/numpy, so the
#: only way ``pfsmgraph.hmm._viterbi`` fails to import is a broken working copy,
#: which is precisely what a skip would conceal. The module named is the *kernel*
#: rather than the package, because the row is a claim about one lifecycle phase
#: of one algorithm: ``pfsmgraph.hmm`` imports fine with no decode in it.
#:
#: ``cython`` carries no ``optional_on`` either, and here that clause finally bites
#: rather than merely being stated. A compiled backend has a way to be *present
#: in the source tree and absent from the environment* that a pure-Python one
#: does not: the ``.pyx`` is committed, so the phase is unambiguously
#: implemented, while the extension module exists only if it was built. ADR 0003
#: calls that a hard failure and never a skip -- "a backend that is implemented
#: but not importable (missing or stale Cython build)" is its own wording -- so
#: an unbuilt extension errors the session at startup instead of quietly
#: reporting a green run over one backend. Note what this rules out: ``optional_on`` is
#: for absences that are *legitimate* in the environment, like no CUDA device,
#: and a missing compiler is not one of those. It is a broken working copy.
#:
#: ``cpu_parallel`` is the first row whose absence is **legitimate**, and it is
#: what renamed this field. numba reaches an environment through
#: ``pfsmgraph-hmm``'s ``cpu-parallel`` extra, so an install that declined the
#: extra is entitled to lack the backend exactly as a machine without a GPU is
#: entitled to lack a CUDA device -- a reported skip, named in the header, never
#: an escalation. That is a *third* category the first two rows could not
#: exhibit: python and cython can only fail because the working copy is broken.
#:
#: Note this deviates from the ``dp-compile`` phase-3 skill, deliberately. That
#: skill prescribes numba as a hard dependency with ``hardware=None``, reasoning
#: that "a registered backend that will not import is a hard failure rather than
#: a skip, so making the import optional would turn every install without the
#: extra into a broken one". The reasoning is sound for a registry with two
#: states and false for this one: ``optional_on`` gives it three, so the broken
#: install the rule guards against cannot occur here. ADR 0004 governs the other
#: half -- acceleration is opt-in, and the decode is correct on the pure-Python
#: backend.
BACKENDS: Final[tuple[Backend, ...]] = (
    Backend("python", "pfsmgraph.hmm._viterbi"),
    Backend("cython", "pfsmgraph.hmm._viterbi_cython"),
    Backend("cpu_parallel", "pfsmgraph.hmm._viterbi_cpu_parallel", optional_on="numba"),
)


@dataclass(frozen=True)
class Availability:
    name: str
    available: bool
    reason: str | None = None


def detect(backends: Sequence[Backend] = BACKENDS) -> tuple[Availability, ...]:
    """Resolve every registered backend, escalating the ones ADR 0003 escalates."""
    states: list[Availability] = []
    for backend in backends:
        try:
            importlib.import_module(backend.module)
        except ImportError as exc:
            if backend.optional_on is None:
                raise BackendError(
                    f"backend {backend.name!r} is implemented but {backend.module!r} "
                    f"did not import: {exc}. ADR 0003 makes this a hard failure and "
                    f"never a skip -- a missing or stale build means the working copy "
                    f"is broken, which is precisely what a skip would conceal."
                ) from exc
            states.append(Availability(backend.name, False, f"no {backend.optional_on} detected"))
        else:
            states.append(Availability(backend.name, True))
    return tuple(states)


def format_header(states: Iterable[Availability]) -> str:
    """The one line ADR 0003 specifies, printed once at session start."""
    states = tuple(states)
    if not states:
        return EMPTY_HEADER
    return "backends: " + " · ".join(
        f"{s.name} ✓" if s.available else f"{s.name} ✗ ({s.reason})" for s in states
    )


def check_required(
    states: Iterable[Availability], env: Mapping[str, str] | None = None
) -> None:
    """Escalate skips to failures for the backends ``PFSMGRAPH_REQUIRE_BACKENDS`` names.

    Without this, a CI runner that loses its GPU degrades to a green run whose
    header nobody reads.
    """
    env = os.environ if env is None else env
    required = [n.strip() for n in env.get(REQUIRE_ENV, "").split(",") if n.strip()]
    if not required:
        return

    known = {s.name: s for s in states}
    # Validate the names before checking availability, so a typo -- or a CI that
    # believes in a backend this working copy has never had -- fails loudly
    # instead of passing because nothing by that name was missing.
    unknown = [n for n in required if n not in known]
    if unknown:
        raise BackendError(
            f"{REQUIRE_ENV} names {unknown}, which are not in the matrix "
            f"(registered: {sorted(known) or 'none'})."
        )
    missing = [n for n in required if not known[n].available]
    if missing:
        raise BackendError(
            f"{REQUIRE_ENV} requires {missing}, which this run would have skipped: "
            + "; ".join(f"{n}: {known[n].reason}" for n in missing)
        )
