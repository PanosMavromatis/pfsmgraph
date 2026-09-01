"""Backend availability for the ADR 0003 parameterized suites.

Test-only infrastructure. Nothing under ``packages/`` imports this module and no
built artifact contains it, which is deliberate: enumerating backends is most of
what a runtime backend-selection API needs, and ADR 0003 leaves that API's
behaviour explicitly open. Shipping the enumeration now would prejudge it.

:data:`BACKENDS` is empty, and that is the correct steady state rather than a
placeholder awaiting the next commit. ADR 0002 scopes the three-phase lifecycle
to "wherever dynamic programming appears"; no DP kernel exists yet, and
``dataseq`` contributes none at any maturity -- it is a container and an encoder.
The matrix fills when ``align`` or ``hmm`` lands a recurrence.
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
    :param hardware: what an absence may legitimately be blamed on, e.g.
        ``"CUDA device"``. ``None`` means nothing external is required, so a
        failed import is a broken working copy and is escalated.
    """

    name: str
    module: str
    hardware: str | None = None


#: Adding a backend is adding a row. Empty today; see the module docstring.
BACKENDS: Final[tuple[Backend, ...]] = ()


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
            if backend.hardware is None:
                raise BackendError(
                    f"backend {backend.name!r} is implemented but {backend.module!r} "
                    f"did not import: {exc}. ADR 0003 makes this a hard failure and "
                    f"never a skip -- a missing or stale build means the working copy "
                    f"is broken, which is precisely what a skip would conceal."
                ) from exc
            states.append(Availability(backend.name, False, f"no {backend.hardware} detected"))
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
