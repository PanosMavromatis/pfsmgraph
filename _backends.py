"""ADR 0003's backend policy for the parameterized suites: the header and escalation.

Test-only infrastructure, and **it holds no table.** Since 2026-09-14 the table of
backends lives in the distribution that owns the kernels -- ``pfsmgraph/hmm/_backends.py``,
shipped in the wheel, because ADR 0021's ``viterbi(..., backend=...)`` resolves through
it at runtime. What stays here is the part ADR 0003 scopes to this repository's own
checkout rather than to an install:

- **the session header**, one line naming every backend of every algorithm, with the
  reason for each absence;
- **the escalation rule**, which turns some absences into a startup failure;
- **``PFSMGRAPH_REQUIRE_BACKENDS``**, which lets CI escalate the rest.

Availability comes from each table module's private ``_status(algorithm)``, the probe
behind the public ``backends()``, so the header and a user's ``backends()`` call run one
probe per process and report the same reason. ``_status`` rather than ``backends``,
because the header also reports private algorithms -- ``forward_backward`` has a row and
no public call yet.

**The two policies differ on exactly one row kind, and correctly** (ADR 0021 section 3).
A row whose absence is attributable to *nothing* (pure Python) or to *the compiled
extension* escalates here, since in a checkout either means a broken working copy or a
missing or stale build. At runtime the second is an ordinary pure-wheel install and the
API reports it as unavailable. Absences attributable to numba or a CUDA device are
legitimate in both, so here they are reported skips.

Until 2026-09-14 this module held ``BACKENDS``, one ``Backend`` row per Viterbi phase with
an ``optional_on`` field (called ``hardware`` until 2026-09-10). That table is now the
package's ``_TABLE``, and ``optional_on`` its ``needs``.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

#: The table modules to read, one per distribution that has DP kernels. ``align``
#: adds its own when it gets one; no distribution imports another's table.
TABLES: Final[tuple[str, ...]] = ("pfsmgraph.hmm._backends",)

#: Printed when no backend is registered. ADR 0003 requires the header print even
#: then: a missing line is indistinguishable from a hook that is absent, sited in
#: the wrong directory, or that raised and was swallowed.
EMPTY_HEADER: Final = "backends: none registered — no DP kernel has reached ADR 0002 phase 1"

REQUIRE_ENV: Final = "PFSMGRAPH_REQUIRE_BACKENDS"

#: Absences that are a broken checkout rather than a modest environment: nothing
#: external is needed (``None``), or the extension of a committed ``.pyx`` was
#: never built. ADR 0003: "implemented but not importable" is a hard failure.
ESCALATED_NEEDS: Final = frozenset({None, "compiled extension"})


class BackendError(Exception):
    """A backend problem ADR 0003 requires be a hard failure, never a skip."""


@dataclass(frozen=True)
class Availability:
    """One backend of one algorithm, as probed in this session."""

    algorithm: str
    name: str
    available: bool
    reason: str | None = None


def detect(tables: Sequence[object] | None = None) -> tuple[Availability, ...]:
    """Probe every backend of every algorithm, escalating the ones ADR 0003 escalates.

    :param tables: table modules (anything with ``_TABLE`` and ``_status``); defaults
        to importing :data:`TABLES`. A table module that will not import is itself a
        broken checkout, and escalates.
    """
    if tables is None:
        tables = []
        for name in TABLES:
            try:
                tables.append(importlib.import_module(name))
            except ImportError as exc:
                raise BackendError(f"backend table {name!r} did not import: {exc}") from exc

    states: list[Availability] = []
    for table in tables:
        for algorithm, rows in table._TABLE.items():
            for row, status in zip(rows, table._status(algorithm)):
                if not status.available and row.needs in ESCALATED_NEEDS:
                    raise BackendError(
                        f"backend {row.name!r} of {algorithm} is implemented but "
                        f"{row.module!r} did not import: {status.reason}. ADR 0003 makes "
                        f"this a hard failure and never a skip -- in a checkout a missing "
                        f"or stale build means the working copy is broken, which is "
                        f"precisely what a skip would conceal."
                    )
                states.append(Availability(algorithm, row.name, status.available, status.reason))
    return tuple(states)


def format_header(states: Iterable[Availability]) -> str:
    """The one line ADR 0003 specifies, grouped by algorithm, printed once at startup."""
    groups: dict[str, list[str]] = {}
    for s in states:
        cell = f"{s.name} ✓" if s.available else f"{s.name} ✗ ({s.reason})"
        groups.setdefault(s.algorithm, []).append(cell)
    if not groups:
        return EMPTY_HEADER
    return "backends: " + " | ".join(
        f"{algorithm} " + " · ".join(cells) for algorithm, cells in groups.items()
    )


def check_required(
    states: Iterable[Availability], env: Mapping[str, str] | None = None
) -> None:
    """Escalate skips to failures for the backends ``PFSMGRAPH_REQUIRE_BACKENDS`` names.

    A name is required wherever an algorithm has that row, so ``cuda`` covers every
    algorithm with a CUDA phase. Without this, a CI runner that loses its GPU degrades
    to a green run whose header nobody reads.
    """
    env = os.environ if env is None else env
    required = [n.strip() for n in env.get(REQUIRE_ENV, "").split(",") if n.strip()]
    if not required:
        return

    states = tuple(states)
    known = {s.name for s in states}
    # Validate the names before checking availability, so a typo -- or a CI that
    # believes in a backend this working copy has never had -- fails loudly
    # instead of passing because nothing by that name was missing.
    unknown = [n for n in required if n not in known]
    if unknown:
        raise BackendError(
            f"{REQUIRE_ENV} names {unknown}, which are not in the matrix "
            f"(registered: {sorted(known) or 'none'})."
        )
    missing = [s for s in states if s.name in required and not s.available]
    if missing:
        raise BackendError(
            f"{REQUIRE_ENV} requires {sorted({s.name for s in missing})}, which this run "
            f"would have skipped: "
            + "; ".join(f"{s.algorithm} {s.name}: {s.reason}" for s in missing)
        )
