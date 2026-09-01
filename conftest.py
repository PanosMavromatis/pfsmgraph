"""Root pytest configuration for the pfsmgraph workspace.

Carries the ADR 0003 session header and nothing else; the logic lives in
``_backends`` beside it, where it is importable and therefore testable.

**This file must stay at the rootdir.** ``pytest_report_header`` is a startup
hook, while conftest files under ``packages/*/tests/`` are loaded during
collection -- a hook defined there registers too late and is discarded with no
warning at all. Measured on pytest 9.1.1: of two conftests each defining the
hook, only the root one printed. Moving this file into a package would delete
the header silently, and "hook in the wrong directory" is indistinguishable in
the output from "no backends to report". ``tests/test_backends.py`` asserts the
placement for that reason.
"""

from __future__ import annotations

from typing import Final

import pytest

from _backends import Availability, BackendError, check_required, detect, format_header

pytest_plugins = ["pytester"]

_BACKENDS: Final = pytest.StashKey[tuple[Availability, ...]]()


def pytest_configure(config: pytest.Config) -> None:
    try:
        states = detect()
        check_required(states)
    except BackendError as exc:
        # UsageError renders as a clean one-line error; a bare raise here would
        # surface as an INTERNALERROR traceback.
        raise pytest.UsageError(str(exc)) from exc
    config.stash[_BACKENDS] = states


def pytest_report_header(config: pytest.Config) -> str:
    return format_header(config.stash[_BACKENDS])
