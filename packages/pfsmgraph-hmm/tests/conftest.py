"""Fixtures shared by `pfsmgraph-hmm`'s suites.

**`backend` is ADR 0003's fixture parameter**, and ADR 0021 is what made it
possible: every public call that runs a DP kernel takes `backend=`, so a test
written against the public API can run once per backend without reaching into
one. Adding a backend is then adding a row to `pfsmgraph/hmm/_backends.py`, not
a test.

The parameters are **every backend the call has, available or not.** An
unavailable backend stays a parameter and skips inside the fixture with the
reason `backends()` reports, so a run without a CUDA device says so once per
test under `-ra` rather than quietly having fewer tests. A lifecycle phase not
yet implemented has no row, and therefore no parameter -- ADR 0003's third
category.

The session header and its escalation live in the repo-root `conftest.py`, and
must stay there: `pytest_report_header` is a startup hook, and a conftest under
`packages/*/tests/` is loaded too late for it. Fixtures have no such constraint.

It is keyed to `viterbi`. `baum_welch` has a different table, so a module
testing it overrides the fixture with one of the same name.
"""

from __future__ import annotations

import pytest

from pfsmgraph.hmm import backends
from pfsmgraph.hmm._backends import _TABLE, _status


@pytest.fixture(scope="module", params=[row.name for row in _TABLE["viterbi"]])
def backend(request):
    """One `viterbi` backend name, or a skip naming why it cannot run here."""
    status = {s.name: s for s in backends("viterbi")}[request.param]
    if not status.available:
        pytest.skip(f"backend {status.name!r} unavailable: {status.reason}")
    return request.param


@pytest.fixture(scope="module", params=[row.name for row in _TABLE["forward_backward"]])
def score_backend(request):
    """One `forward_backward` phase name for `score_backend=`, or a skip naming why it
    cannot run here. `torch` has no row in that table, so it is never a parameter.

    Read through the private `_status`, because `backends()` answers only for public
    calls and `forward_backward` is not one."""
    status = {s.name: s for s in _status("forward_backward")}[request.param]
    if not status.available:
        pytest.skip(f"phase {status.name!r} unavailable: {status.reason}")
    return request.param
