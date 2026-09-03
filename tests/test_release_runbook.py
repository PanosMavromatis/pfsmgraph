"""Checks that every ``just`` recipe named in the release runbook exists.

``docs/ops/release.md`` is deliberately **not** covered by the ADR 0013 verifier in
``test_api_docs.py``, and the boundary is worth stating because the gap looks like an
oversight. ADR 0013 governs *API documentation*: its guard is that every code block is
executed and its output pasted from the run, which catches output drift in Python
examples. The runbook pastes no output at all -- its blocks are shell commands, and
executing them would publish to PyPI, prompt the macOS Keychain, and install from the
network. Widening the ADR to name a document its own mechanism cannot check would make
the ADR's central claim false, which is worse than the gap it would paper over.

The drift the runbook is actually exposed to is *referential*: a recipe gets renamed in
the ``justfile`` and the document goes on naming the old one. That is a much smaller
class than ADR 0013's -- following a stale name fails loudly, since ``just`` refuses an
unknown recipe -- but it fails at the one moment nobody wants to be debugging a runbook,
and asserting it costs the few lines below.

Prose *about* what a recipe does is not checkable by any test and is not attempted here;
the ``/agents-docs-update`` sweep is the only guard on that.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNBOOK = REPO_ROOT / "docs" / "ops" / "release.md"

# A recipe name is only read from a context that is unambiguously a command: an inline
# code span, or a line inside a fenced block. Scanning the prose instead would match an
# ordinary adverbial "just ..." and turn an English sentence into a failing assertion.
_SPAN = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^```[a-z]*\n(.*?)^```", re.S | re.M)
_INVOCATION = re.compile(r"^just ([a-z][a-z0-9-]*)")


def _recipes_named_in_runbook() -> set[str]:
    text = RUNBOOK.read_text()
    candidates = _SPAN.findall(text)
    candidates += [
        line for block in _FENCE.findall(text) for line in block.splitlines()
    ]
    return {
        match.group(1)
        for candidate in candidates
        if (match := _INVOCATION.match(candidate.strip()))
    }


def test_runbook_exists() -> None:
    assert RUNBOOK.is_file(), f"the release runbook is missing at {RUNBOOK}"


@pytest.mark.skipif(shutil.which("just") is None, reason="just is not installed")
def test_every_recipe_named_in_the_runbook_exists() -> None:
    named = _recipes_named_in_runbook()
    # The runbook is the entry point to the release path; if it names nothing, the
    # extraction above has silently stopped working and every assertion below is vacuous.
    assert named, "no just recipes found in the runbook -- extraction is broken"

    summary = subprocess.run(
        ["just", "--summary"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    defined = set(summary.stdout.split())
    missing = sorted(named - defined)
    assert not missing, (
        f"{RUNBOOK.name} names recipes the justfile does not define: {missing}. "
        f"Defined: {sorted(defined)}"
    )
