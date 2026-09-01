"""Executes every ``docs/api/`` code block and checks its pasted output.

[ADR 0013](../docs/design/adr/0013-api-documentation-layout-and-tooling.md) chose
hand-written Markdown over a generator, and named one guard against the drift that
choice invites: *every code block is executed and its output pasted from the run*.
Until this file existed that guard was a habit rather than a mechanism -- the docs
were in fact correct, verified 2026-09-01 at 44 of 44 examples, but nothing would
have said so if they had not been.

**Stock ``doctest`` cannot run these documents**, and both reasons are deliberate
choices that make the *documentation* better and the checking harder:

- Setup lives in plain ``python`` blocks with no ``>>>`` prompts, so a reader sees a
  clean script instead of prompt-cluttered lines. ``doctest`` sees only the ``>>>``
  block and has no namespace to evaluate it in -- every example fails with
  ``NameError``.
- Errors are pasted as the readable last line, ``ValueError: cannot collate an empty
  batch``, without ``doctest``'s required ``Traceback (most recent call last):``
  header.

So this runner follows the documents' own convention: plain blocks are setup and are
executed into one namespace per file, in document order; ``>>>`` examples are
evaluated against it; and an expected output shaped like ``SomeError: message`` is
matched against the exception actually raised rather than against stdout. A block
that will not compile at all is an API signature (``SequenceRecord(codes:
np.ndarray, ...)`` is not a valid call expression) and is skipped -- but only if it
looks like one, so a genuinely broken setup block fails instead of vanishing.
"""

from __future__ import annotations

import io
import pathlib
import re
from contextlib import redirect_stdout

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
API_DOCS = sorted((REPO_ROOT / "docs" / "api").glob("*/*.md"))

_FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
_ERROR_LINE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt|Warning)): ", re.S
)


def _normalize(text: str) -> str:
    """Collapse whitespace, so a line-wrapped repr still compares equal."""
    return " ".join(text.split())


def _examples(body: str):
    """Yield ``(source, expected)`` pairs from a ``>>>`` block."""
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">>> "):
            i += 1
            continue
        source = [lines[i][4:]]
        i += 1
        while i < len(lines) and lines[i].startswith("... "):
            source.append(lines[i][4:])
            i += 1
        expected: list[str] = []
        while i < len(lines) and not lines[i].startswith(">>> ") and lines[i].strip():
            expected.append(lines[i])
            i += 1
        yield "\n".join(source), "\n".join(expected).strip()


_STATEMENT = re.compile(
    r"^\s*(import|from|for|while|if|elif|else|with|try|except|finally|def|class"
    r"|return|raise|assert|del|global|nonlocal|pass|yield)\b"
)
_ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][\w.\[\]]*\s*=[^=]")


def _looks_like_a_signature(block: str) -> bool:
    """Call-shaped lines and nothing else, which is how these docs write signatures.

    Signatures wrap across lines, may end in ``-> ReturnType`` rather than ``)``,
    and a block may list two overloads, so shape alone is not enough. What
    separates them from a *broken* setup block is that they contain no statement
    and no assignment: ``vocab = SymbolTable([`` is a truncated setup block and
    must be reported, not skipped as prose.
    """
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines or "(" not in block:
        return False
    return not any(_STATEMENT.match(line) or _ASSIGNMENT.match(line) for line in lines)


def _run(path: pathlib.Path) -> tuple[list[str], int]:
    """Execute one document; return its mismatches and how many were checked."""
    source = path.read_text()
    namespace: dict[str, object] = {}
    problems: list[str] = []
    checked = 0

    for match in _FENCE.finditer(source):
        block = match.group(1)
        line = source[: match.start()].count("\n") + 1
        where = f"{path.name}:{line}"

        if ">>>" not in block:
            try:
                exec(compile(block, str(path), "exec"), namespace)
            except SyntaxError:
                if not _looks_like_a_signature(block):
                    problems.append(f"{where} setup block does not compile:\n{block}")
            except Exception as exc:  # noqa: BLE001 - reported, not handled
                problems.append(f"{where} setup raised {type(exc).__name__}: {exc}")
            continue

        for statement, expected in _examples(block):
            checked += 1
            wants_error = bool(_ERROR_LINE.match(expected))
            stdout = io.StringIO()
            try:
                with redirect_stdout(stdout):
                    try:
                        value = eval(compile(statement, str(path), "eval"), namespace)
                        rendered = "" if value is None else repr(value)
                    except SyntaxError:
                        exec(compile(statement, str(path), "exec"), namespace)
                        rendered = ""
            except Exception as exc:  # noqa: BLE001 - the documented outcome
                actual = f"{type(exc).__name__}: {exc}"
                if not wants_error:
                    problems.append(
                        f"{where} {statement!r}\n  documented: {expected}\n  raised:     {actual}"
                    )
                elif _normalize(actual) != _normalize(expected):
                    problems.append(
                        f"{where} {statement!r}\n  documented: {expected}\n  actual:     {actual}"
                    )
                continue

            got = (stdout.getvalue() + rendered).strip()
            if wants_error:
                problems.append(
                    f"{where} {statement!r}\n  documented an error: {expected}\n"
                    f"  but it returned:     {got or '<no output>'}"
                )
            elif _normalize(got) != _normalize(expected):
                problems.append(
                    f"{where} {statement!r}\n  documented: {expected}\n  actual:     {got}"
                )

    return problems, checked


def test_api_docs_are_discovered() -> None:
    """A glob that matches nothing would pass every test below in silence.

    This is the same failure shape ``test_backends.py`` pins for the conftest's
    placement: the check disappears without any output changing.
    """
    assert API_DOCS, f"no documents found under {REPO_ROOT / 'docs' / 'api'}"


@pytest.mark.parametrize(
    "path", API_DOCS, ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_documented_output_matches_the_run(path: pathlib.Path) -> None:
    problems, checked = _run(path)
    assert not problems, "\n".join(problems)
    if "```python" in path.read_text():
        assert checked, f"{path.name} has python blocks but no >>> example was checked"
