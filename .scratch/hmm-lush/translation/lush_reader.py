"""Tokenizer for ``_raw_data``, the one place faithfulness is not optional.

``_raw_data`` is Lush source text consumed with ``(read)``. Everywhere else this
package renders decisions idiomatically, but here the tracked corpora are the
specification: if this reader disagrees with Lush, it cannot load them at all.

The grammar actually exercised by the two specimens:

- ``;`` begins a comment that runs to end of line.
- ``|...|`` is Lisp's *multiple-escape*: everything between the bars is one
  symbol name, whatever it contains -- whitespace included. ``set11a_dInt``
  relies on this for ``|E. -2nd|`` and ``|H _|``.
- ``()`` terminates a sequence. In Lush this is the empty list read as a falsy
  value, which is the branch ``format-sds`` uses to flush its accumulator.
- Any other run of non-whitespace is a bare symbol.
- Line structure carries no meaning. ``set11a_dInt`` puts one measure per line
  under a ``; ---- m. |1|`` header purely for reading by eye.

Note the ordering: comments are stripped **before** bars are interpreted, which
is what Lush does and what makes that ``|1|`` inside a comment harmless.
"""

from __future__ import annotations

from collections.abc import Iterator

__all__ = ["RawDataSyntaxError", "read_raw_data", "tokenize"]

#: Yielded in place of a symbol where ``()`` closes a sequence.
END_OF_SEQUENCE = object()


class RawDataSyntaxError(ValueError):
    """Raised on malformed ``_raw_data``.

    DEVIATION: the original has no error path at all -- an unterminated ``|``
    would run to end of file and be absorbed silently. Raising is the point of
    the exercise, since strictness at the boundary is what ADR 0011 asks for and
    what the original's build-time-only encoding never had to consider.
    """


def tokenize(text: str) -> Iterator[object]:
    """Yield symbol names and ``END_OF_SEQUENCE`` markers from ``_raw_data``."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ";":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif c == "|":
            j = text.find("|", i + 1)
            if j < 0:
                line = text.count("\n", 0, i) + 1
                raise RawDataSyntaxError(
                    f"unterminated multiple-escape '|' at line {line}"
                )
            yield text[i + 1 : j]
            i = j + 1
        elif c == "(":
            j = text.find(")", i + 1)
            if j < 0:
                line = text.count("\n", 0, i) + 1
                raise RawDataSyntaxError(f"unclosed '(' at line {line}")
            if text[i + 1 : j].strip():
                line = text.count("\n", 0, i) + 1
                raise RawDataSyntaxError(
                    f"only the empty list '()' is a sequence terminator, "
                    f"got '{text[i : j + 1]}' at line {line}"
                )
            yield END_OF_SEQUENCE
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in ";|()":
                j += 1
            yield text[i:j]
            i = j


def read_raw_data(path: str) -> list[list[str]]:
    """Parse ``_raw_data`` into one list of symbol names per sequence.

    The returned sequences hold **user symbols only**; ``begin`` and ``end`` are
    added at serialization, exactly as in the original.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    sequences: list[list[str]] = []
    current: list[str] = []
    for token in tokenize(text):
        if token is END_OF_SEQUENCE:
            sequences.append(current)
            current = []
        else:
            current.append(token)  # type: ignore[arg-type]

    if current:
        # DEVIATION: the original drops this silently -- its accumulator is only
        # flushed on the '()' branch, so a file ending without one loses its last
        # sequence. Both specimens terminate properly, so this never fired, but a
        # silent partial load is exactly the class of failure worth refusing.
        raise RawDataSyntaxError(
            f"file ends with {len(current)} symbol(s) after the last '()'; "
            "the original would discard them silently"
        )
    return sequences
