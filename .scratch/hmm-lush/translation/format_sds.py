"""``format-sds``: build a ``.sds`` corpus from ``_raw_data``.

In the original this is a free ``de``, not a method -- it shares no code with
``dsource-seq`` and communicates with it only through the on-disk format. That
separation is the reason the vocabulary is a build-time artefact rather than a
runtime object, and the reason a loaded container cannot encode anything. It is
reproduced here as a module-level function for the same reason: the seam is the
finding, so hiding it behind a method would misreport the original.
"""

from __future__ import annotations

import os

from .dsource_seq import BEGIN, END, FIRST_USER_CODE, DsourceSeq
from .lush_reader import read_raw_data

__all__ = ["build", "format_sds"]


def build(sequences: list[list[str]], name: str = "") -> DsourceSeq:
    """Assign codes and pack sequences, without touching the filesystem.

    Codes are assigned in **first-appearance order**, starting at
    ``FIRST_USER_CODE`` (2), after the reserved ``begin``/``end`` pair. The
    ordering is a property of the corpus text and carries no meaning:
    ``set11a_dInt``'s alphabet opens ``_``, ``+4th``, ``-2nd``, ``+2nd`` only
    because that is the order those intervals first occur.

    DEVIATION: the original scans the alphabet list linearly with ``member`` /
    ``member-pos`` and appends with ``add-to``, so building a corpus of N tokens
    over a vocabulary of V costs O(N*V) in scans plus O(V^2) in appends, and each
    sequence of length L costs O(L^2) to accumulate. A dict is used here instead.
    First-appearance order is preserved exactly -- Python dicts keep insertion
    order -- so the *observable* decision survives while the accidental
    complexity of a list language does not. There is no hash table anywhere in
    the original.
    """
    alphabet = ["begin", "end"]
    codes: dict[str, int] = {}

    encoded: list[list[int]] = []
    for symbols in sequences:
        row = [BEGIN]
        for symbol in symbols:
            code = codes.get(symbol)
            if code is None:
                code = len(alphabet)
                codes[symbol] = code
                alphabet.append(symbol)
            row.append(code)
        row.append(END)
        encoded.append(row)

    assert not codes or min(codes.values()) == FIRST_USER_CODE

    seq_sizes = [len(row) for row in encoded]
    # An empty sequence is representable and occurs: two consecutive '()' give a
    # row of size 2, BEGIN and END and nothing between. 35 of set01z0's 100
    # sequences are of exactly this shape.
    seq_size_max = max(seq_sizes, default=2)
    seq_data = [row + [BEGIN] * (seq_size_max - len(row)) for row in encoded]

    return DsourceSeq(
        name=name,
        alphabet=alphabet,
        size=len(encoded),
        seq_size_max=seq_size_max,
        seq_sizes=seq_sizes,
        seq_data=seq_data,
    )


def format_sds(basename: str) -> DsourceSeq:
    """Read ``<basename>.sds/_raw_data`` and write the rest of the directory.

    Matches the original's signature and its in-place behaviour: ``_raw_data``
    lives inside the ``.sds`` directory it is the input to, and the built corpus
    is written back around it.
    """
    root = f"{basename}.sds"
    sequences = read_raw_data(os.path.join(root, "_raw_data"))
    source = build(sequences, name=basename)
    source.save(basename)
    return source
