"""Reproduce ACCOUNT.md's appendix from the tracked corpora.

    python3 -m translation [basename ...]

Defaults to both specimens. Every number printed is measured by loading the
real ``.sds`` directories, so the account's claims are demonstrated rather than
asserted.
"""

from __future__ import annotations

import sys
from collections import Counter

from .dsource_seq import BEGIN, DsourceSeq
from .format_sds import build
from .lush_reader import read_raw_data

DEFAULTS = [
    "Training/set01z0/set01z0_100",
    "Training/set11a_dInt/set11a_dInt",
]


def report(basename: str) -> None:
    source = DsourceSeq.load(basename)
    cells = source.size * source.seq_size_max
    padding = source.padding_cells()
    empty = sum(1 for n in source.seq_sizes if n == 2)
    escaped = [s for s in source.alphabet if any(c.isspace() for c in s)]

    print(f"\n=== {basename.rsplit('/', 1)[-1]}.sds ===")
    print(f"  _size                {source.size}")
    print(f"  _alphabet_size       {source.alphabet_size}")
    print(f"  _seq_size_max        {source.seq_size_max}")
    print(f"  total entries        {sum(source.seq_sizes)}")
    print(f"  dense matrix cells   {cells}")
    print(f"  padding cells        {padding} ({100 * padding / cells:.0f}%) -- all BEGIN={BEGIN}")
    print(f"  empty sequences      {empty}")
    print(f"  whitespace symbols   {len(escaped)} {escaped if escaped else ''}")
    print(f"  length distribution  {dict(sorted(Counter(source.seq_sizes).items()))}")

    # The flat stream is what hmm consumes, and its only consumer.
    stream = source.fprop_all()
    assert stream.size == sum(source.seq_sizes)
    assert len(stream.path_states) == stream.size + 1
    print(f"  fprop_all stream     {stream.size} symbols, {len(stream.path_states)} states")

    # Rebuilding from _raw_data must reproduce the corpus exactly. This is the
    # real check: it exercises the tokenizer, the first-appearance ordering, the
    # begin/end wrapping and the dense packing against 16-year-old output.
    root = f"{basename}.sds"
    rebuilt = build(read_raw_data(f"{root}/_raw_data"), name=basename)
    for field in ("alphabet", "size", "seq_size_max", "seq_sizes", "seq_data"):
        got, want = getattr(rebuilt, field), getattr(source, field)
        status = "ok" if got == want else f"MISMATCH\n    got  {got}\n    want {want}"
        print(f"  rebuild {field:<13} {status}")


def main(argv: list[str]) -> int:
    for basename in argv[1:] or DEFAULTS:
        report(basename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
