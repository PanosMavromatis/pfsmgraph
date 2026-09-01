"""``dsource-seq``: loads a built ``.sds`` corpus, holds it, flattens it.

The container never builds a vocabulary -- that happens once, at corpus-build
time, in ``format_sds``. A loaded source can decode but has no encoding path at
all. That seam is the original's most consequential structural decision and is
reproduced here deliberately; see ACCOUNT.md section 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .seq_state import SeqState

__all__ = ["BEGIN", "END", "FIRST_USER_CODE", "DsourceSeq"]

#: The original's entire reserved block: two codes, both sequence boundaries.
#: There is no padding, unknown, gap or mask code. ADR 0011 overrides this for
#: the merged package (PAD=0 .. MSK=5, user symbols from 6); it is recorded here
#: as what the original does, not as a candidate to adopt.
BEGIN = 0
END = 1
FIRST_USER_CODE = 2


@dataclass
class DsourceSeq:
    """A corpus of integer-coded sequences, loaded from a ``.sds`` directory."""

    name: str = ""
    alphabet: list[str] = field(default_factory=list)
    size: int = 0
    seq_size_max: int = 0
    seq_sizes: list[int] = field(default_factory=list)
    seq_data: list[list[int]] = field(default_factory=list)

    @property
    def alphabet_size(self) -> int:
        # DEVIATION: the original stores this in its own slot and its own file,
        # and trusts both. Deriving it removes a way for the corpus to disagree
        # with itself -- one of several the original leaves open (ACCOUNT.md 4).
        return len(self.alphabet)

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, basename: str) -> "DsourceSeq":
        """Read a ``<basename>.sds/`` directory."""
        root = f"{basename}.sds"
        alphabet_size = int(_read_one(os.path.join(root, "_alphabet_size")))
        alphabet = _read_alphabet(os.path.join(root, "_alphabet"), alphabet_size)
        size = int(_read_one(os.path.join(root, "_size")))
        seq_size_max = int(_read_one(os.path.join(root, "_seq_size_max")))
        seq_sizes = _read_indexed_ints(os.path.join(root, "_seq_sizes"), size)

        # The decision under evaluation: a dense size x seq_size_max matrix,
        # zero-filled, with only the first seq_sizes[i] cells of each row written.
        # Zero is BEGIN, a real symbol, so padding and data are indistinguishable
        # by inspection -- the representation is safe only because every reader
        # consults seq_sizes, which every reader in the original does.
        seq_data = [[BEGIN] * seq_size_max for _ in range(size)]
        for i in range(size):
            with open(os.path.join(root, f"{i}.seq"), encoding="utf-8") as handle:
                codes = [int(line) for line in handle if line.strip()]
            if len(codes) != seq_sizes[i]:
                # DEVIATION: the original reads exactly seq_sizes[i] values and
                # never looks at what follows, so a .seq longer or shorter than
                # its declared size loads silently. The intended check is an
                # unimplemented comment in load: "[ Will call the 'check
                # consistency' script here. ]".
                raise ValueError(
                    f"{i}.seq holds {len(codes)} codes, _seq_sizes declares "
                    f"{seq_sizes[i]}"
                )
            seq_data[i][: len(codes)] = codes

        return cls(
            name=basename,
            alphabet=alphabet,
            size=size,
            seq_size_max=seq_size_max,
            seq_sizes=seq_sizes,
            seq_data=seq_data,
        )

    # ---------------------------------------------------------------- saving

    def save(self, basename: str | None = None) -> None:
        """Write a ``.sds`` directory that ``load`` can read back."""
        root = f"{basename or self.name}.sds"
        os.makedirs(root, exist_ok=True)
        _write_one(os.path.join(root, "_alphabet_size"), self.alphabet_size)
        _write_alphabet(os.path.join(root, "_alphabet"), self.alphabet)
        _write_one(os.path.join(root, "_size"), self.size)
        _write_one(os.path.join(root, "_seq_size_max"), self.seq_size_max)
        with open(os.path.join(root, "_seq_sizes"), "w", encoding="utf-8") as handle:
            for i, seq_size in enumerate(self.seq_sizes):
                handle.write(f"{i}\t{seq_size}\n")
        for i in range(self.size):
            with open(os.path.join(root, f"{i}.seq"), "w", encoding="utf-8") as handle:
                for code in self.seq_data[i][: self.seq_sizes[i]]:
                    handle.write(f"{code}\n")

    # ------------------------------------------------------------- consuming

    def fprop_all(self) -> SeqState:
        """Concatenate every sequence into one flat stream.

        ``BEGIN``/``END`` stay inline and are the only thing marking where one
        sequence ends and the next begins. ``hmm-trainer.lsh:66`` calls this once
        at setup and is its only caller in the tree; nothing else reads
        ``seq_data``. So the dense matrix above is a staging buffer between the
        ``.seq`` files and this stream, not the corpus representation -- ragged
        data unpacked into a rectangle and immediately repacked ragged.
        """
        state = SeqState(name=self.name, size=sum(self.seq_sizes))
        pos = 0
        for row, seq_size in zip(self.seq_data, self.seq_sizes):
            for code in row[:seq_size]:  # the row's own size, never the width
                state.symbol_data[pos] = code
                pos += 1
        state.set_alphabet(self.alphabet)
        return state

    def decode(self, i: int) -> list[str]:
        """Symbol names of sequence ``i``, padding excluded."""
        return [self.alphabet[c] for c in self.seq_data[i][: self.seq_sizes[i]]]

    # ------------------------------------------------------------ statistics

    def padding_cells(self) -> int:
        """Cells of the dense matrix holding ``BEGIN`` as fill rather than data."""
        return self.size * self.seq_size_max - sum(self.seq_sizes)


# --------------------------------------------------------------------- files


def _read_one(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def _write_one(path: str, value: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{value}\n")


def _read_alphabet(path: str, alphabet_size: int) -> list[str]:
    """Read ``<code>\\t<symbol>`` lines, honouring ``|...|`` escaping."""
    alphabet: list[str] = [""] * alphabet_size
    seen = [False] * alphabet_size
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            code_text, _, name = line.rstrip("\n").partition("\t")
            code = int(code_text)
            if name.startswith("|") and name.endswith("|") and len(name) >= 2:
                name = name[1:-1]
            # DEVIATION: the original self-indexes here too -- '(alphabet n ...)'
            # stores at the index the file names, so a duplicate silently
            # overwrites and a gap silently leaves a null pointer. Refusing both
            # costs two lines.
            if not 0 <= code < alphabet_size:
                raise ValueError(f"alphabet code {code} outside 0..{alphabet_size - 1}")
            if seen[code]:
                raise ValueError(f"alphabet code {code} appears twice")
            alphabet[code], seen[code] = name, True
    if not all(seen):
        missing = [i for i, ok in enumerate(seen) if not ok]
        raise ValueError(f"alphabet is missing codes {missing}")
    return alphabet


def _write_alphabet(path: str, alphabet: list[str]) -> None:
    """Write the alphabet, restoring ``|...|`` where the name needs it.

    DEVIATION, and the substantive one. The original has two writers that
    disagree: ``format-sds`` prints the *symbol* with ``%l`` and so emits the
    escapes, while ``dsource-seq save`` prints ``(ptr-str ...)`` -- a bare string,
    its symbolhood discarded by ``symbol->string`` at load -- with ``%s``. The
    asymmetry is a downstream cost of the ``-gptr-`` alphabet slot, which exists
    so the compiled methods can carry it across the DH boundary (ACCOUNT.md 4).
    Here one writer owns the quoting rule and applies it by inspecting the name,
    which is what a vocabulary that persists strings rather than symbols has to do.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for code, name in enumerate(alphabet):
            escaped = f"|{name}|" if _needs_escape(name) else name
            handle.write(f"{code}\t{escaped}\n")


def _needs_escape(name: str) -> bool:
    return name == "" or any(c.isspace() or c in ";|()" for c in name)


def _read_indexed_ints(path: str, count: int) -> list[int]:
    values = [0] * count
    seen = [False] * count
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            index_text, _, value_text = line.rstrip("\n").partition("\t")
            index = int(index_text)
            if not 0 <= index < count:
                raise ValueError(f"index {index} outside 0..{count - 1} in {path}")
            if seen[index]:
                raise ValueError(f"index {index} appears twice in {path}")
            values[index], seen[index] = int(value_text), True
    if not all(seen):
        raise ValueError(f"{path} is missing indices")
    return values
