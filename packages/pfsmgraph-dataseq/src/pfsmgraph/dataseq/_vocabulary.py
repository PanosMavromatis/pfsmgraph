"""The symbol-to-code mapping: a protocol, and a minimal implementation of it.

The container programs against :class:`Vocabulary`, a structural protocol, so
that settling the concrete encoder API does not touch the container. That
separation is the whole point: the encoder API -- the real constructor
signature, the spelling of the strictness switch, and how ``align`` consumes
the mapping at its boundary -- is settled by a later goal and recorded in
[ADR 0010].

:class:`SymbolTable` is the minimal conforming implementation that lets the
container be exercised end-to-end and tested. It is **provisional**. It is
strict with no fallback at all, which means the later opt-in ``UNK`` fallback
is something added to it rather than a default reversed in it.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from ._reserved import RESERVED_SYMBOLS, USER_BASE

#: The dtype every code array uses. int32 is what the alignment proof-of-concept
#: settled on and what a Cython or CUDA buffer wants; codes never approach its
#: range, and fixing it here keeps the DP packages free of a conversion.
CODE_DTYPE = np.int32


@runtime_checkable
class Vocabulary(Protocol):
    """The mapping every ``dataseq`` consumer may rely on.

    Structural, not nominal: anything with these three members satisfies it.
    """

    @property
    def size(self) -> int:
        """Total number of codes, reserved block included."""
        ...

    def encode(self, symbols: Sequence[str]) -> np.ndarray:
        """Map symbols to a 1-D ``int32`` array. Unseen symbols must raise."""
        ...

    def decode(self, codes: Sequence[int]) -> list[str]:
        """Map codes back to symbols.

        Must be **total** over ``range(size)``, reserved codes included. A
        decode that handles only user codes cannot render a padded batch, which
        is the one array shape most likely to be decoded.
        """
        ...


class SymbolTable:
    """A frozen, first-appearance-ordered symbol table. Provisional -- see module docstring.

    Immutable by construction: the symbols are held as a tuple and the two
    lookup maps are built once. There is no method that adds a symbol, so a
    table cannot drift after the sequences encoded against it.

    Ordering is **first appearance**, which is a correctness constraint rather
    than a preference. Of the imported implementations one assigned codes by
    iterating a ``set`` -- a live reproducibility bug, since CPython randomises
    string hashing per process -- and another ordered by frequency, which is
    deterministic but makes every code a function of the whole corpus, so
    adding one file renumbers the alphabet.
    """

    __slots__ = ("_symbols", "_sym_to_code", "_code_to_sym")

    def __init__(self, symbols: Iterable[str]) -> None:
        ordered: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            if symbol in self._reserved_names():
                raise ValueError(
                    f"{symbol!r} is a reserved symbol name and cannot be a user symbol. "
                    f"Reserved: {', '.join(RESERVED_SYMBOLS)}"
                )
            seen.add(symbol)
            ordered.append(symbol)

        self._symbols: tuple[str, ...] = tuple(ordered)
        self._sym_to_code: dict[str, int] = {
            symbol: USER_BASE + i for i, symbol in enumerate(self._symbols)
        }
        # Total by construction: the reserved block is inserted here, so decode
        # cannot be partial over the low codes the way it is easy to leave it.
        self._code_to_sym: dict[int, str] = {
            code: symbol for code, symbol in enumerate(RESERVED_SYMBOLS)
        }
        self._code_to_sym.update(
            {code: symbol for symbol, code in self._sym_to_code.items()}
        )

    @staticmethod
    def _reserved_names() -> frozenset[str]:
        return frozenset(RESERVED_SYMBOLS)

    @classmethod
    def from_sequences(cls, sequences: Iterable[Sequence[str]]) -> SymbolTable:
        """Build a table from a corpus, in order of first appearance."""
        return cls(symbol for sequence in sequences for symbol in sequence)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The user symbols, in code order."""
        return self._symbols

    @property
    def size(self) -> int:
        return USER_BASE + len(self._symbols)

    def code(self, symbol: str) -> int:
        """The code for one symbol. Raises :class:`KeyError` if unseen."""
        try:
            return self._sym_to_code[symbol]
        except KeyError:
            raise KeyError(
                f"{symbol!r} is not in this vocabulary "
                f"({len(self._symbols)} user symbols). Encoding is strict."
            ) from None

    def encode(self, symbols: Sequence[str]) -> np.ndarray:
        """Encode strictly: any unseen symbol raises rather than falling back."""
        table = self._sym_to_code
        codes = np.empty(len(symbols), dtype=CODE_DTYPE)
        for i, symbol in enumerate(symbols):
            try:
                codes[i] = table[symbol]
            except KeyError:
                raise KeyError(
                    f"{symbol!r} at position {i} is not in this vocabulary "
                    f"({len(self._symbols)} user symbols). Encoding is strict."
                ) from None
        return codes

    def decode(self, codes: Sequence[int]) -> list[str]:
        """Decode totally, reserved codes included."""
        table = self._code_to_sym
        out: list[str] = []
        for i, code in enumerate(codes):
            try:
                out.append(table[int(code)])
            except KeyError:
                raise KeyError(
                    f"code {int(code)} at position {i} is out of range "
                    f"for this vocabulary (size {self.size})"
                ) from None
        return out

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"SymbolTable(size={self.size}, user_symbols={len(self._symbols)})"
