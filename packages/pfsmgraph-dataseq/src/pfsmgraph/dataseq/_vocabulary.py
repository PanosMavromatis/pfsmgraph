"""The symbol-to-code mapping: a protocol, and the implementation of it.

The container programs against :class:`Vocabulary`, a structural protocol, so
that a different encoder can be substituted without touching the container.
:class:`SymbolTable` is the implementation this distribution ships.

The API here is settled, and three parts of it are contracts rather than
choices:

* **Encoding is strict by default.** ``encode`` raises on an unseen symbol;
  ``on_unknown="unk"`` is the explicit opt-in that maps it to ``UNK``. The
  semantics are fixed by ADR 0011 and are not renegotiable; only the spelling
  was open, and it is settled here.
* **Decoding is total** over ``range(size)``, reserved codes included --
  because the array most likely to be decoded is a padded batch, which is full
  of reserved codes by construction.
* **The symbol-to-code mapping is public.** ``code`` and ``sym_to_code`` are
  cross-distribution API: ``align`` builds an ``(size, size)`` scoring matrix
  from the whole mapping at construction time, and must not have to reach into
  a private attribute to do it.

The reserved block itself lives in :mod:`._reserved` as module constants, with
no constructor parameter anywhere that could relocate it.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ._reserved import RESERVED_SYMBOLS, UNK, USER_BASE

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

    @property
    def sym_to_code(self) -> Mapping[str, int]:
        """The user symbol to code mapping, read-only.

        Part of the protocol because it is consumed across a distribution
        boundary: ``pfsmgraph-align`` reads the whole mapping once to build a
        scoring matrix. An implementation must not return something a consumer
        can mutate.
        """
        ...

    def code(self, symbol: str) -> int:
        """The code for one symbol. Must raise if the symbol is unseen."""
        ...

    def encode(
        self,
        symbols: Sequence[str],
        on_unknown: Literal["raise", "unk"] = "raise",
    ) -> np.ndarray:
        """Map symbols to a 1-D ``int32`` array.

        Strict by default: an unseen symbol raises. ``on_unknown="unk"`` is the
        only alternative, and maps it to ``UNK``.
        """
        ...

    def decode(self, codes: Sequence[int]) -> list[str]:
        """Map codes back to symbols.

        Must be **total** over ``range(size)``, reserved codes included. A
        decode that handles only user codes cannot render a padded batch, which
        is the one array shape most likely to be decoded.
        """
        ...


class SymbolTable:
    """A frozen, first-appearance-ordered symbol table.

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

    @property
    def sym_to_code(self) -> Mapping[str, int]:
        """The user symbol to code mapping, as a read-only view.

        A view rather than a copy: ``align`` reads this once per scoring
        matrix, and a fresh ``dict`` per access is O(size) work that turns
        quadratic the moment a consumer calls it inside a loop. The proxy is
        live, so it cannot go stale -- and since a table has no method that
        adds a symbol, "live" and "immutable" are the same thing here.
        """
        return MappingProxyType(self._sym_to_code)

    def code(self, symbol: str) -> int:
        """The code for one symbol. Raises :class:`KeyError` if unseen."""
        try:
            return self._sym_to_code[symbol]
        except KeyError:
            raise KeyError(
                f"{symbol!r} is not in this vocabulary "
                f"({len(self._symbols)} user symbols). Encoding is strict."
            ) from None

    def encode(
        self,
        symbols: Sequence[str],
        on_unknown: Literal["raise", "unk"] = "raise",
    ) -> np.ndarray:
        """Encode strictly, unless the caller opts out.

        ``on_unknown="raise"`` (the default) raises :class:`KeyError` naming the
        offending symbol and its position. ``on_unknown="unk"`` maps it to
        ``UNK`` instead. There is no third policy, and no way to make leniency
        the default -- ADR 0011 fixes the direction of this switch.

        The value is validated *before* the loop rather than at the first
        unknown symbol. A misspelled policy is a bug in the caller either way,
        but validating lazily would let ``on_unknown="UNK"`` behave as
        ``"raise"`` for as long as every symbol happened to be known, and
        surface only on the first unseen one -- which, for an encoder whose
        whole job is handling unseen symbols, is precisely the wrong time.
        """
        if on_unknown not in ("raise", "unk"):
            raise ValueError(
                f"on_unknown must be 'raise' or 'unk', not {on_unknown!r}"
            )

        table = self._sym_to_code
        codes = np.empty(len(symbols), dtype=CODE_DTYPE)
        for i, symbol in enumerate(symbols):
            try:
                codes[i] = table[symbol]
            except KeyError:
                if on_unknown == "unk":
                    codes[i] = UNK
                    continue
                raise KeyError(
                    f"{symbol!r} at position {i} is not in this vocabulary "
                    f"({len(self._symbols)} user symbols). Encoding is strict; "
                    f'pass on_unknown="unk" to map unseen symbols to UNK.'
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
