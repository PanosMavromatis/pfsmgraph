"""tokalign/_types.py — Core types for the tokalign package."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Alphabet:
    """An ordered set of symbols with bidirectional string ↔ integer mapping.

    The alphabet defines the universe of valid symbols for a given alignment
    problem. Symbols can be any hashable string — single characters, words,
    multi-character tokens, etc.

    The gap symbol is always included and occupies a reserved index.
    """

    symbols: tuple[str, ...]
    gap_symbol: str = "."

    # Indices 0–2 are reserved for interop with external libraries:
    #   0 = padding, 1 = beginning-of-sequence, 2 = end-of-sequence
    RESERVED_INDICES: int = 3

    # Derived mappings (built in __post_init__)
    _sym_to_idx: dict[str, int] = field(init=False, repr=False)
    _idx_to_sym: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        if self.gap_symbol in self.symbols:
            raise ValueError(
                f"Gap symbol '{self.gap_symbol}' must not appear in the symbol list. "
                f"It is added automatically."
            )

        # Indices 0–2 reserved; gap gets index 3; user symbols start at 4
        offset = self.RESERVED_INDICES
        sym_to_idx = {self.gap_symbol: offset}
        idx_to_sym = {offset: self.gap_symbol}
        for i, s in enumerate(self.symbols):
            idx = offset + 1 + i
            sym_to_idx[s] = idx
            idx_to_sym[idx] = s
        object.__setattr__(self, "_sym_to_idx", sym_to_idx)
        object.__setattr__(self, "_idx_to_sym", idx_to_sym)

    @property
    def size(self) -> int:
        """Total number of indices including reserved slots and the gap."""
        return self.RESERVED_INDICES + 1 + len(self.symbols)

    @property
    def gap_index(self) -> int:
        return self.RESERVED_INDICES

    def encode(self, sequence: Sequence[str]) -> np.ndarray:
        """Convert a sequence of string symbols to an integer array.

        Raises KeyError if any symbol is not in the alphabet.
        """
        try:
            return np.array([self._sym_to_idx[s] for s in sequence], dtype=np.int32)
        except KeyError as e:
            raise KeyError(
                f"Symbol {e} is not in this alphabet. "
                f"Valid symbols: {self.symbols}"
            ) from e

    def decode(self, indices: np.ndarray | list[int]) -> list[str]:
        """Convert an integer array back to string symbols."""
        return [self._idx_to_sym[int(i)] for i in indices]

    def encode_pair(
        self, seq_a: Sequence[str], seq_b: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode two sequences, validating both against this alphabet."""
        return self.encode(seq_a), self.encode(seq_b)


@dataclass
class ScoringMatrix:
    """A scoring matrix over an Alphabet.

    Internally stored as a 2D numpy array indexed by symbol integers.
    Constructed from human-readable (string, string) → score mappings
    but accessed by integer index in the hot loop.
    """

    alphabet: Alphabet
    _matrix: np.ndarray  # shape: (alphabet.size, alphabet.size)
    gap_open: float = -10.0
    gap_extend: float = -0.5

    @classmethod
    def from_dict(
        cls,
        alphabet: Alphabet,
        scores: dict[tuple[str, str], float],
        default: float = 0.0,
        gap_open: float = -10.0,
        gap_extend: float = -0.5,
    ) -> ScoringMatrix:
        """Build a scoring matrix from a dict of (symbol, symbol) → score.

        Any pair not in the dict gets the default score.
        The dict does not need to include gap pairs — those are handled
        by gap_open and gap_extend.
        """
        n = alphabet.size
        matrix = np.full((n, n), default, dtype=np.float64)

        for (sym_a, sym_b), score in scores.items():
            i = alphabet._sym_to_idx[sym_a]
            j = alphabet._sym_to_idx[sym_b]
            matrix[i, j] = score
            matrix[j, i] = score  # Symmetric by default

        return cls(alphabet=alphabet, _matrix=matrix, gap_open=gap_open, gap_extend=gap_extend)

    @classmethod
    def identity(
        cls,
        alphabet: Alphabet,
        match: float = 1.0,
        mismatch: float = -1.0,
        gap_open: float = -2.0,
        gap_extend: float = -0.5,
    ) -> ScoringMatrix:
        """Simple identity matrix: `match` on diagonal, `mismatch` elsewhere."""
        n = alphabet.size
        matrix = np.full((n, n), mismatch, dtype=np.float64)
        np.fill_diagonal(matrix, match)
        # Gap row/column gets zeros — actual gap penalties are handled separately
        # Zero out reserved indices and gap row/column
        for idx in range(alphabet.RESERVED_INDICES + 1):
            matrix[idx, :] = 0.0
            matrix[:, idx] = 0.0
        return cls(alphabet=alphabet, _matrix=matrix, gap_open=gap_open, gap_extend=gap_extend)

    def score(self, i: int, j: int) -> float:
        """Look up score by integer indices. This is the hot-path method."""
        return self._matrix[i, j]

    def score_symbols(self, sym_a: str, sym_b: str) -> float:
        """Look up score by symbol strings. Convenience method, not for hot paths."""
        return self._matrix[
            self.alphabet._sym_to_idx[sym_a],
            self.alphabet._sym_to_idx[sym_b],
        ]


@dataclass
class AlignmentResult:
    """The result of aligning two sequences."""

    score: float
    aligned_a: list[str]        # Aligned sequence A (with gap symbols inserted)
    aligned_b: list[str]        # Aligned sequence B (with gap symbols inserted)
    alphabet: Alphabet
    traceback: np.ndarray | None = None  # Optional: the full traceback matrix

    @property
    def identity(self) -> float:
        """Fraction of positions where aligned symbols match."""
        matches = sum(
            1 for a, b in zip(self.aligned_a, self.aligned_b)
            if a == b and a != self.alphabet.gap_symbol
        )
        length = len(self.aligned_a)
        return matches / length if length > 0 else 0.0

    def format(self, delimiter: str = " ") -> str:
        """Human-readable alignment display.

        Uses delimiter between symbols (important for multi-character symbols).
        """
        max_len = max(
            max((len(s) for s in self.aligned_a), default=0),
            max((len(s) for s in self.aligned_b), default=0),
        )
        row_a = delimiter.join(s.rjust(max_len) for s in self.aligned_a)
        match_line = delimiter.join(
            "|".rjust(max_len) if a == b and a != self.alphabet.gap_symbol
            else " ".rjust(max_len)
            for a, b in zip(self.aligned_a, self.aligned_b)
        )
        row_b = delimiter.join(s.rjust(max_len) for s in self.aligned_b)
        return f"{row_a}\n{match_line}\n{row_b}"
