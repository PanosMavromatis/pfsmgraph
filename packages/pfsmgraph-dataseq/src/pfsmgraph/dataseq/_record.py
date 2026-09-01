"""One sequence: its codes, its true length, and an optional label."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._vocabulary import CODE_DTYPE


@dataclass(frozen=True, eq=False)
class SequenceRecord:
    """A single encoded sequence.

    ``codes`` is a 1-D ``int32`` array, stored read-only. ``length`` is always
    the **true** length -- a record never holds padding, so the two cannot
    disagree. That is structural rather than conventional, and deliberately so:
    the Lush original packed ragged sequences into a dense matrix whose short
    rows were tail-filled with a code that meant a real symbol, and stayed
    correct only because every reader consulted a parallel lengths array.
    Padding here exists only in a collated batch, alongside the mask that
    identifies it.
    """

    codes: np.ndarray
    label: str | None = None

    def __post_init__(self) -> None:
        codes = np.asarray(self.codes, dtype=CODE_DTYPE)
        if codes.ndim != 1:
            raise ValueError(
                f"codes must be 1-D, got {codes.ndim}-D with shape {codes.shape}"
            )
        # Copy before freezing: asarray may have returned the caller's own array,
        # and making that read-only would mutate an object we do not own.
        codes = codes.copy()
        codes.setflags(write=False)
        object.__setattr__(self, "codes", codes)

    @property
    def length(self) -> int:
        """The true length. Never a padded width."""
        return int(self.codes.shape[0])

    def __len__(self) -> int:
        return self.length

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceRecord):
            return NotImplemented
        return self.label == other.label and np.array_equal(self.codes, other.codes)

    def __repr__(self) -> str:
        label = "" if self.label is None else f", label={self.label!r}"
        return f"SequenceRecord(length={self.length}{label})"
