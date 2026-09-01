"""The container: a sequence of :class:`SequenceRecord`, and the vocabulary they share."""

from __future__ import annotations

from typing import Iterable, Sequence

from ._record import SequenceRecord
from ._vocabulary import Vocabulary


class SequenceDataset:
    """A read-only collection of encoded sequences over one vocabulary.

    Compatible with ``torch.utils.data.Dataset`` **without importing torch**:
    that base class is duck-typed, so ``__len__`` and ``__getitem__`` are the
    whole contract. Keeping torch out is what lets the base layer of the family
    stay usable by the packages that have nothing to do with deep learning.

    Items are ragged, so a stock ``DataLoader`` needs a ``collate_fn`` at batch
    sizes above one -- pass :func:`pfsmgraph.dataseq.pad_collate`. Padding is
    therefore a property of a batch and never of the container, which is what
    makes an unmasked padded tensor hard to produce by accident::

        loader = DataLoader(ds, batch_size=32, collate_fn=pad_collate)

    **This is not a corpus loader.** There is no directory walking, no file
    format, and no pandas. Two of the imported implementations made the loader
    a constructor and neither could reuse the other's, because ingestion is the
    part that differs most between projects. :meth:`from_symbols` is the seam:
    the caller produces sequences of symbols however it likes, and hands them
    over.
    """

    __slots__ = ("_records", "_vocabulary")

    def __init__(
        self,
        records: Iterable[SequenceRecord],
        vocabulary: Vocabulary,
    ) -> None:
        self._records: tuple[SequenceRecord, ...] = tuple(records)
        for i, record in enumerate(self._records):
            if not isinstance(record, SequenceRecord):
                raise TypeError(
                    f"records[{i}] is {type(record).__name__}, expected SequenceRecord"
                )
        self._vocabulary = vocabulary

    @classmethod
    def from_symbols(
        cls,
        sequences: Iterable[Sequence[str]],
        vocabulary: Vocabulary,
        labels: Sequence[str] | None = None,
    ) -> SequenceDataset:
        """Encode sequences of symbols against ``vocabulary``.

        Encoding happens once, here, at the boundary -- everything downstream is
        integer-only. Unseen symbols raise, because the vocabulary is strict.
        """
        sequences = list(sequences)
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(sequences):
                raise ValueError(
                    f"labels and sequences must have the same length; "
                    f"got {len(labels)} labels for {len(sequences)} sequences"
                )
        records = [
            SequenceRecord(
                codes=vocabulary.encode(sequence),
                label=None if labels is None else labels[i],
            )
            for i, sequence in enumerate(sequences)
        ]
        return cls(records, vocabulary)

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    @property
    def lengths(self) -> tuple[int, ...]:
        """True lengths, in item order."""
        return tuple(record.length for record in self._records)

    def decode(self, index: int) -> list[str]:
        """Decode one item back to symbols."""
        return self._vocabulary.decode(self._records[index].codes)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> SequenceRecord:
        return self._records[index]

    def __iter__(self):
        return iter(self._records)

    def __repr__(self) -> str:
        return (
            f"SequenceDataset(n_sequences={len(self._records)}, "
            f"vocabulary_size={self._vocabulary.size})"
        )
