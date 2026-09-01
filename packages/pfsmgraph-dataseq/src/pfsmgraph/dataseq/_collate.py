"""Batching: where padding is introduced, and where its mask comes from."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._record import SequenceRecord
from ._reserved import PAD
from ._vocabulary import CODE_DTYPE


def pad_collate(batch: Sequence[SequenceRecord]) -> dict[str, np.ndarray]:
    """Pad a batch of ragged records to the batch maximum.

    Pass as ``collate_fn`` to a stock ``DataLoader``; no subclassing is needed
    on either side::

        DataLoader(dataset, batch_size=32, collate_fn=pad_collate)

    Returns ``codes`` ``(B, L)`` padded with ``PAD``, ``lengths`` ``(B,)``, and
    a boolean ``mask`` ``(B, L)`` that is ``True`` at real positions. The mask
    is returned unconditionally rather than offered as an option, because
    padding that cannot be distinguished from data is the failure the reserved
    block exists to prevent, and emitting the padding without the mask would
    reintroduce it one layer up.

    Arrays are numpy, not tensors: the base layer does not depend on torch.
    ``torch.from_numpy`` converts each in one call, without a copy.
    """
    if len(batch) == 0:
        raise ValueError("cannot collate an empty batch")

    lengths = np.array([record.length for record in batch], dtype=np.int64)
    width = int(lengths.max())

    codes = np.full((len(batch), width), PAD, dtype=CODE_DTYPE)
    mask = np.zeros((len(batch), width), dtype=bool)
    for i, record in enumerate(batch):
        n = record.length
        codes[i, :n] = record.codes
        mask[i, :n] = True

    return {"codes": codes, "lengths": lengths, "mask": mask}
