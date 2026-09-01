"""Data sequence container and symbol-to-code encoder.

The base layer of the ``pfsmgraph`` family: everything else depends on this and
it depends on nothing in the family. It has no torch and no pandas dependency
-- the container is compatible with ``torch.utils.data.Dataset`` by being
duck-typed, and ingestion belongs to the caller.

Typical use::

    from pfsmgraph.dataseq import SequenceDataset, SymbolTable, pad_collate

    sequences = [["D3", "F3", "G3"], ["F3", "E3"]]
    vocab = SymbolTable.from_sequences(sequences)
    ds = SequenceDataset.from_symbols(sequences, vocab, labels=["V01", "V02"])

    # Stock DataLoader, no subclassing:
    #   DataLoader(ds, batch_size=32, collate_fn=pad_collate)

Encoding is strict: an unseen symbol raises rather than silently becoming
padding or a NaN. The reserved block is fixed and not configurable
(:mod:`pfsmgraph.dataseq._reserved`).

Note that :class:`SymbolTable` is the provisional encoder implementation; the
encoder API is settled separately and recorded in ADR 0010.
"""

from ._collate import pad_collate
from ._record import SequenceRecord
from ._reserved import (
    BOS,
    EOS,
    GAP,
    MSK,
    PAD,
    RESERVED_CODES,
    RESERVED_SYMBOLS,
    UNK,
    USER_BASE,
)
from ._dataset import SequenceDataset
from ._vocabulary import CODE_DTYPE, SymbolTable, Vocabulary

__all__ = [
    "BOS",
    "CODE_DTYPE",
    "EOS",
    "GAP",
    "MSK",
    "PAD",
    "RESERVED_CODES",
    "RESERVED_SYMBOLS",
    "SequenceDataset",
    "SequenceRecord",
    "SymbolTable",
    "UNK",
    "USER_BASE",
    "Vocabulary",
    "pad_collate",
]
