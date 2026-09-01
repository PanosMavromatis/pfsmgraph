"""SequenceDataset: the container, and its Dataset compatibility."""

import numpy as np
import pytest

from pfsmgraph.dataseq import SequenceDataset, SequenceRecord, SymbolTable

SEQUENCES = [["D3", "F3", "G3"], ["F3", "E3"], ["D3"]]


@pytest.fixture
def vocab():
    return SymbolTable.from_sequences(SEQUENCES)


@pytest.fixture
def dataset(vocab):
    return SequenceDataset.from_symbols(SEQUENCES, vocab, labels=["V01", "V02", "V03"])


def test_len_is_the_number_of_sequences(dataset):
    assert len(dataset) == 3


def test_getitem_returns_a_record(dataset):
    assert isinstance(dataset[0], SequenceRecord)
    assert dataset[0].length == 3


def test_negative_indexing(dataset):
    assert dataset[-1].label == "V03"


def test_lengths_are_true_lengths(dataset):
    assert dataset.lengths == (3, 2, 1)


def test_labels_are_carried(dataset):
    assert [dataset[i].label for i in range(len(dataset))] == ["V01", "V02", "V03"]


def test_labels_optional(vocab):
    ds = SequenceDataset.from_symbols(SEQUENCES, vocab)
    assert ds[0].label is None


def test_label_length_mismatch_raises(vocab):
    with pytest.raises(ValueError, match="same length"):
        SequenceDataset.from_symbols(SEQUENCES, vocab, labels=["only-one"])


def test_decode_roundtrips(dataset):
    assert dataset.decode(0) == ["D3", "F3", "G3"]


def test_unknown_symbol_raises_at_construction(vocab):
    with pytest.raises(KeyError):
        SequenceDataset.from_symbols([["NOPE"]], vocab)


def test_empty_dataset(vocab):
    ds = SequenceDataset.from_symbols([], vocab)
    assert len(ds) == 0
    assert ds.lengths == ()


def test_empty_sequence_is_kept(vocab):
    ds = SequenceDataset.from_symbols([[]], vocab)
    assert ds[0].length == 0


def test_non_record_rejected(vocab):
    with pytest.raises(TypeError, match="SequenceRecord"):
        SequenceDataset([("not", "a", "record")], vocab)


def test_iterable(dataset):
    assert [record.length for record in dataset] == [3, 2, 1]


def test_vocabulary_is_exposed(dataset, vocab):
    assert dataset.vocabulary is vocab


def test_container_does_not_import_torch():
    """The base layer stays usable by packages that have nothing to do with DL.

    Checked in a fresh interpreter: ``sys.modules`` in this one says only that
    *something* in the session imported torch, which the interop tests do.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pfsmgraph.dataseq, sys; "
            "print('torch' in sys.modules or 'pandas' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_satisfies_torch_dataset_protocol_structurally(dataset):
    # torch.utils.data.Dataset is duck-typed: __len__ and __getitem__ are the
    # whole contract, which is why no torch import is needed to satisfy it.
    assert hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")
    assert dataset[len(dataset) - 1] is not None
