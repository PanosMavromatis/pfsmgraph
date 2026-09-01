"""The DataLoader criterion, verified by execution rather than asserted.

torch is not a dependency of this package -- it is imported here only to check
the interop claim, and these tests skip when it is absent. The skip is narrow
and named: it covers the integration, never the container's own behaviour,
which is tested without torch elsewhere.
"""

import pytest

from pfsmgraph.dataseq import SequenceDataset, SymbolTable, pad_collate

torch = pytest.importorskip(
    "torch", reason="torch not installed; DataLoader interop unverified in this environment"
)

SEQUENCES = [["D3", "F3", "G3"], ["F3", "E3"], ["D3"], ["G3", "G3", "E3", "D3"]]


@pytest.fixture
def dataset():
    vocab = SymbolTable.from_sequences(SEQUENCES)
    return SequenceDataset.from_symbols(SEQUENCES, vocab)


def test_stock_dataloader_works_without_subclassing(dataset):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["codes"].shape == (2, 3)
    assert batches[1]["codes"].shape == (2, 4)


def test_not_a_torch_dataset_by_isinstance(dataset):
    """Conformance here is behavioural, not nominal -- and that is the point.

    ``torch.utils.data.Dataset`` is a plain class rather than a protocol or an
    ABC, so ``isinstance`` is False without inheriting from it, which would mean
    importing torch in the base layer. ``DataLoader`` never performs that check
    for map-style datasets: it needs ``__len__`` and ``__getitem__``. Recorded as
    a test so the distinction is not later mistaken for a defect.
    """
    assert not isinstance(dataset, torch.utils.data.Dataset)


def test_default_collate_rejects_ragged_items(dataset):
    """Why pad_collate is shipped rather than left to the caller."""
    from torch.utils.data import DataLoader

    with pytest.raises(TypeError):
        next(iter(DataLoader(dataset, batch_size=2)))


def test_batch_converts_to_tensors_without_a_copy(dataset):
    from torch.utils.data import DataLoader

    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=pad_collate)))
    codes = torch.from_numpy(batch["codes"])
    assert codes.dtype == torch.int32
    assert codes.shape == (2, 3)
