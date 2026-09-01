"""pad_collate: where padding is introduced, always with its mask."""

import numpy as np
import pytest

from pfsmgraph.dataseq import PAD, SequenceDataset, SymbolTable, pad_collate

SEQUENCES = [["D3", "F3", "G3"], ["F3", "E3"], ["D3"]]


@pytest.fixture
def batch():
    vocab = SymbolTable.from_sequences(SEQUENCES)
    return list(SequenceDataset.from_symbols(SEQUENCES, vocab))


def test_pads_to_the_batch_maximum(batch):
    out = pad_collate(batch)
    assert out["codes"].shape == (3, 3)


def test_padding_value_is_pad(batch):
    out = pad_collate(batch)
    assert out["codes"][2, 1] == PAD
    assert out["codes"][2, 2] == PAD


def test_lengths_are_the_true_lengths(batch):
    assert list(pad_collate(batch)["lengths"]) == [3, 2, 1]


def test_mask_marks_real_positions(batch):
    mask = pad_collate(batch)["mask"]
    assert mask.dtype == bool
    assert list(mask[1]) == [True, True, False]


def test_mask_and_lengths_agree(batch):
    out = pad_collate(batch)
    assert list(out["mask"].sum(axis=1)) == list(out["lengths"])


def test_real_codes_survive_padding(batch):
    out = pad_collate(batch)
    assert list(out["codes"][0]) == list(batch[0].codes)


def test_mask_is_not_optional(batch):
    # Emitting padding without the mask would reintroduce, one layer up, the
    # ambiguity the reserved block exists to prevent.
    assert "mask" in pad_collate(batch)


def test_empty_batch_raises():
    with pytest.raises(ValueError, match="empty batch"):
        pad_collate([])


def test_uniform_lengths_need_no_padding(batch):
    out = pad_collate(batch[:1])
    assert out["codes"].shape == (1, 3)
    assert out["mask"].all()


def test_all_empty_sequences():
    vocab = SymbolTable([])
    records = list(SequenceDataset.from_symbols([[], []], vocab))
    out = pad_collate(records)
    assert out["codes"].shape == (2, 0)
    assert list(out["lengths"]) == [0, 0]


def test_returns_numpy_not_tensors(batch):
    # The base layer has no torch dependency; torch.from_numpy is the caller's
    # one-line conversion, without a copy.
    out = pad_collate(batch)
    assert all(isinstance(v, np.ndarray) for v in out.values())
