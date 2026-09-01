"""SequenceRecord: true length, read-only codes, no padding."""

import numpy as np
import pytest

from pfsmgraph.dataseq import CODE_DTYPE, SequenceRecord


def test_length_is_the_true_length():
    assert SequenceRecord(np.array([6, 7, 8])).length == 3


def test_len_agrees_with_length():
    record = SequenceRecord(np.array([6, 7]))
    assert len(record) == record.length


def test_codes_are_coerced_to_int32():
    assert SequenceRecord([6, 7, 8]).codes.dtype == CODE_DTYPE


def test_codes_are_read_only():
    record = SequenceRecord(np.array([6, 7]))
    with pytest.raises(ValueError):
        record.codes[0] = 99


def test_construction_does_not_freeze_the_callers_array():
    # asarray may hand back the caller's own buffer; freezing that would reach
    # into an object the record does not own.
    caller = np.array([6, 7], dtype=CODE_DTYPE)
    SequenceRecord(caller)
    caller[0] = 99  # must not raise


def test_record_is_frozen():
    record = SequenceRecord(np.array([6]))
    with pytest.raises(Exception):
        record.label = "changed"


def test_two_dimensional_codes_rejected():
    with pytest.raises(ValueError, match="1-D"):
        SequenceRecord(np.zeros((2, 3)))


def test_empty_sequence_is_valid():
    assert SequenceRecord(np.array([], dtype=CODE_DTYPE)).length == 0


def test_equality_compares_codes_and_label():
    assert SequenceRecord([6, 7], "a") == SequenceRecord([6, 7], "a")
    assert SequenceRecord([6, 7], "a") != SequenceRecord([6, 8], "a")
    assert SequenceRecord([6, 7], "a") != SequenceRecord([6, 7], "b")


def test_label_is_optional():
    assert SequenceRecord([6]).label is None
