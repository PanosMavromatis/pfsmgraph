"""SymbolTable: strict encoding, total decoding, first-appearance order."""

import numpy as np
import pytest

from pfsmgraph.dataseq import CODE_DTYPE, SymbolTable, USER_BASE, Vocabulary


@pytest.fixture
def table():
    return SymbolTable(["D3", "F3", "G3"])


def test_satisfies_the_protocol(table):
    assert isinstance(table, Vocabulary)


def test_user_symbols_start_at_user_base(table):
    assert list(table.encode(["D3", "F3", "G3"])) == [6, 7, 8]


def test_size_includes_the_reserved_block(table):
    assert table.size == USER_BASE + 3


def test_first_appearance_order_not_sorted_or_frequency():
    # 'Z' appears first and once; 'A' appears later and twice. Neither sorting
    # nor frequency would put 'Z' first.
    table = SymbolTable.from_sequences([["Z", "A"], ["A"]])
    assert table.symbols == ("Z", "A")
    assert table.code("Z") == USER_BASE


def test_duplicates_collapse_without_reordering():
    assert SymbolTable(["A", "B", "A", "C"]).symbols == ("A", "B", "C")


def test_encode_dtype_is_int32(table):
    assert table.encode(["D3"]).dtype == CODE_DTYPE


def test_encode_empty_sequence(table):
    codes = table.encode([])
    assert codes.shape == (0,)
    assert codes.dtype == CODE_DTYPE


def test_encode_unknown_symbol_raises(table):
    # Inverted from the rudimentary implementation, which mapped unknown
    # symbols onto PAD deliberately and pinned it with a test. After that
    # collapse, "never seen" and "not present" are the same integer.
    with pytest.raises(KeyError, match="strict"):
        table.encode(["D3", "NOPE"])


def test_encode_error_names_the_position(table):
    with pytest.raises(KeyError, match="position 1"):
        table.encode(["D3", "NOPE"])


def test_decode_roundtrip(table):
    symbols = ["G3", "D3", "F3"]
    assert table.decode(table.encode(symbols)) == symbols


def test_decode_is_total_over_reserved_codes(table):
    # The proof-of-concept raised KeyError: 0 here, because its reverse map was
    # built from the gap index up -- so a zero-padded batch, the shape most
    # likely to be decoded, was the one that could not be.
    assert table.decode([0, 1, 2, 3, 4, 5]) == ["PAD", "UNK", "BOS", "EOS", "GAP", "MSK"]


def test_decode_out_of_range_raises(table):
    with pytest.raises(KeyError, match="out of range"):
        table.decode([table.size])


def test_reserved_name_cannot_be_a_user_symbol():
    with pytest.raises(ValueError, match="reserved"):
        SymbolTable(["D3", "PAD"])


def test_empty_vocabulary_is_just_the_reserved_block():
    table = SymbolTable([])
    assert table.size == USER_BASE
    assert table.symbols == ()


def test_multi_character_symbols():
    # The family's symbols are words, not characters -- this is why encoding at
    # the boundary matters rather than being cosmetic.
    table = SymbolTable(["Maior", "Finalis", "V7"])
    assert table.decode(table.encode(["V7", "Maior"])) == ["V7", "Maior"]


def test_no_method_adds_a_symbol(table):
    """A table cannot drift after sequences have been encoded against it."""
    with pytest.raises(AttributeError):
        table.add("NEW")
