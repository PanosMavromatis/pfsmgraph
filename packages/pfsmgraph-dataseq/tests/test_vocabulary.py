"""SymbolTable: strict encoding, total decoding, first-appearance order."""

import inspect

import numpy as np
import pytest

from pfsmgraph.dataseq import (
    CODE_DTYPE,
    RESERVED_SYMBOLS,
    SymbolTable,
    UNK,
    USER_BASE,
    Vocabulary,
)


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


# --- the strictness switch (ADR 0011) -----------------------------------


def test_encode_is_strict_by_default(table):
    with pytest.raises(KeyError, match="Encoding is strict"):
        table.encode(["D3", "ZZ"])


def test_the_strict_error_names_the_position_and_the_way_out(table):
    with pytest.raises(KeyError) as excinfo:
        table.encode(["D3", "F3", "ZZ"])
    # args[0], not str(): KeyError.__str__ is repr(args[0]), which re-escapes
    # the quotes the message itself contains.
    message = excinfo.value.args[0]
    assert "'ZZ'" in message
    assert "position 2" in message
    assert 'on_unknown="unk"' in message


def test_unk_opt_in_maps_unseen_symbols(table):
    assert list(table.encode(["D3", "ZZ"], on_unknown="unk")) == [6, UNK]


def test_unk_opt_in_still_decodes(table):
    """The fallback is a real code, not a hole -- decode stays total over it."""
    codes = table.encode(["ZZ", "D3"], on_unknown="unk")
    assert table.decode(codes) == ["UNK", "D3"]


def test_unk_is_never_the_default(table):
    """ADR 0011 fixes the direction of the switch: leniency is opt-in only."""
    signature = inspect.signature(table.encode)
    assert signature.parameters["on_unknown"].default == "raise"


def test_a_bad_policy_raises_even_on_empty_input(table):
    """Validated before the loop, not at the first unknown symbol.

    Lazy validation would let a misspelled policy behave as "raise" for as
    long as every symbol happened to be known, then change behaviour in
    production on the first unseen one.
    """
    with pytest.raises(ValueError, match="on_unknown must be"):
        table.encode([], on_unknown="UNK")


def test_a_bad_policy_raises_before_any_symbol_is_looked_up(table):
    with pytest.raises(ValueError, match="on_unknown must be"):
        table.encode(["D3", "F3"], on_unknown="ignore")


# --- the cross-distribution accessor ------------------------------------


def test_sym_to_code_publishes_the_mapping(table):
    assert dict(table.sym_to_code) == {"D3": 6, "F3": 7, "G3": 8}


def test_sym_to_code_is_read_only(table):
    """`align` is a consumer in another distribution; it must not be able to
    corrupt the table it is reading."""
    with pytest.raises(TypeError):
        table.sym_to_code["ZZ"] = 99


def test_sym_to_code_excludes_the_reserved_block(table):
    """It maps *user* symbols. Reserved names are not encodable, so publishing
    them here would offer a round trip that `encode` refuses to make."""
    assert not set(RESERVED_SYMBOLS) & set(table.sym_to_code)


def test_sym_to_code_agrees_with_code_and_encode(table):
    for symbol, expected in table.sym_to_code.items():
        assert table.code(symbol) == expected
        assert table.encode([symbol])[0] == expected
