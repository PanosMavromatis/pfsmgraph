"""The reserved block is fixed, contiguous, and not configurable (ADR 0011)."""

import pfsmgraph.dataseq as ds
from pfsmgraph.dataseq import _reserved


def test_codes_are_the_adr_values():
    assert (ds.PAD, ds.UNK, ds.BOS, ds.EOS, ds.GAP, ds.MSK) == (0, 1, 2, 3, 4, 5)


def test_pad_is_zero():
    # Load-bearing: torch's zero-fill idioms write 0 into padded positions, so
    # any other value would make "absent" mean a real symbol.
    assert ds.PAD == 0


def test_user_base_is_six_and_leaves_no_hole():
    assert ds.USER_BASE == 6
    assert ds.USER_BASE == len(ds.RESERVED_SYMBOLS)


def test_symbols_and_codes_agree():
    assert ds.RESERVED_CODES == {s: i for i, s in enumerate(ds.RESERVED_SYMBOLS)}


def test_gap_exists():
    # None of the three imported containers had a gap code; align exists to
    # produce this symbol, so its absence would be felt one package away.
    assert "GAP" in ds.RESERVED_CODES


def test_block_is_not_configurable():
    """No constructor, class, or parameter can relocate the block.

    The proof-of-concept lost this by annotating RESERVED_INDICES as a dataclass
    field rather than a ClassVar, which made the block a positional argument.
    Here there is nothing to pass, so the test asserts the *absence* of a seam.
    """
    own = {
        name: value
        for name, value in vars(_reserved).items()
        if not name.startswith("_")
        and getattr(value, "__module__", _reserved.__name__) == _reserved.__name__
    }
    # Everything this module defines is a plain constant: no class to subclass,
    # no function to call, no parameter to pass.
    assert own, "expected the module to define its constants"
    assert all(isinstance(v, (int, tuple, dict)) for v in own.values())
    assert not any(callable(v) for v in own.values())
