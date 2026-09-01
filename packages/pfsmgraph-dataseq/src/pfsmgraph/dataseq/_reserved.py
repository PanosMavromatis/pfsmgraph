"""The fixed reserved symbol block.

Hard-coded per [ADR 0011]. These are module constants rather than a class
attribute or a constructor parameter, and that is deliberate: the block is
**not configurable**, and the way it is spelled here is what makes that true
structurally rather than by documentation.

The proof-of-concept made exactly that mistake -- ``RESERVED_INDICES: int = 3``
annotated as a dataclass field rather than a ``ClassVar``, which put the
"fixed" block into the generated ``__init__`` as a positional argument (see
``.scratch/align-poc/COMPARISON.md`` section 3.1). There is nothing here to
pass, override, or subclass.

``PAD`` must be 0. PyTorch's zero-fill idioms (``pad_sequence``,
``torch.zeros()`` buffers) write zeros into padded positions, so any other
value would make "absent" silently mean some real symbol. Three of the four
imported implementations put something else at 0 and each paid for it
differently; ``.scratch/RESERVED-BLOCK.md`` tabulates them.
"""

from typing import Final

PAD: Final[int] = 0
UNK: Final[int] = 1
BOS: Final[int] = 2
EOS: Final[int] = 3
GAP: Final[int] = 4
MSK: Final[int] = 5

#: The first code available to a user symbol. Everything below is reserved.
USER_BASE: Final[int] = 6

#: Reserved codes in code order; index into this with a code below USER_BASE.
RESERVED_SYMBOLS: Final[tuple[str, ...]] = ("PAD", "UNK", "BOS", "EOS", "GAP", "MSK")

#: Symbol-to-code for the reserved block, derived so the two cannot disagree.
RESERVED_CODES: Final[dict[str, int]] = {
    symbol: code for code, symbol in enumerate(RESERVED_SYMBOLS)
}
