"""Control symbol encoding for sequence tokens.

Maps control symbols (PAD, BOS, EOS, etc.) to integer codes occupying the
lowest positions of the shared vocabulary space. Add new control symbols to
``_control_sym_encode``; the reverse lookup is derived automatically.
"""

# Symbol-to-code mapping for control tokens.
# Expand this dictionary as new control symbols are introduced.
# Codes must remain unique and contiguous starting from 0.
_control_sym_encode: dict[str, int] = {
    'PAD': 0,
    'BOS': 1,
    'EOS': 2,
}

# Code-to-symbol reverse lookup, derived from ``_control_sym_encode``.
_control_code_decode: dict[int, str] = {
    v: k for k, v in _control_sym_encode.items()
}
