"""Tests for tokalign._types — Alphabet, ScoringMatrix, AlignmentResult."""

import numpy as np
import pytest

from tokalign._types import Alphabet, AlignmentResult, ScoringMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dna_alpha():
    return Alphabet(symbols=("A", "C", "G", "T"))


@pytest.fixture
def multi_char_alpha():
    return Alphabet(symbols=("REST", "V7", "dim"))


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

class TestAlphabet:
    def test_default_gap_symbol(self, dna_alpha):
        assert dna_alpha.gap_symbol == "."

    def test_custom_gap_symbol(self):
        alpha = Alphabet(symbols=("X", "Y"), gap_symbol="*")
        assert alpha.gap_symbol == "*"

    def test_gap_in_symbols_raises(self):
        with pytest.raises(ValueError, match="must not appear"):
            Alphabet(symbols=("A", ".", "C"))

    def test_size_includes_reserved_gap_and_symbols(self, dna_alpha):
        # 3 reserved + 1 gap + 4 user symbols = 8
        assert dna_alpha.size == 8

    def test_gap_index_is_reserved_count(self, dna_alpha):
        assert dna_alpha.gap_index == 3

    def test_user_symbols_start_after_gap(self, dna_alpha):
        assert dna_alpha._sym_to_idx["A"] == 4
        assert dna_alpha._sym_to_idx["T"] == 7

    def test_encode_values(self, dna_alpha):
        result = dna_alpha.encode(["A", "G", "T"])
        np.testing.assert_array_equal(result, [4, 6, 7])
        assert result.dtype == np.int32

    def test_encode_unknown_symbol_raises(self, dna_alpha):
        with pytest.raises(KeyError, match="not in this alphabet"):
            dna_alpha.encode(["A", "Z"])

    def test_encode_empty_sequence(self, dna_alpha):
        result = dna_alpha.encode([])
        assert len(result) == 0
        assert result.dtype == np.int32

    def test_decode_roundtrip(self, dna_alpha):
        original = ["A", "C", "G", "T"]
        encoded = dna_alpha.encode(original)
        decoded = dna_alpha.decode(encoded)
        assert decoded == original

    def test_decode_gap_index(self, dna_alpha):
        decoded = dna_alpha.decode([dna_alpha.gap_index])
        assert decoded == ["."]

    def test_encode_pair(self, dna_alpha):
        enc_a, enc_b = dna_alpha.encode_pair(["A", "C"], ["G", "T"])
        np.testing.assert_array_equal(enc_a, [4, 5])
        np.testing.assert_array_equal(enc_b, [6, 7])

    def test_multi_character_symbols(self, multi_char_alpha):
        encoded = multi_char_alpha.encode(["REST", "V7", "dim"])
        decoded = multi_char_alpha.decode(encoded)
        assert decoded == ["REST", "V7", "dim"]


# ---------------------------------------------------------------------------
# ScoringMatrix
# ---------------------------------------------------------------------------

class TestScoringMatrix:
    def test_from_dict_symmetric(self, dna_alpha):
        scores = {("A", "C"): 3.0}
        sm = ScoringMatrix.from_dict(dna_alpha, scores)
        assert sm.score_symbols("A", "C") == 3.0
        assert sm.score_symbols("C", "A") == 3.0

    def test_from_dict_default_fill(self, dna_alpha):
        sm = ScoringMatrix.from_dict(dna_alpha, {}, default=-2.0)
        # Unspecified user-symbol pair should get the default
        assert sm.score_symbols("A", "T") == -2.0

    def test_identity_diagonal(self, dna_alpha):
        sm = ScoringMatrix.identity(dna_alpha, match=5.0, mismatch=-3.0)
        for sym in dna_alpha.symbols:
            assert sm.score_symbols(sym, sym) == 5.0

    def test_identity_off_diagonal(self, dna_alpha):
        sm = ScoringMatrix.identity(dna_alpha, match=5.0, mismatch=-3.0)
        assert sm.score_symbols("A", "T") == -3.0

    def test_identity_reserved_and_gap_zeroed(self, dna_alpha):
        sm = ScoringMatrix.identity(dna_alpha, match=5.0, mismatch=-3.0)
        # Indices 0–3 (reserved + gap) should all be zero
        for idx in range(dna_alpha.RESERVED_INDICES + 1):
            for j in range(dna_alpha.size):
                assert sm.score(idx, j) == 0.0
                assert sm.score(j, idx) == 0.0

    def test_score_matches_score_symbols(self, dna_alpha):
        sm = ScoringMatrix.identity(dna_alpha, match=2.0, mismatch=-1.0)
        i = dna_alpha._sym_to_idx["A"]
        j = dna_alpha._sym_to_idx["G"]
        assert sm.score(i, j) == sm.score_symbols("A", "G")

    def test_gap_penalty_attributes(self, dna_alpha):
        sm = ScoringMatrix.identity(dna_alpha, gap_open=-8.0, gap_extend=-1.5)
        assert sm.gap_open == -8.0
        assert sm.gap_extend == -1.5


# ---------------------------------------------------------------------------
# AlignmentResult
# ---------------------------------------------------------------------------

class TestAlignmentResult:
    def test_identity_fraction(self, dna_alpha):
        result = AlignmentResult(
            score=10.0,
            aligned_a=["A", "C", "G", "T"],
            aligned_b=["A", "C", "T", "T"],
            alphabet=dna_alpha,
        )
        # 3 matches out of 4 positions
        assert result.identity == pytest.approx(0.75)

    def test_identity_excludes_gaps(self, dna_alpha):
        result = AlignmentResult(
            score=5.0,
            aligned_a=["A", ".", "G"],
            aligned_b=["A", ".", "T"],
            alphabet=dna_alpha,
        )
        # gap-gap match should not count; A matches, G/T mismatch → 1/3
        assert result.identity == pytest.approx(1 / 3)

    def test_identity_empty_alignment(self, dna_alpha):
        result = AlignmentResult(
            score=0.0, aligned_a=[], aligned_b=[], alphabet=dna_alpha
        )
        assert result.identity == 0.0

    def test_format_three_lines(self, dna_alpha):
        result = AlignmentResult(
            score=5.0,
            aligned_a=["A", "C", "."],
            aligned_b=["A", "T", "G"],
            alphabet=dna_alpha,
        )
        lines = result.format().split("\n")
        assert len(lines) == 3

    def test_format_match_pipes(self, dna_alpha):
        result = AlignmentResult(
            score=5.0,
            aligned_a=["A", "C"],
            aligned_b=["A", "T"],
            alphabet=dna_alpha,
        )
        lines = result.format().split("\n")
        # First position matches → pipe, second doesn't → space
        assert "|" in lines[1]

    def test_format_multi_char_padding(self, multi_char_alpha):
        result = AlignmentResult(
            score=3.0,
            aligned_a=["REST", "V7"],
            aligned_b=["REST", "dim"],
            alphabet=multi_char_alpha,
        )
        lines = result.format().split("\n")
        # All lines should be the same length (padded)
        assert len(lines[0]) == len(lines[1]) == len(lines[2])
