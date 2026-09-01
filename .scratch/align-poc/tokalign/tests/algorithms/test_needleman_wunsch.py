"""Tests for the Needleman-Wunsch algorithm — all TC-XX cases from FORMALIZATION.md."""

import pytest

from tokalign._backends import get_available_backends
from tokalign._types import Alphabet, ScoringMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_backends = get_available_backends("needleman_wunsch")


@pytest.fixture(params=_backends, ids=[name for name, _ in _backends])
def align_fn(request):
    """Yield each available backend's align function."""
    _, fn = request.param
    return fn


def _make_alphabet(n_symbols: int) -> Alphabet:
    """Create an alphabet with n_symbols user symbols (ordinals 0 .. n-1)."""
    symbols = tuple(f"s{i}" for i in range(n_symbols))
    return symbols, Alphabet(symbols=symbols)


def _idx_to_sym(alphabet: Alphabet, idx: int) -> str:
    """Map a user-symbol ordinal to its string symbol.

    Ordinals are 0-based over `alphabet.symbols` and carry no knowledge of where
    user codes begin. They used to be "formalization indices" starting at 4, which
    collided with ADR 0011 putting GAP at 4 -- `_idx_to_sym(4)` then returned the
    gap symbol instead of the first user symbol. Rebased 2026-09-01; gaps are
    written as the "GAP" sentinel in expectations, never as an integer.
    """
    return alphabet.symbols[idx]


def _seq_to_strings(alphabet: Alphabet, indices: list[int]) -> list[str]:
    """Convert a list of formalization integer indices to string symbols."""
    return [_idx_to_sym(alphabet, i) for i in indices]


def _expected_with_gaps(alphabet: Alphabet, indices: list, gap_sentinel="GAP") -> list[str]:
    """Convert expected output indices (with GAP sentinels) to string symbols."""
    return [
        alphabet.gap_symbol if i == gap_sentinel else _idx_to_sym(alphabet, i)
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Shared alphabet and scoring for TC-01 through TC-08, TC-13, TC-16
# (identity matrix: match=2.0, mismatch=-1.0, g_o=-5.0, g_e=-1.0)
# ---------------------------------------------------------------------------

# Need symbols for indices 4–8 (s0–s4)
_STD_SYMBOLS = tuple(f"s{i}" for i in range(5))
_STD_ALPHABET = Alphabet(symbols=_STD_SYMBOLS)
_STD_SCORING = ScoringMatrix.identity(
    _STD_ALPHABET, match=2.0, mismatch=-1.0, gap_open=-5.0, gap_extend=-1.0
)


class TestStandardScoring:
    """TC-01 through TC-08 and TC-13, TC-16: identity(2.0, -1.0), g_o=-5.0, g_e=-1.0."""

    def test_tc01_identical_sequences(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0, 1, 2])
        seq_b = _seq_to_strings(_STD_ALPHABET, [0, 1, 2])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(6.0)
        assert result.aligned_a == _seq_to_strings(_STD_ALPHABET, [0, 1, 2])
        assert result.aligned_b == _seq_to_strings(_STD_ALPHABET, [0, 1, 2])

    def test_tc02_partially_overlapping(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0, 1, 2, 3])
        seq_b = _seq_to_strings(_STD_ALPHABET, [0, 1, 3, 4])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(2.0)
        assert len(result.aligned_a) == 4
        assert len(result.aligned_b) == 4
        # No gaps
        assert _STD_ALPHABET.gap_symbol not in result.aligned_a
        assert _STD_ALPHABET.gap_symbol not in result.aligned_b

    def test_tc03_completely_disjoint(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0, 1])
        seq_b = _seq_to_strings(_STD_ALPHABET, [2, 3])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(-2.0)
        assert _STD_ALPHABET.gap_symbol not in result.aligned_a
        assert _STD_ALPHABET.gap_symbol not in result.aligned_b

    def test_tc04_empty_sequence_a(self, align_fn):
        seq_a: list[str] = []
        seq_b = _seq_to_strings(_STD_ALPHABET, [0, 1, 2])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(-8.0)
        assert result.aligned_a == [_STD_ALPHABET.gap_symbol] * 3
        assert result.aligned_b == _seq_to_strings(_STD_ALPHABET, [0, 1, 2])

    def test_tc05_both_empty(self, align_fn):
        result = align_fn([], [], _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(0.0)
        assert result.aligned_a == []
        assert result.aligned_b == []

    def test_tc06_single_element_matching(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0])
        seq_b = _seq_to_strings(_STD_ALPHABET, [0])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(2.0)
        assert result.aligned_a == _seq_to_strings(_STD_ALPHABET, [0])
        assert result.aligned_b == _seq_to_strings(_STD_ALPHABET, [0])

    def test_tc07_single_element_nonmatching(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0])
        seq_b = _seq_to_strings(_STD_ALPHABET, [1])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(-1.0)
        assert _STD_ALPHABET.gap_symbol not in result.aligned_a
        assert _STD_ALPHABET.gap_symbol not in result.aligned_b

    def test_tc08_one_much_longer(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0])
        seq_b = _seq_to_strings(_STD_ALPHABET, [0, 1, 2, 3, 4])
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)

        assert result.score == pytest.approx(-7.0)
        # 4 gaps in aligned_a
        assert result.aligned_a.count(_STD_ALPHABET.gap_symbol) == 4
        # aligned_b has all 5 symbols (no gaps)
        assert _STD_ALPHABET.gap_symbol not in result.aligned_b

    def test_tc13_gap_penalties_dominate(self, align_fn):
        # Different scoring: match=1.0, mismatch=-1.0, same gap penalties
        scoring = ScoringMatrix.identity(
            _STD_ALPHABET, match=1.0, mismatch=-1.0, gap_open=-100.0, gap_extend=-100.0
        )
        seq_a = _seq_to_strings(_STD_ALPHABET, [0, 1, 2])
        seq_b = _seq_to_strings(_STD_ALPHABET, [3, 1, 4])

        # Need larger alphabet: indices 4-8 → symbols s0-s4
        result = align_fn(seq_a, seq_b, _STD_ALPHABET, scoring)

        assert result.score == pytest.approx(-1.0)
        assert _STD_ALPHABET.gap_symbol not in result.aligned_a
        assert _STD_ALPHABET.gap_symbol not in result.aligned_b

    def test_tc16_symmetry(self, align_fn):
        seq_a = _seq_to_strings(_STD_ALPHABET, [0, 1, 2])
        seq_b = _seq_to_strings(_STD_ALPHABET, [2, 1, 0])

        result_ab = align_fn(seq_a, seq_b, _STD_ALPHABET, _STD_SCORING)
        result_ba = align_fn(seq_b, seq_a, _STD_ALPHABET, _STD_SCORING)

        assert result_ab.score == pytest.approx(result_ba.score)


# ---------------------------------------------------------------------------
# TC-09 through TC-12: identity scoring with g_o=-3.0, g_e=-1.0
# ---------------------------------------------------------------------------

_MILD_GAP_SYMBOLS = tuple(f"s{i}" for i in range(5))
_MILD_GAP_ALPHABET = Alphabet(symbols=_MILD_GAP_SYMBOLS)
_MILD_GAP_SCORING = ScoringMatrix.identity(
    _MILD_GAP_ALPHABET, match=2.0, mismatch=-1.0, gap_open=-3.0, gap_extend=-1.0
)


class TestMildGapScoring:
    """TC-09 through TC-12: identity(2.0, -1.0), g_o=-3.0, g_e=-1.0."""

    def test_tc09_gap_at_start(self, align_fn):
        seq_a = _seq_to_strings(_MILD_GAP_ALPHABET, [2, 0, 1])
        seq_b = _seq_to_strings(_MILD_GAP_ALPHABET, [0, 1])
        result = align_fn(seq_a, seq_b, _MILD_GAP_ALPHABET, _MILD_GAP_SCORING)

        assert result.score == pytest.approx(0.0)
        assert result.aligned_a == _seq_to_strings(_MILD_GAP_ALPHABET, [2, 0, 1])
        assert result.aligned_b == _expected_with_gaps(
            _MILD_GAP_ALPHABET, ["GAP", 0, 1]
        )

    def test_tc10_gap_at_end(self, align_fn):
        seq_a = _seq_to_strings(_MILD_GAP_ALPHABET, [0, 1])
        seq_b = _seq_to_strings(_MILD_GAP_ALPHABET, [0, 1, 2])
        result = align_fn(seq_a, seq_b, _MILD_GAP_ALPHABET, _MILD_GAP_SCORING)

        assert result.score == pytest.approx(0.0)
        assert result.aligned_a == _expected_with_gaps(
            _MILD_GAP_ALPHABET, [0, 1, "GAP"]
        )
        assert result.aligned_b == _seq_to_strings(_MILD_GAP_ALPHABET, [0, 1, 2])

    def test_tc11_gap_in_middle(self, align_fn):
        seq_a = _seq_to_strings(_MILD_GAP_ALPHABET, [0, 1, 2])
        seq_b = _seq_to_strings(_MILD_GAP_ALPHABET, [0, 3, 1, 2])
        result = align_fn(seq_a, seq_b, _MILD_GAP_ALPHABET, _MILD_GAP_SCORING)

        assert result.score == pytest.approx(2.0)
        assert result.aligned_a == _expected_with_gaps(
            _MILD_GAP_ALPHABET, [0, "GAP", 1, 2]
        )
        assert result.aligned_b == _seq_to_strings(_MILD_GAP_ALPHABET, [0, 3, 1, 2])


class TestTC12MultipleGaps:
    """TC-12: identity(3.0, -1.0), g_o=-2.0, g_e=-0.5."""

    def test_tc12_multiple_gaps(self, align_fn):
        symbols = tuple(f"s{i}" for i in range(5))
        alphabet = Alphabet(symbols=symbols)
        scoring = ScoringMatrix.identity(
            alphabet, match=3.0, mismatch=-1.0, gap_open=-2.0, gap_extend=-0.5
        )
        seq_a = _seq_to_strings(alphabet, [0, 2, 1])
        seq_b = _seq_to_strings(alphabet, [0, 3, 1, 4])
        result = align_fn(seq_a, seq_b, alphabet, scoring)

        assert result.score == pytest.approx(2.5)
        assert len(result.aligned_a) >= 4
        # Count match positions
        matches = sum(
            1 for a, b in zip(result.aligned_a, result.aligned_b)
            if a == b and a != alphabet.gap_symbol
        )
        assert matches >= 2


class TestTC14HighMatchScores:
    """TC-14: identity(100.0, -1.0), g_o=-0.1, g_e=-0.1."""

    def test_tc14_very_high_match_scores(self, align_fn):
        symbols = tuple(f"s{i}" for i in range(3))
        alphabet = Alphabet(symbols=symbols)
        scoring = ScoringMatrix.identity(
            alphabet, match=100.0, mismatch=-1.0, gap_open=-0.1, gap_extend=-0.1
        )
        seq_a = _seq_to_strings(alphabet, [0, 1, 2])
        seq_b = _seq_to_strings(alphabet, [0, 1, 2])
        result = align_fn(seq_a, seq_b, alphabet, scoring)

        assert result.score == pytest.approx(300.0)
        assert result.aligned_a == _seq_to_strings(alphabet, [0, 1, 2])
        assert result.aligned_b == _seq_to_strings(alphabet, [0, 1, 2])


class TestTC15DegenerateScoring:
    """TC-15: all-zeros scoring matrix, g_o=-1.0, g_e=-0.5."""

    def test_tc15_degenerate_scoring(self, align_fn):
        symbols = tuple(f"s{i}" for i in range(4))
        alphabet = Alphabet(symbols=symbols)
        scoring = ScoringMatrix.identity(
            alphabet, match=0.0, mismatch=0.0, gap_open=-1.0, gap_extend=-0.5
        )
        seq_a = _seq_to_strings(alphabet, [0, 1])
        seq_b = _seq_to_strings(alphabet, [2, 3])
        result = align_fn(seq_a, seq_b, alphabet, scoring)

        assert result.score == pytest.approx(0.0)
        assert alphabet.gap_symbol not in result.aligned_a
        assert alphabet.gap_symbol not in result.aligned_b


class TestTC17AffineVsLinear:
    """TC-17: identity(5.0, -1.0), g_o=-4.0, g_e=-1.0."""

    def test_tc17_affine_gap_behavior(self, align_fn):
        symbols = tuple(f"s{i}" for i in range(5))
        alphabet = Alphabet(symbols=symbols)
        scoring = ScoringMatrix.identity(
            alphabet, match=5.0, mismatch=-1.0, gap_open=-4.0, gap_extend=-1.0
        )
        seq_a = _seq_to_strings(alphabet, [0, 1, 2, 3, 4])
        seq_b = _seq_to_strings(alphabet, [0, 4])
        result = align_fn(seq_a, seq_b, alphabet, scoring)

        assert result.score == pytest.approx(3.0)
        # One contiguous gap of length 3 in sequence B
        gap_count_b = result.aligned_b.count(alphabet.gap_symbol)
        assert gap_count_b == 3


class TestTC18PaperExample:
    """TC-18: adapted from Needleman & Wunsch 1970, Figure 2."""

    def test_tc18_paper_example(self, align_fn):
        # 11 distinct symbols needed (indices 4–14 → s0–s10)
        # A=4, B=5, C=6, N=7, J=8, R=9, Q=10, L=11, P=12, M=13, K=14
        symbols = tuple(f"s{i}" for i in range(11))
        alphabet = Alphabet(symbols=symbols)
        scoring = ScoringMatrix.identity(
            alphabet, match=1.0, mismatch=0.0, gap_open=0.0, gap_extend=0.0
        )

        # a = A B C N J R Q C L C R P M  → indices 4..13
        seq_a = _seq_to_strings(alphabet, [0, 1, 2, 3, 4, 5, 6, 2, 7, 2, 5, 8, 9])
        # b = A J C J N R C K C R B P    → K=14 (distinct from L=11)
        seq_b = _seq_to_strings(alphabet, [0, 4, 2, 4, 3, 5, 2, 10, 2, 5, 1, 8])

        result = align_fn(seq_a, seq_b, alphabet, scoring)

        assert result.score == pytest.approx(8.0)
        # Exactly 8 positions where aligned symbols match
        matches = sum(
            1 for a, b in zip(result.aligned_a, result.aligned_b)
            if a == b and a != alphabet.gap_symbol
        )
        assert matches == 8
