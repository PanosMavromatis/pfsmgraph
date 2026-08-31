"""
Unit tests for the Alignment class in segalign.seq.alignment module.

These tests validate the Alignment class functionality including initialization,
validation, analysis properties, and integration methods.
"""

import pytest
import pandas as pd
from segalign.seq import Alignment


class TestAlignmentInit:
    """Test basic Alignment initialization and validation."""
    
    def test_valid_pairwise_alignment(self):
        """Test Alignment initialization with valid pairwise sequences."""
        sequences = [
            ['D3', 'F3', None, 'G3'],
            ['D3', None, 'E3', 'G3']
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        
        assert alignment.sequences == sequences
        assert alignment.labels == labels
        assert alignment.score is None
        assert alignment.algorithm is None
        assert len(alignment) == 4
        assert alignment.alignment_length == 4
    
    def test_valid_multiple_alignment(self):
        """Test Alignment initialization with multiple sequences."""
        sequences = [
            ['D3', 'F3', None],
            ['D3', None, 'G3'],
            [None, 'F3', 'G3']
        ]
        labels = ['seq1', 'seq2', 'seq3']
        score = 2.5
        algorithm = "T-Coffee"
        
        alignment = Alignment(sequences, labels, score, algorithm)
        
        assert alignment.sequences == sequences
        assert alignment.labels == labels
        assert alignment.score == score
        assert alignment.algorithm == algorithm
        assert len(alignment) == 3
    
    def test_empty_sequences(self):
        """Test Alignment with empty sequences."""
        sequences = [[], []]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        
        assert len(alignment) == 0
        assert alignment.alignment_length == 0
    
    def test_insufficient_sequences_error(self):
        """Test that fewer than 2 sequences raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 sequences required"):
            Alignment([['D3', 'F3']], ['seq1'])
    
    def test_mismatched_lengths_error(self):
        """Test that mismatched sequence and label lengths raise ValueError."""
        sequences = [['D3', 'F3'], ['D3', 'G3']]
        labels = ['seq1']  # Only one label for two sequences
        
        with pytest.raises(ValueError, match="sequences and labels must have the same length"):
            Alignment(sequences, labels)
    
    def test_unequal_sequence_lengths_error(self):
        """Test that sequences of different lengths raise ValueError."""
        sequences = [
            ['D3', 'F3', 'G3'],  # Length 3
            ['D3', 'F3']         # Length 2
        ]
        labels = ['seq1', 'seq2']
        
        with pytest.raises(ValueError, match="All sequences must have the same length"):
            Alignment(sequences, labels)
    
    def test_repr(self):
        """Test string representation of Alignment."""
        sequences = [['D3', 'F3'], ['D3', 'G3']]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        repr_str = repr(alignment)
        
        assert "Alignment(n_sequences=2, length=2)" in repr_str
    
    def test_repr_with_score(self):
        """Test string representation with score."""
        sequences = [['D3', 'F3'], ['D3', 'G3']]
        labels = ['seq1', 'seq2']
        score = 1.5
        
        alignment = Alignment(sequences, labels, score=score)
        repr_str = repr(alignment)
        
        assert "score=1.5" in repr_str


class TestAlignmentProperties:
    """Test Alignment analysis properties."""
    
    def test_gap_counts_no_gaps(self):
        """Test gap counts with no gaps."""
        sequences = [
            ['D3', 'F3', 'G3'],
            ['D3', 'E3', 'G3']
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        
        assert alignment.gap_counts == [0, 0]
    
    def test_gap_counts_with_gaps(self):
        """Test gap counts with various gap patterns."""
        sequences = [
            ['D3', None, 'G3', None],
            [None, 'F3', 'G3', 'A3'],
            ['D3', 'F3', None, None]
        ]
        labels = ['seq1', 'seq2', 'seq3']
        
        alignment = Alignment(sequences, labels)
        
        assert alignment.gap_counts == [2, 1, 2]
    
    def test_gap_counts_caching(self):
        """Test that gap counts are cached."""
        sequences = [['D3', None], [None, 'F3']]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        
        # First call should compute and cache
        gaps1 = alignment.gap_counts
        # Second call should return cached result
        gaps2 = alignment.gap_counts
        
        assert gaps1 == gaps2 == [1, 1]
        assert alignment._gap_counts is not None
    
    def test_match_statistics_perfect_match(self):
        """Test match statistics with perfect matches."""
        sequences = [
            ['D3', 'F3', 'G3'],
            ['D3', 'F3', 'G3']
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        stats = alignment.match_statistics
        
        assert stats['matches'] == 3
        assert stats['mismatches'] == 0
        assert stats['gaps'] == 0
        assert stats['positions'] == 3
        assert stats['match_rate'] == 1.0
    
    def test_match_statistics_with_mismatches(self):
        """Test match statistics with mismatches."""
        sequences = [
            ['D3', 'F3', 'G3'],
            ['D3', 'E3', 'A3']
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        stats = alignment.match_statistics
        
        assert stats['matches'] == 1  # Only first position matches
        assert stats['mismatches'] == 2  # Second and third positions
        assert stats['gaps'] == 0
        assert stats['positions'] == 3
        assert stats['match_rate'] == 1/3
    
    def test_match_statistics_with_gaps(self):
        """Test match statistics with gaps."""
        sequences = [
            ['D3', None, 'G3'],
            [None, 'F3', 'G3']
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        stats = alignment.match_statistics
        
        assert stats['matches'] == 1  # Only third position matches
        assert stats['mismatches'] == 0  # No positions with differing non-None tokens
        assert stats['gaps'] == 2  # Two None values total
        assert stats['positions'] == 3
        assert stats['match_rate'] == 1/3
    
    def test_match_statistics_complex(self):
        """Test match statistics with complex pattern."""
        sequences = [
            ['D3', 'F3', None, 'G3'],  # 1 gap
            ['D3', 'E3', 'F3', 'G3'],  # 0 gaps
            [None, 'F3', 'F3', 'A3']   # 1 gap
        ]
        labels = ['seq1', 'seq2', 'seq3']
        
        alignment = Alignment(sequences, labels)
        stats = alignment.match_statistics
        
        # Position 0: D3, D3, None -> not a match (has gap)
        # Position 1: F3, E3, F3 -> mismatch (E3 differs)
        # Position 2: None, F3, F3 -> not counted as match or mismatch (has gap)
        # Position 3: G3, G3, A3 -> mismatch (A3 differs)
        
        assert stats['matches'] == 0
        assert stats['mismatches'] == 2  # Positions 1 and 3
        assert stats['gaps'] == 2  # Two None values total
        assert stats['positions'] == 4
        assert stats['match_rate'] == 0.0
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        sequences = [['D3', None], [None, 'F3']]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        
        # Access properties to populate cache
        _ = alignment.gap_counts
        _ = alignment.match_statistics
        
        assert alignment._gap_counts is not None
        assert alignment._match_statistics is not None
        
        # Invalidate cache
        alignment.invalidate_cache()
        
        assert alignment._gap_counts is None
        assert alignment._match_statistics is None


class TestFromNeedlemanWunsch:
    """Test the from_needleman_wunsch class method."""
    
    def test_basic_construction(self):
        """Test basic construction from Needleman-Wunsch results."""
        aligned_seq1 = ['D3', 'F3', None, 'G3']
        aligned_seq2 = ['D3', None, 'E3', 'G3']
        score = 1.5
        
        alignment = Alignment.from_needleman_wunsch(
            aligned_seq1, aligned_seq2, score
        )
        
        assert alignment.sequences == [aligned_seq1, aligned_seq2]
        assert alignment.labels == ['seq1', 'seq2']
        assert alignment.score == score
        assert alignment.algorithm == "Needleman-Wunsch"
    
    def test_custom_labels(self):
        """Test construction with custom labels."""
        aligned_seq1 = ['D3', 'F3']
        aligned_seq2 = ['D3', 'G3']
        score = 2.0
        label1 = "AdTeLevavi/V01"
        label2 = "AdTeLevavi/V02"
        
        alignment = Alignment.from_needleman_wunsch(
            aligned_seq1, aligned_seq2, score, label1, label2
        )
        
        assert alignment.labels == [label1, label2]
        assert alignment.score == score
        assert alignment.algorithm == "Needleman-Wunsch"


class TestToDataFrame:
    """Test the to_dataframe method."""
    
    def test_basic_dataframe(self):
        """Test basic DataFrame conversion."""
        sequences = [
            ['D3', 'F3', None],
            ['D3', None, 'G3']
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        df = alignment.to_dataframe()
        
        expected_columns = ['position', 'seq1', 'seq2', 'has_gap', 'all_match']
        assert list(df.columns) == expected_columns
        assert len(df) == 3
        
        # Check position column
        assert list(df['position']) == [0, 1, 2]
        
        # Check sequence columns
        assert list(df['seq1']) == ['D3', 'F3', None]
        assert list(df['seq2']) == ['D3', None, 'G3']
    
    def test_dataframe_metadata_columns(self):
        """Test metadata columns in DataFrame."""
        sequences = [
            ['D3', 'F3', None, 'G3'],  # Perfect match, mismatch, gap, match
            ['D3', 'E3', 'A3', 'G3']   # Perfect match, mismatch, non-gap, match
        ]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        df = alignment.to_dataframe()
        
        expected_has_gap = [False, False, True, False]
        # Position 2: None vs A3 -> single non-gap token, so considered a "match"
        expected_all_match = [True, False, True, True]
        
        assert list(df['has_gap']) == expected_has_gap
        assert list(df['all_match']) == expected_all_match
    
    def test_dataframe_empty_alignment(self):
        """Test DataFrame conversion with empty alignment."""
        sequences = [[], []]
        labels = ['seq1', 'seq2']
        
        alignment = Alignment(sequences, labels)
        df = alignment.to_dataframe()
        
        assert len(df) == 0
        expected_columns = ['position', 'seq1', 'seq2', 'has_gap', 'all_match']
        assert list(df.columns) == expected_columns
    
    def test_dataframe_multiple_sequences(self):
        """Test DataFrame conversion with multiple sequences."""
        sequences = [
            ['D3', 'F3'],
            ['D3', 'E3'],
            ['D3', 'F3']
        ]
        labels = ['tract1', 'tract2', 'tract3']
        
        alignment = Alignment(sequences, labels)
        df = alignment.to_dataframe()
        
        expected_columns = ['position', 'tract1', 'tract2', 'tract3', 'has_gap', 'all_match']
        assert list(df.columns) == expected_columns
        
        # Position 0: all D3 -> perfect match
        # Position 1: F3, E3, F3 -> mismatch (E3 differs)
        assert list(df['all_match']) == [True, False]
        assert list(df['has_gap']) == [False, False]


class TestAlignmentIntegration:
    """Test integration with other components."""
    
    def test_integration_with_needleman_wunsch(self):
        """Test integration with actual needleman_wunsch function."""
        # Import here to avoid circular dependencies in tests
        from segalign.glob import needleman_wunsch
        
        seq1 = ['D3', 'F3', 'G3']
        seq2 = ['D3', 'G3']
        
        # Run alignment algorithm
        aligned1, aligned2, score = needleman_wunsch(seq1, seq2)
        
        # Create Alignment object
        alignment = Alignment.from_needleman_wunsch(
            aligned1, aligned2, score, 'test1', 'test2'
        )
        
        # Verify the alignment makes sense
        assert len(alignment) == len(aligned1) == len(aligned2)
        assert alignment.score == score
        assert alignment.algorithm == "Needleman-Wunsch"
        
        # Check that original sequences are preserved (accounting for gaps)
        seq1_recovered = [token for token in aligned1 if token is not None]
        seq2_recovered = [token for token in aligned2 if token is not None]
        
        assert seq1_recovered == seq1
        assert seq2_recovered == seq2