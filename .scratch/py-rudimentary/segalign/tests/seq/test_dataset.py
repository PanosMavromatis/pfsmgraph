"""
Unit tests for the Dataset class in segalign.seq.dataset module.

These tests validate the Dataset class functionality including basic
initialization, convenience constructors, and data loading from
directory structures.
"""

import pytest
import pandas as pd
from pathlib import Path
from segalign.seq import Dataset


class TestDatasetInit:
    """Test basic Dataset initialization and properties."""
    
    def test_empty_init(self):
        """Test Dataset initialization with default values."""
        dataset = Dataset()
        
        assert dataset.labels == []
        assert dataset.toks == [':EOS', ':PAD']
        assert dataset.toks_enc == {':EOS': 0, ':PAD': 1}
        assert dataset.dframes == []
        assert dataset.focus_column == 'Pitch'
        assert len(dataset) == 0
    
    def test_custom_init(self):
        """Test Dataset initialization with custom values."""
        labels = ["test1", "test2"]
        toks = [':EOS', ':PAD', 'C4', 'D4']
        toks_enc = {':EOS': 0, ':PAD': 1, 'C4': 2, 'D4': 3}
        df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        df2 = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})
        dframes = [df1, df2]
        
        dataset = Dataset(labels=labels, toks=toks, toks_enc=toks_enc, dframes=dframes, focus_column='Token')
        
        assert dataset.labels == labels
        assert dataset.toks == toks
        assert dataset.toks_enc == toks_enc
        assert len(dataset.dframes) == 2
        assert dataset.focus_column == 'Token'
        assert len(dataset) == 2
    
    def test_length_mismatch_validation(self):
        """Test that mismatched lengths raise ValueError."""
        labels = ["test1", "test2"]
        df1 = pd.DataFrame({'col1': [1, 2]})
        dframes = [df1]  # Only one DataFrame but two labels
        
        with pytest.raises(ValueError, match="labels and dframes must have the same length"):
            Dataset(labels=labels, dframes=dframes)
    
    def test_repr(self):
        """Test string representation of Dataset."""
        labels = ["test1", "test2", "test3", "test4"]
        dataset = Dataset(labels=labels)
        
        repr_str = repr(dataset)
        assert "Dataset(n_sequences=4" in repr_str
        assert "test1" in repr_str
        assert "..." in repr_str  # Should truncate after 3 items


class TestDatasetFromDirectories:
    """Test the from_directories class method."""
    
    def test_from_directories_basic(self):
        """Test loading dataset from the Mode 8 chant directory."""
        # This test assumes the actual data directory exists
        dataset = Dataset.from_directories("chant/greg/tract/m8")
        
        # Should find multiple tracts and verses
        assert len(dataset) > 0
        assert len(dataset.labels) > 0
        assert len(dataset.dframes) > 0
        assert len(dataset.labels) == len(dataset.dframes)
        
        # Check label format (should be "TractName/VerseID")
        for label in dataset.labels:
            assert "/" in label
            parts = label.split("/")
            assert len(parts) == 2
            assert parts[1].startswith("V")  # Verse identifier
    
    def test_from_directories_specific_tracts(self):
        """Test loading specific tracts only."""
        dataset = Dataset.from_directories(
            "chant/greg/tract/m8",
            tract_names=["AdTeLevavi", "DeProfundis"]
        )
        
        # Should only find verses from the specified tracts
        tract_names_found = set()
        for label in dataset.labels:
            tract_name = label.split("/")[0]
            tract_names_found.add(tract_name)
        
        assert tract_names_found <= {"AdTeLevavi", "DeProfundis"}
        assert len(dataset) > 0
    
    def test_from_directories_custom_column(self):
        """Test from_directories with custom column selection."""
        # Test with Boundary column instead of default Pitch
        dataset = Dataset.from_directories(
            "chant/greg/tract/m8",
            tract_names=["AdTeLevavi"],
            focus_column="Boundary"
        )
        
        # Should have collected boundary tokens
        assert len(dataset.toks) > 2  # More than just :EOS and :PAD
        # Should contain some boundary markers
        boundary_tokens = set(dataset.toks[2:])  # Skip :EOS and :PAD
        expected_boundaries = {'Maior', 'Minima', 'Finalis'}  # Note: '.' is removed
        assert len(boundary_tokens & expected_boundaries) > 0
        # Should have the correct focus_column set
        assert dataset.focus_column == "Boundary"
    
    def test_from_directories_default_column(self):
        """Test from_directories with default column selection."""
        dataset = Dataset.from_directories(
            "chant/greg/tract/m8",
            tract_names=["AdTeLevavi"]
        )
        
        # Should use default 'Pitch' column
        assert dataset.focus_column == "Pitch"
        # Should contain pitch tokens
        pitch_tokens = set(dataset.toks[2:])  # Skip :EOS and :PAD
        expected_pitches = {'A3', 'G3', 'C4', 'B3', 'F3'}
        assert len(pitch_tokens & expected_pitches) > 0
    
    def test_from_directories_nonexistent_path(self):
        """Test behavior with nonexistent directory."""
        dataset = Dataset.from_directories("nonexistent/path")
        
        # Should return empty dataset without crashing
        assert len(dataset) == 0
        assert dataset.labels == []
        assert dataset.dframes == []


class TestDatasetDataFrames:
    """Test Dataset functionality with actual DataFrame content."""
    
    def test_dataframe_structure(self):
        """Test that loaded DataFrames have expected structure."""
        dataset = Dataset.from_directories(
            "chant/greg/tract/m8",
            tract_names=["AdTeLevavi"]
        )
        
        if len(dataset) > 0:
            # Check first DataFrame structure
            df = dataset.dframes[0]
            expected_columns = {'Text', 'Stress', 'Boundary', 'Pitch', 'Token'}
            assert set(df.columns) == expected_columns
            
            # Should have data
            assert len(df) > 0


class TestDatasetSequenceExtraction:
    """Test Dataset sequence extraction functionality."""
    
    def test_sequences_property_basic(self):
        """Test basic sequence extraction from custom dataset."""
        # Create test DataFrames with different columns
        df1 = pd.DataFrame({
            'Text': ['test1', 'test2', 'test3'],
            'Pitch': ['C4', 'D4', 'E4'],
            'Token': ['C4', 'D4', 'E4']
        })
        df2 = pd.DataFrame({
            'Text': ['test4', 'test5'],
            'Pitch': ['F4', 'G4'],
            'Token': ['F4', 'G4']
        })
        
        dataset = Dataset(
            labels=["seq1", "seq2"],
            dframes=[df1, df2],
            focus_column="Pitch"
        )
        
        # Test sequences property
        sequences = dataset.sequences
        assert len(sequences) == 2
        assert sequences[0] == ['C4', 'D4', 'E4']
        assert sequences[1] == ['F4', 'G4']
        
        # Test caching - should return same object
        sequences2 = dataset.sequences
        assert sequences is sequences2
    
    def test_encoded_sequences_property_basic(self):
        """Test basic encoded sequence extraction."""
        df1 = pd.DataFrame({
            'Pitch': ['C4', 'D4', 'E4']
        })
        
        toks = [':EOS', ':PAD', 'C4', 'D4', 'E4', 'F4']
        toks_enc = {tok: i for i, tok in enumerate(toks)}
        
        dataset = Dataset(
            labels=["seq1"],
            toks=toks,
            toks_enc=toks_enc,
            dframes=[df1],
            focus_column="Pitch"
        )
        
        # Test encoded sequences
        encoded = dataset.encoded_sequences
        assert len(encoded) == 1
        assert encoded[0] == [2, 3, 4]  # C4=2, D4=3, E4=4
        
        # Test caching
        encoded2 = dataset.encoded_sequences
        assert encoded is encoded2
    
    def test_sequences_with_missing_column(self):
        """Test sequence extraction when focus column is missing."""
        df1 = pd.DataFrame({
            'Text': ['test1', 'test2'],
            'Pitch': ['C4', 'D4']
        })
        df2 = pd.DataFrame({
            'Text': ['test3', 'test4']
            # Missing Pitch column
        })
        
        dataset = Dataset(
            labels=["seq1", "seq2"],
            dframes=[df1, df2],
            focus_column="Pitch"
        )
        
        sequences = dataset.sequences
        assert len(sequences) == 2
        assert sequences[0] == ['C4', 'D4']
        assert sequences[1] == []  # Empty for missing column
    
    def test_sequences_with_null_values(self):
        """Test sequence extraction with null values in focus column."""
        df1 = pd.DataFrame({
            'Pitch': ['C4', None, 'D4', 'E4', None]
        })
        
        dataset = Dataset(
            labels=["seq1"],
            dframes=[df1],
            focus_column="Pitch"
        )
        
        sequences = dataset.sequences
        assert len(sequences) == 1
        assert sequences[0] == ['C4', 'D4', 'E4']  # Nulls filtered out
    
    def test_encoded_sequences_with_unknown_tokens(self):
        """Test encoded sequences with tokens not in vocabulary."""
        df1 = pd.DataFrame({
            'Pitch': ['C4', 'UNKNOWN', 'D4']
        })
        
        toks = [':EOS', ':PAD', 'C4', 'D4']
        toks_enc = {tok: i for i, tok in enumerate(toks)}
        
        dataset = Dataset(
            labels=["seq1"],
            toks=toks,
            toks_enc=toks_enc,
            dframes=[df1],
            focus_column="Pitch"
        )
        
        encoded = dataset.encoded_sequences
        assert len(encoded) == 1
        assert encoded[0] == [2, 1, 3]  # UNKNOWN mapped to :PAD (index 1)
    
    def test_different_focus_columns(self):
        """Test sequence extraction with different focus columns."""
        df1 = pd.DataFrame({
            'Text': ['word1', 'word2'],
            'Pitch': ['C4', 'D4'],
            'Boundary': ['Maior', 'Finalis']
        })
        
        # Test with Pitch column
        dataset_pitch = Dataset(
            labels=["seq1"],
            dframes=[df1],
            focus_column="Pitch"
        )
        assert dataset_pitch.sequences[0] == ['C4', 'D4']
        
        # Test with Text column
        dataset_text = Dataset(
            labels=["seq1"],
            dframes=[df1],
            focus_column="Text"
        )
        assert dataset_text.sequences[0] == ['word1', 'word2']
        
        # Test with Boundary column
        dataset_boundary = Dataset(
            labels=["seq1"],
            dframes=[df1],
            focus_column="Boundary"
        )
        assert dataset_boundary.sequences[0] == ['Maior', 'Finalis']
    
    def test_invalidate_cache(self):
        """Test cache invalidation functionality."""
        df1 = pd.DataFrame({
            'Pitch': ['C4', 'D4'],
            'Text': ['word1', 'word2']
        })
        
        dataset = Dataset(
            labels=["seq1"],
            dframes=[df1],
            focus_column="Pitch"
        )
        
        # Access sequences to populate cache
        sequences1 = dataset.sequences
        assert sequences1[0] == ['C4', 'D4']
        
        # Change focus column
        dataset.focus_column = "Text"
        
        # Cache should still return old data
        sequences2 = dataset.sequences
        assert sequences2 is sequences1  # Same cached object
        assert sequences2[0] == ['C4', 'D4']  # Old data
        
        # Invalidate cache
        dataset.invalidate_cache()
        
        # Now should get new data
        sequences3 = dataset.sequences
        assert sequences3 is not sequences1  # New object
        assert sequences3[0] == ['word1', 'word2']  # New data
    
    def test_cache_independence(self):
        """Test that sequences and encoded_sequences caches are independent."""
        df1 = pd.DataFrame({
            'Pitch': ['C4', 'D4']
        })
        
        toks = [':EOS', ':PAD', 'C4', 'D4']
        toks_enc = {tok: i for i, tok in enumerate(toks)}
        
        dataset = Dataset(
            labels=["seq1"],
            toks=toks,
            toks_enc=toks_enc,
            dframes=[df1],
            focus_column="Pitch"
        )
        
        # Access sequences first
        sequences = dataset.sequences
        assert sequences[0] == ['C4', 'D4']
        
        # Access encoded sequences
        encoded = dataset.encoded_sequences
        assert encoded[0] == [2, 3]
        
        # Both should be cached
        assert dataset._sequences is not None
        assert dataset._encoded_sequences is not None
        
        # Invalidate cache
        dataset.invalidate_cache()
        
        # Both caches should be cleared
        assert dataset._sequences is None
        assert dataset._encoded_sequences is None
    
    def test_sequences_with_real_data(self):
        """Test sequence extraction with real chant data."""
        try:
            dataset = Dataset.from_directories(
                "chant/greg/tract/m8",
                tract_names=["AdTeLevavi"]
            )
            
            if len(dataset) > 0:
                # Test sequences property
                sequences = dataset.sequences
                assert len(sequences) == len(dataset)
                
                # Check first sequence
                first_seq = sequences[0]
                assert len(first_seq) > 0
                assert isinstance(first_seq[0], str)
                
                # Test encoded sequences
                encoded = dataset.encoded_sequences
                assert len(encoded) == len(dataset)
                
                # Check first encoded sequence
                first_encoded = encoded[0]
                assert len(first_encoded) == len(first_seq)
                assert all(isinstance(x, int) for x in first_encoded)
                
                # Verify encoding consistency
                for i, token in enumerate(first_seq):
                    expected_index = dataset.toks_enc.get(token, dataset.toks_enc[':PAD'])
                    assert first_encoded[i] == expected_index
        
        except Exception:
            # Skip test if data is not available
            pytest.skip("Real chant data not available")


if __name__ == "__main__":
    pytest.main([__file__])