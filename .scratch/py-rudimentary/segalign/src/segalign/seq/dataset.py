"""
Dataset class for musical sequence data management and analysis.

This module provides the Dataset class for organizing musical token sequences,
labels, and detailed sequence data for computational music theory research
and sequence alignment algorithms.
"""

from typing import List, Optional, Dict
import pandas as pd
from pathlib import Path
from collections import Counter


class Dataset:
    """
    Container for musical sequence datasets used in alignment algorithms.
    
    The Dataset class organizes related musical sequences with their metadata,
    token vocabulary, and detailed annotations, designed for use with sequence 
    alignment algorithms like SS-2, Smith-Waterman variants, and T-Coffee adaptations.
    
    Attributes:
        labels: List of string identifiers for each sequence (e.g., tract names, verse IDs)
        toks: List of token strings representing the vocabulary of musical tokens
              (pitches like 'D3', 'F3', boundaries like 'Maior', 'Finalis')
        toks_enc: Dictionary mapping integer indices to token strings for decoding
        dframes: List of pandas DataFrames containing detailed sequence data with
               columns for Text, Stress, Boundary, Pitch, and Token information
        focus_column: String indicating which column the vocabulary was built from (default: 'Pitch')
        sequences: Property returning List[List[str]] - token sequences extracted from focus column
        encoded_sequences: Property returning List[List[int]] - integer-encoded token sequences
    
    Example:
        >>> labels = ["AdTeLevavi/V01", "AdTeLevavi/V02"]
        >>> toks = [":EOS", ":PAD", "D3", "F3", "G3", "Maior"]
        >>> toks_enc = {":EOS": 0, ":PAD": 1, "D3": 2, "F3": 3}
        >>> dframes = [df1, df2]  # pandas DataFrames with detailed annotations
        >>> dataset = Dataset(labels, toks, toks_enc, dframes)
        >>> len(dataset.labels)  # Number of sequences in dataset
        2
    """
    
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        toks: Optional[List[str]] = None,
        toks_enc: Optional[Dict[str, int]] = None,
        dframes: Optional[List[pd.DataFrame]] = None,
        focus_column: str = 'Pitch'
    ) -> None:
        """
        Initialize a Dataset with musical sequence data.
        
        Args:
            labels: List of string identifiers for sequences. Defaults to empty list.
            toks: List of token strings (vocabulary). Defaults to [':EOS', ':PAD'].
            toks_enc: Dictionary mapping tokens to indices. Defaults to {':EOS': 0, ':PAD': 1}.
            dframes: List of pandas DataFrames with detailed sequence data. Defaults to empty list.
            focus_column: Column name that vocabulary was built from. Defaults to 'Pitch'.
            
        Raises:
            ValueError: If labels and dframes lists have mismatched lengths.
        """
        self.labels = labels if labels is not None else []
        self.toks = toks if toks is not None else [':EOS', ':PAD']
        self.toks_enc = toks_enc if toks_enc is not None else {':EOS': 0, ':PAD': 1}
        self.dframes = dframes if dframes is not None else []
        self.focus_column = focus_column
        
        # Cache for sequence extraction
        self._sequences = None
        self._encoded_sequences = None
        
        # Validate that labels and dframes have the same length when both are provided
        if self.labels and self.dframes and len(self.labels) != len(self.dframes):
            raise ValueError(
                f"labels and dframes must have the same length. "
                f"Got lengths: labels={len(self.labels)}, dframes={len(self.dframes)}"
            )
    
    def __repr__(self) -> str:
        """Return string representation of the Dataset."""
        n_seqs = len(self.labels)
        return f"Dataset(n_sequences={n_seqs}, labels={self.labels[:3]}{'...' if n_seqs > 3 else ''})"
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.labels)
    
    def _extract_sequences(self) -> List[List[str]]:
        """
        Extract token sequences from DataFrames using the focus column.
        
        Returns:
            List of token sequences, where each sequence is a list of string tokens
            extracted from the focus column of the corresponding DataFrame.
            Empty sequences are returned for DataFrames missing the focus column.
        """
        sequences = []
        for df in self.dframes:
            if self.focus_column in df.columns:
                # Extract non-null values from focus column, excluding '.' tokens
                sequence = df[self.focus_column].dropna()
                sequence = [token for token in sequence if token != '.']
                sequences.append(sequence)
            else:
                # Return empty sequence if focus column is missing
                sequences.append([])
        return sequences
    
    def _encode_sequences(self, sequences: List[List[str]]) -> List[List[int]]:
        """
        Encode string sequences to integer sequences using the token encoding dictionary.
        
        Args:
            sequences: List of token sequences as string lists.
            
        Returns:
            List of encoded sequences, where each sequence is a list of integer token indices.
            Unknown tokens are mapped to the ':PAD' token index for robustness.
        """
        encoded = []
        pad_index = self.toks_enc.get(':PAD', 1)  # Default PAD index is 1
        
        for seq in sequences:
            encoded_seq = []
            for token in seq:
                # Get token index, defaulting to PAD index for unknown tokens
                token_index = self.toks_enc.get(token, pad_index)
                encoded_seq.append(token_index)
            encoded.append(encoded_seq)
        
        return encoded
    
    @property
    def sequences(self) -> List[List[str]]:
        """
        Get token sequences as lists of strings from the focus column.
        
        This property lazily computes and caches token sequences extracted from
        the focus column of each DataFrame. Subsequent access returns the cached
        result for performance.
        
        Returns:
            List of token sequences, where each sequence corresponds to a DataFrame
            in self.dframes and contains string tokens from the focus column.
            Empty sequences are returned for DataFrames missing the focus column.
            
        Example:
            >>> dataset = Dataset.from_directories("chant/greg/tract/m8", tract_names=["AdTeLevavi"])
            >>> sequences = dataset.sequences
            >>> print(sequences[0][:5])  # First 5 tokens of first sequence
            ['D3', 'F3', 'G3', 'G3', 'E3']
        """
        if self._sequences is None:
            self._sequences = self._extract_sequences()
        return self._sequences
    
    @property
    def encoded_sequences(self) -> List[List[int]]:
        """
        Get token sequences as lists of encoded integers.
        
        This property lazily computes and caches integer-encoded token sequences
        using the toks_enc dictionary. Tokens are mapped to their corresponding
        indices, with unknown tokens defaulting to the ':PAD' token index.
        
        Returns:
            List of encoded sequences, where each sequence corresponds to a DataFrame
            in self.dframes and contains integer indices for tokens from the focus column.
            Empty sequences are returned for DataFrames missing the focus column.
            
        Example:
            >>> dataset = Dataset.from_directories("chant/greg/tract/m8", tract_names=["AdTeLevavi"])
            >>> encoded = dataset.encoded_sequences
            >>> print(encoded[0][:5])  # First 5 encoded tokens of first sequence
            [9, 6, 3, 3, 10]
        """
        if self._encoded_sequences is None:
            self._encoded_sequences = self._encode_sequences(self.sequences)
        return self._encoded_sequences
    
    def invalidate_cache(self) -> None:
        """
        Clear cached sequence data to force recomputation on next access.
        
        This method should be called when the underlying data (dframes, focus_column,
        or toks_enc) has been modified and cached sequences are no longer valid.
        Subsequent access to the sequences or encoded_sequences properties will
        trigger fresh computation.
        
        Example:
            >>> dataset = Dataset.from_directories("chant/greg/tract/m8")
            >>> _ = dataset.sequences  # Compute and cache sequences
            >>> dataset.focus_column = "Boundary"  # Change focus column
            >>> dataset.invalidate_cache()  # Clear stale cache
            >>> boundary_sequences = dataset.sequences  # Recompute with new column
        """
        self._sequences = None
        self._encoded_sequences = None
    
    @classmethod
    def from_directories(
        cls,
        base_path: str,
        tract_names: Optional[List[str]] = None,
        focus_column: str = 'Pitch'
    ) -> 'Dataset':
        """
        Create a Dataset by loading data from nested directory structure.
        
        This convenience constructor scans the specified base path for tract
        directories and their verse subdirectories, loading All.csv files
        as detailed sequence data. It automatically builds the token vocabulary
        by collecting unique values from the specified column across all verses.
        
        Args:
            base_path: Relative path from package data directory containing tract subdirectories
            tract_names: Optional list of specific tract names to load. If None,
                        loads all available tracts in the directory.
            focus_column: Column name to extract tokens from (default: 'Pitch').
                         Common options: 'Pitch', 'Token', 'Boundary', 'Text', 'Stress'.
            
        Returns:
            Dataset instance populated with data from the directory structure.
            The toks list will contain [':EOS', ':PAD'] followed by sorted unique
            values from the specified column across all loaded verses.
            
        Example:
            >>> # Load all tracts using default 'Pitch' column for vocabulary
            >>> dataset = Dataset.from_directories("chant/greg/tract/m8")
            >>> # Load specific tracts with custom column
            >>> dataset = Dataset.from_directories(
            ...     "chant/greg/tract/m8",
            ...     tract_names=["AdTeLevavi", "DeProfundis"],
            ...     column_label="Token"
            ... )
        """
        # Use __file__ to locate the package data directory relative to this module
        module_dir = Path(__file__).parent.parent  # Go up to segalign/ from seq/
        data_dir = module_dir / "data"
        base_path = data_dir / base_path
        labels = []
        dframes = []
        
        # Check if base path exists
        if not base_path.exists():
            return cls(labels=labels, dframes=dframes)
        
        # Get tract directories to process
        if tract_names is None:
            # Find all tract directories
            tract_dirs = [d for d in base_path.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
        else:
            # Use specified tract names
            tract_dirs = [base_path / name for name in tract_names
                         if (base_path / name).exists()]
        
        toks = [':EOS', ':PAD']
        toks_enc = {}
        collected_tokens = Counter()

        # Process each tract directory
        for tract_dir in sorted(tract_dirs):
            tract_name = tract_dir.name
            
            # Find verse directories (V01, V02, etc.)
            verse_dirs = [d for d in tract_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('V')]
            
            # Process each verse
            for verse_dir in sorted(verse_dirs):
                verse_name = verse_dir.name
                all_csv_path = verse_dir / "All.csv"
                
                # Load the detailed sequence data if All.csv exists
                if all_csv_path.exists():
                    try:
                        df = pd.read_csv(all_csv_path, sep='\t')
                        labels.append(f"{tract_name}/{verse_name}")
                        dframes.append(df)
                        
                        # Collect tokens and count their frequencies
                        if focus_column in df.columns:
                            for token in df[focus_column].dropna():
                                collected_tokens[token] += 1
                    except Exception as e:
                        print(f"Warning: Could not load {all_csv_path}: {e}")

        # Add collected tokens to vocabulary (sorted by frequency)
        if collected_tokens:
            # Remove the '.' token
            if '.' in collected_tokens:
                del collected_tokens['.']
            # Sort by frequency (most frequent first)
            sorted_tokens = [token for token, _ in collected_tokens.most_common()]
            toks.extend(sorted_tokens)

        for i, tok in enumerate(toks):
            toks_enc[tok] = i
        
        return cls(labels=labels, toks=toks, toks_enc=toks_enc, dframes=dframes, focus_column=focus_column)