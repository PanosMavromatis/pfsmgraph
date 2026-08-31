"""
Alignment class for musical sequence alignment results.

This module provides the Alignment class for storing and analyzing the results
of sequence alignment algorithms applied to musical token sequences.
"""

from typing import List, Optional, Dict, Union
import pandas as pd


class Alignment:
    """
    Container for aligned musical token sequences with analysis capabilities.
    
    The Alignment class stores the results of sequence alignment algorithms,
    providing convenient access to aligned sequences, gap analysis, and match
    statistics. Designed to work with both pairwise and multiple sequence
    alignments from various algorithms like Needleman-Wunsch, Smith-Waterman,
    and T-Coffee adaptations.
    
    Gaps in aligned sequences are represented as None values, making them
    easily identifiable for analysis and visualization purposes.
    
    Attributes:
        sequences: List of aligned sequences, where each sequence is a list of
                  string tokens with None representing gaps
        labels: List of string identifiers for each sequence (e.g., tract names)
        score: Optional alignment score from the algorithm (default: None)
        algorithm: Optional name of the alignment algorithm used (default: None)
        alignment_length: Property returning the length of the aligned sequences
        gap_counts: Property returning gap counts per sequence
        match_statistics: Property returning match/mismatch/gap statistics
    
    Example:
        >>> sequences = [
        ...     ['D3', 'F3', None, 'G3'],
        ...     ['D3', None, 'E3', 'G3']
        ... ]
        >>> labels = ['AdTeLevavi/V01', 'AdTeLevavi/V02']
        >>> alignment = Alignment(sequences, labels, score=1.5)
        >>> len(alignment)
        4
        >>> alignment.gap_counts
        [1, 1]
    """
    
    def __init__(
        self,
        sequences: List[List[Optional[str]]],
        labels: List[str],
        score: Optional[float] = None,
        algorithm: Optional[str] = None
    ) -> None:
        """
        Initialize an Alignment with aligned sequences and metadata.
        
        Args:
            sequences: List of aligned sequences, where each sequence is a list
                      of string tokens with None representing gaps. All sequences
                      must have the same length.
            labels: List of string identifiers for each sequence. Must have the
                   same length as the sequences list.
            score: Optional alignment score from the algorithm that produced
                  this alignment.
            algorithm: Optional name of the alignment algorithm used to create
                      this alignment (e.g., "Needleman-Wunsch").
        
        Raises:
            ValueError: If sequences and labels have different lengths, if
                       sequences have different lengths, or if fewer than 2
                       sequences are provided.
        """
        if len(sequences) < 2:
            raise ValueError(f"At least 2 sequences required, got {len(sequences)}")
        
        if len(sequences) != len(labels):
            raise ValueError(
                f"sequences and labels must have the same length. "
                f"Got lengths: sequences={len(sequences)}, labels={len(labels)}"
            )
        
        # Check that all sequences have the same length
        seq_lengths = [len(seq) for seq in sequences]
        if len(set(seq_lengths)) > 1:
            raise ValueError(
                f"All sequences must have the same length. "
                f"Got lengths: {seq_lengths}"
            )
        
        self.sequences = sequences
        self.labels = labels
        self.score = score
        self.algorithm = algorithm
        
        # Cache for computed properties
        self._gap_counts = None
        self._match_statistics = None
    
    def __repr__(self) -> str:
        """Return string representation of the Alignment."""
        n_seqs = len(self.sequences)
        length = len(self.sequences[0]) if self.sequences else 0
        score_str = f", score={self.score}" if self.score is not None else ""
        return f"Alignment(n_sequences={n_seqs}, length={length}{score_str})"
    
    def __len__(self) -> int:
        """Return the length of the aligned sequences."""
        return len(self.sequences[0]) if self.sequences else 0
    
    @property
    def alignment_length(self) -> int:
        """Length of the aligned sequences."""
        return len(self)
    
    @property
    def gap_counts(self) -> List[int]:
        """
        Number of gaps (None values) in each aligned sequence.
        
        Returns:
            List of integers, where each integer is the number of gaps
            in the corresponding sequence.
        """
        if self._gap_counts is None:
            self._gap_counts = [
                sum(1 for token in seq if token is None)
                for seq in self.sequences
            ]
        return self._gap_counts
    
    @property
    def match_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Comprehensive statistics about matches, mismatches, and gaps.
        
        Returns:
            Dictionary containing:
                - 'matches': Number of positions where all sequences have identical tokens
                - 'mismatches': Number of positions with differing non-None tokens
                - 'gaps': Total number of gap positions across all sequences
                - 'positions': Total alignment length
                - 'match_rate': Fraction of positions that are perfect matches
        """
        if self._match_statistics is None:
            matches = 0
            mismatches = 0
            total_gaps = sum(self.gap_counts)
            
            for i in range(self.alignment_length):
                position_tokens = [seq[i] for seq in self.sequences]
                non_gap_tokens = [token for token in position_tokens if token is not None]
                
                if len(non_gap_tokens) == len(position_tokens):  # No gaps at this position
                    if len(set(non_gap_tokens)) == 1:  # All tokens identical
                        matches += 1
                    else:  # Different tokens
                        mismatches += 1
                # Positions with gaps are neither matches nor mismatches
            
            self._match_statistics = {
                'matches': matches,
                'mismatches': mismatches,
                'gaps': total_gaps,
                'positions': self.alignment_length,
                'match_rate': matches / self.alignment_length if self.alignment_length > 0 else 0.0
            }
        
        return self._match_statistics
    
    def invalidate_cache(self) -> None:
        """Clear cached computed properties."""
        self._gap_counts = None
        self._match_statistics = None
    
    @classmethod
    def from_needleman_wunsch(
        cls,
        aligned_seq1: List[Optional[str]],
        aligned_seq2: List[Optional[str]],
        score: float,
        label1: str = "seq1",
        label2: str = "seq2"
    ) -> "Alignment":
        """
        Create an Alignment from Needleman-Wunsch algorithm results.
        
        Convenience constructor for creating Alignment objects directly from
        the output of the needleman_wunsch function.
        
        Args:
            aligned_seq1: First aligned sequence from needleman_wunsch
            aligned_seq2: Second aligned sequence from needleman_wunsch
            score: Alignment score from needleman_wunsch
            label1: Label for the first sequence (default: "seq1")
            label2: Label for the second sequence (default: "seq2")
        
        Returns:
            Alignment object containing the aligned sequences and metadata.
        
        Example:
            >>> from segalign.glob import needleman_wunsch
            >>> seq1 = ['D3', 'F3', 'G3']
            >>> seq2 = ['D3', 'G3']
            >>> aligned1, aligned2, score = needleman_wunsch(seq1, seq2)
            >>> alignment = Alignment.from_needleman_wunsch(
            ...     aligned1, aligned2, score, 'AdTeLevavi/V01', 'AdTeLevavi/V02'
            ... )
            >>> alignment.gap_counts
            [0, 1]
        """
        return cls(
            sequences=[aligned_seq1, aligned_seq2],
            labels=[label1, label2],
            score=score,
            algorithm="Needleman-Wunsch"
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert alignment to pandas DataFrame for analysis and visualization.
        
        Creates a DataFrame where each row represents a position in the alignment
        and each column represents one of the aligned sequences. Additional
        metadata columns include position index and gap indicators.
        
        Returns:
            pandas DataFrame with columns for each sequence plus metadata:
                - 'position': 0-based position index in the alignment
                - One column per sequence (named using the sequence labels)
                - 'has_gap': Boolean indicating if any sequence has a gap at this position
                - 'all_match': Boolean indicating if all non-gap tokens match at this position
        
        Example:
            >>> alignment = Alignment([['D3', None, 'G3'], ['D3', 'F3', 'G3']], 
            ...                      ['seq1', 'seq2'])
            >>> df = alignment.to_dataframe()
            >>> df.columns.tolist()
            ['position', 'seq1', 'seq2', 'has_gap', 'all_match']
        """
        data = {'position': list(range(self.alignment_length))}
        
        # Add columns for each sequence
        for label, sequence in zip(self.labels, self.sequences):
            data[label] = sequence
        
        # Add metadata columns
        has_gap = []
        all_match = []
        
        for i in range(self.alignment_length):
            position_tokens = [seq[i] for seq in self.sequences]
            
            # Check for gaps
            has_gap.append(None in position_tokens)
            
            # Check for matches (only among non-None tokens)
            non_gap_tokens = [token for token in position_tokens if token is not None]
            if len(non_gap_tokens) <= 1:
                all_match.append(True)  # Single token or all gaps = match
            else:
                all_match.append(len(set(non_gap_tokens)) == 1)
        
        data['has_gap'] = has_gap
        data['all_match'] = all_match
        
        return pd.DataFrame(data)