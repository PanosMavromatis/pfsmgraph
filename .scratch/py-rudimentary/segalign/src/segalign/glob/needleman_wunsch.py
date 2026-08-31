import numpy as np
from typing import List, Callable, Tuple, Any

def needleman_wunsch(seq1: List[Any], seq2: List[Any], 
                      match_score: float = 1.0, 
                      mismatch_score: float = -0.5, 
                      gap_penalty: float = -1.0,
                      similarity_fn: Callable = None) -> Tuple[List[Any], List[Any], float]:
    """
    Global alignment of two musical token sequences using the Needleman-Wunsch algorithm.
    
    Implements the classic dynamic programming algorithm from Needleman & Wunsch (1970)
    adapted for symbolic music analysis. The algorithm finds the optimal global alignment
    between two sequences of musical tokens (pitches, boundaries, text, etc.) by maximizing
    the cumulative similarity score while accounting for gaps and mismatches.
    
    This implementation uses None to represent gaps in the aligned sequences, making it
    suitable for downstream analysis and visualization of musical sequence relationships.
    
    Args:
        seq1: First sequence of musical tokens to align (e.g., ['D3', 'F3', 'G3', 'Maior'])
        seq2: Second sequence of musical tokens to align (e.g., ['C3', 'F3', 'A3', 'Finalis'])
        match_score: Score awarded for matching tokens (default: 1.0)
        mismatch_score: Score penalty for mismatched tokens (default: -0.5)
        gap_penalty: Score penalty for introducing gaps (default: -1.0)
        similarity_fn: Optional custom similarity function taking two tokens and returning
                      a float score. If None, uses exact token matching with match_score
                      for matches and mismatch_score for mismatches.
    
    Returns:
        Tuple containing:
            - aligned_seq1: First sequence with gaps (None) inserted for optimal alignment
            - aligned_seq2: Second sequence with gaps (None) inserted for optimal alignment  
            - alignment_score: Total score of the optimal alignment
    
    Example:
        >>> seq1 = ['D3', 'F3', 'G3']
        >>> seq2 = ['D3', 'G3'] 
        >>> aligned1, aligned2, score = needleman_wunsch(seq1, seq2)
        >>> print(aligned1)  # ['D3', 'F3', 'G3']
        >>> print(aligned2)  # ['D3', None, 'G3']
        >>> print(score)     # 1.0 (match) + (-1.0) (gap) + 1.0 (match) = 1.0
    
    References:
        Needleman, S. B. & Wunsch, C. D. (1970). A general method applicable to the 
        search for similarities in the amino acid sequence of two proteins. 
        Journal of Molecular Biology, 48(3), 443-453.
    """
    # If no similarity function is provided, use the default scoring
    if similarity_fn is None:
        def similarity_fn(a, b):
            return match_score if a == b else mismatch_score
    
    # Initialize the scoring matrix with zeros
    # As described in Needleman & Wunsch, we create a matrix to store
    # the maximum similarity scores between all possible sub-sequences
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))
    
    # Fill in the first row and column with gap penalties
    # This corresponds to aligning with all gaps
    for i in range(m + 1):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(n + 1):
        score_matrix[0, j] = j * gap_penalty
        
    # Fill the scoring matrix using the recurrence relation
    # This is the core of the Needleman-Wunsch algorithm as described in the paper:
    # "The maximum match is obtained by adding one to the best previous score for 
    # xi and yj being a match, or by selecting the best previous score with xi or yj 
    # not paired with the other."
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the similarity score for this pair of elements
            match = score_matrix[i-1, j-1] + similarity_fn(seq1[i-1], seq2[j-1])
            
            # Consider introducing gaps in either sequence
            delete = score_matrix[i-1, j] + gap_penalty  # Gap in seq2
            insert = score_matrix[i, j-1] + gap_penalty  # Gap in seq1
            
            # Take the maximum of the three options
            score_matrix[i, j] = max(match, delete, insert)
    
    # Traceback to find the optimal alignment
    # Starting from the bottom right of the matrix (the complete alignment)
    # and working backwards to the origin
    aligned_seq1, aligned_seq2 = [], []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i-1, j-1] + similarity_fn(seq1[i-1], seq2[j-1]):
            # Diagonal move (match/mismatch)
            aligned_seq1.insert(0, seq1[i-1])
            aligned_seq2.insert(0, seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_penalty:
            # Vertical move (gap in seq2)
            aligned_seq1.insert(0, seq1[i-1])
            aligned_seq2.insert(0, None)  # Use None to represent a gap
            i -= 1
        else:
            # Horizontal move (gap in seq1)
            aligned_seq1.insert(0, None)  # Use None to represent a gap
            aligned_seq2.insert(0, seq2[j-1])
            j -= 1
    
    # Return the aligned sequences and the alignment score
    return aligned_seq1, aligned_seq2, score_matrix[m, n]

# --- Different versions of Needleman-Wunsch, to be compared with the optimized version
# --- in benchmark studies, and to guarantee that the optimized versions behave exactly
# --- like the direct implementation

def needleman_wunsch_0(seq1: List[Any], seq2: List[Any], 
                      match_score: float = 1.0, 
                      mismatch_score: float = -0.5, 
                      gap_penalty: float = -1.0,
                      similarity_fn: Callable = None) -> Tuple[List[Any], List[Any], float]:
    """
    Direct implementation of the Needleman-Wunsch algorithm.
    """
    # If no similarity function is provided, use the default scoring
    if similarity_fn is None:
        def similarity_fn(a, b):
            return match_score if a == b else mismatch_score
    
    # Initialize the scoring matrix with zeros
    # As described in Needleman & Wunsch, we create a matrix to store
    # the maximum similarity scores between all possible sub-sequences
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))
    
    # Fill in the first row and column with gap penalties
    # This corresponds to aligning with all gaps
    for i in range(m + 1):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(n + 1):
        score_matrix[0, j] = j * gap_penalty
        
    # Fill the scoring matrix using the recurrence relation
    # This is the core of the Needleman-Wunsch algorithm as described in the paper:
    # "The maximum match is obtained by adding one to the best previous score for 
    # xi and yj being a match, or by selecting the best previous score with xi or yj 
    # not paired with the other."
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the similarity score for this pair of elements
            match = score_matrix[i-1, j-1] + similarity_fn(seq1[i-1], seq2[j-1])
            
            # Consider introducing gaps in either sequence
            delete = score_matrix[i-1, j] + gap_penalty  # Gap in seq2
            insert = score_matrix[i, j-1] + gap_penalty  # Gap in seq1
            
            # Take the maximum of the three options
            score_matrix[i, j] = max(match, delete, insert)
    
    # Traceback to find the optimal alignment
    # Starting from the bottom right of the matrix (the complete alignment)
    # and working backwards to the origin
    aligned_seq1, aligned_seq2 = [], []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i-1, j-1] + similarity_fn(seq1[i-1], seq2[j-1]):
            # Diagonal move (match/mismatch)
            aligned_seq1.insert(0, seq1[i-1])
            aligned_seq2.insert(0, seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_penalty:
            # Vertical move (gap in seq2)
            aligned_seq1.insert(0, seq1[i-1])
            aligned_seq2.insert(0, None)  # Use None to represent a gap
            i -= 1
        else:
            # Horizontal move (gap in seq1)
            aligned_seq1.insert(0, None)  # Use None to represent a gap
            aligned_seq2.insert(0, seq2[j-1])
            j -= 1
    
    # Return the aligned sequences and the alignment score
    return aligned_seq1, aligned_seq2, score_matrix[m, n]
