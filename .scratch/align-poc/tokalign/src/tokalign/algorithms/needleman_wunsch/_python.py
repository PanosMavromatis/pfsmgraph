"""Needleman-Wunsch global alignment — pure Python backend (Phase 1).

Mechanical translation of FORMALIZATION.md. Uses the three-matrix affine gap
penalty formulation (M, X, Y) with explicit traceback.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Sequence

from ..._types import Alphabet, AlignmentResult, ScoringMatrix


class Direction(IntEnum):
    """Traceback direction values, matching the formalization's enum."""

    NIL = 0
    DIAG = 1
    UP_OPEN = 2
    UP_EXTEND = 3
    LEFT_OPEN = 4
    LEFT_EXTEND = 5


def align(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    alphabet: Alphabet,
    scoring_matrix: ScoringMatrix,
) -> AlignmentResult:
    """Perform global alignment of two sequences using Needleman-Wunsch.

    Parameters
    ----------
    seq_a : Sequence[str]
        First sequence of string symbols.
    seq_b : Sequence[str]
        Second sequence of string symbols.
    alphabet : Alphabet
        The alphabet defining valid symbols and encoding.
    scoring_matrix : ScoringMatrix
        Scoring matrix and gap penalties for the alignment.

    Returns
    -------
    AlignmentResult
        The optimal global alignment: score, aligned sequences, and traceback.
    """
    # Encode at the boundary
    enc_a, enc_b = alphabet.encode_pair(seq_a, seq_b)

    m = len(enc_a)
    n = len(enc_b)
    g_o = scoring_matrix.gap_open
    g_e = scoring_matrix.gap_extend

    NEG_INF = float("-inf")

    # --- Allocate DP matrices ---
    M = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    X = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    Y = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    T = [[Direction.NIL] * (n + 1) for _ in range(m + 1)]

    # --- Base cases ---
    M[0][0] = 0.0
    # X[0][0] = -inf (already set)
    # Y[0][0] = -inf (already set)

    for i in range(1, m + 1):
        # M[i][0] = -inf (already set)
        X[i][0] = g_o + i * g_e
        # Y[i][0] = -inf (already set)
        T[i][0] = Direction.UP_OPEN

    for j in range(1, n + 1):
        # M[0][j] = -inf (already set)
        # X[0][j] = -inf (already set)
        Y[0][j] = g_o + j * g_e
        T[0][j] = Direction.LEFT_OPEN

    # --- Fill ---
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = scoring_matrix.score(int(enc_a[i - 1]), int(enc_b[j - 1]))

            # Compute M[i, j]
            m_diag = M[i - 1][j - 1] + s
            m_from_x = X[i - 1][j - 1] + s
            m_from_y = Y[i - 1][j - 1] + s
            M[i][j] = max(m_diag, m_from_x, m_from_y)

            # Compute X[i, j]
            x_open = M[i - 1][j] + g_o + g_e
            x_extend = X[i - 1][j] + g_e
            X[i][j] = max(x_open, x_extend)

            # Compute Y[i, j]
            y_open = M[i][j - 1] + g_o + g_e
            y_extend = Y[i][j - 1] + g_e
            Y[i][j] = max(y_open, y_extend)

            # Record traceback direction
            best = max(M[i][j], X[i][j], Y[i][j])
            if best == M[i][j]:
                T[i][j] = Direction.DIAG
            elif best == X[i][j]:
                if x_open >= x_extend:
                    T[i][j] = Direction.UP_OPEN
                else:
                    T[i][j] = Direction.UP_EXTEND
            else:
                if y_open >= y_extend:
                    T[i][j] = Direction.LEFT_OPEN
                else:
                    T[i][j] = Direction.LEFT_EXTEND

    score = max(M[m][n], X[m][n], Y[m][n])

    # --- Traceback ---
    traced_a, traced_b = _nw_traceback(
        M, X, Y, enc_a, enc_b, scoring_matrix, g_o, g_e, alphabet.gap_index
    )

    # Decode at the boundary
    aligned_a = alphabet.decode(traced_a)
    aligned_b = alphabet.decode(traced_b)

    return AlignmentResult(
        score=score,
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        alphabet=alphabet,
        traceback=T,
    )


def _nw_traceback(
    M: list[list[float]],
    X: list[list[float]],
    Y: list[list[float]],
    a: Sequence[int],
    b: Sequence[int],
    scoring_matrix: ScoringMatrix,
    g_o: float,
    g_e: float,
    gap_index: int,
) -> tuple[list[int], list[int]]:
    """Trace back through the DP matrices to recover the aligned sequences.

    Parameters
    ----------
    M, X, Y : list[list[float]]
        The three DP matrices.
    a, b : Sequence[int]
        Encoded integer sequences.
    scoring_matrix : ScoringMatrix
        For looking up match/mismatch scores during traceback.
    g_o, g_e : float
        Gap open and extend penalties.
    gap_index : int
        The integer index representing a gap.

    Returns
    -------
    tuple[list[int], list[int]]
        Aligned integer sequences with gap_index at gap positions.
    """
    m = len(a)
    n = len(b)

    aligned_a: list[int] = []
    aligned_b: list[int] = []

    i = m
    j = n

    # Determine starting matrix from terminal cell
    terminal = max(M[m][n], X[m][n], Y[m][n])
    if terminal == M[m][n]:
        state = "M"
    elif terminal == X[m][n]:
        state = "X"
    else:
        state = "Y"

    while i > 0 or j > 0:
        if state == "M":
            # Diagonal move: both sequences consume a symbol
            aligned_a.append(int(a[i - 1]))
            aligned_b.append(int(b[j - 1]))

            # Determine which matrix contributed to M[i, j]
            s = scoring_matrix.score(int(a[i - 1]), int(b[j - 1]))
            if M[i][j] == M[i - 1][j - 1] + s:
                state = "M"
            elif M[i][j] == X[i - 1][j - 1] + s:
                state = "X"
            else:
                state = "Y"
            i -= 1
            j -= 1

        elif state == "X":
            # Vertical move: gap in sequence B
            aligned_a.append(int(a[i - 1]))
            aligned_b.append(gap_index)

            # Determine if gap was opened or extended
            if X[i][j] == M[i - 1][j] + g_o + g_e:
                state = "M"
            else:
                state = "X"
            i -= 1

        else:  # state == "Y"
            # Horizontal move: gap in sequence A
            aligned_a.append(gap_index)
            aligned_b.append(int(b[j - 1]))

            # Determine if gap was opened or extended
            if Y[i][j] == M[i][j - 1] + g_o + g_e:
                state = "M"
            else:
                state = "Y"
            j -= 1

    # Reverse since we built from the end
    aligned_a.reverse()
    aligned_b.reverse()

    return aligned_a, aligned_b
