"""
Global alignment module for musical sequence alignment algorithms.

This module implements global alignment algorithms adapted from computational
biology for symbolic music analysis, including SS-2, Smith-Waterman variants,
and T-Coffee-inspired approaches for musical token sequences.
"""

from .needleman_wunsch import needleman_wunsch

__version__ = "0.1.0"
__all__ = ["needleman_wunsch"]