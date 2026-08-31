"""
Sequence module for musical token sequence processing and analysis.

This module provides classes and functions for handling symbolic music
representations as string-encoded sequences, supporting computational
music theory research and analysis workflows.
"""

from .dataset import Dataset
from .alignment import Alignment

__version__ = "0.1.0"
__all__ = ["Dataset", "Alignment"]