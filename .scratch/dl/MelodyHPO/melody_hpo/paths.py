"""Project and data root path discovery.

Resolves the project root by walking up from this file to find the directory
containing ``pyproject.toml``. The data root defaults to ``<project_root>/data``
and can be overridden with the ``DATA_ROOT`` environment variable.
"""

import os
from pathlib import Path

# Walk up from this file's directory to locate pyproject.toml
_pkg_dir = Path(__file__).resolve().parent
_candidate = _pkg_dir
while not (_candidate / "pyproject.toml").exists():
    _parent = _candidate.parent
    if _parent == _candidate:
        raise FileNotFoundError("Could not find pyproject.toml in any parent directory")
    _candidate = _parent

PROJECT_ROOT: Path = _candidate
"""Absolute path to the project root (directory containing ``pyproject.toml``)."""

DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
"""Absolute path to the data directory. Override with the ``DATA_ROOT`` env var."""
