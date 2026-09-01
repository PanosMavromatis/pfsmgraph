"""Backend discovery for tokalign algorithms.

Centralises the logic for detecting which backends (Python, Cython, Numba)
are available for a given algorithm. Numba backends require both the module
and a CUDA-capable GPU to be considered available.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Sequence

_BACKEND_NAMES = ("python", "cython", "numba")
_BACKEND_MODULES = {
    "python": "_python",
    "cython": "_cython",
    "numba": "_numba",
}


def is_gpu_available() -> bool:
    """Check whether a CUDA-capable GPU is available via Numba."""
    try:
        from numba import cuda
        return cuda.is_available()
    except ImportError:
        return False


def get_available_backends(
    algorithm_name: str,
) -> list[tuple[str, Callable[..., Any]]]:
    """Discover which backends are installed and usable for a given algorithm.

    Parameters
    ----------
    algorithm_name : str
        Algorithm name in snake_case (e.g., ``"needleman_wunsch"``).

    Returns
    -------
    list[tuple[str, Callable]]
        List of ``(backend_name, align_fn)`` pairs for all available backends,
        in order: python, cython, numba.

    Notes
    -----
    - The Python backend is always expected to be present; an ``ImportError``
      here propagates (it indicates a broken installation).
    - The Cython backend is included if its compiled module can be imported.
    - The Numba backend is included only if both the module can be imported
      **and** a CUDA-capable GPU is detected via ``numba.cuda.is_available()``.
    """
    backends: list[tuple[str, Callable[..., Any]]] = []

    # Python — always present (ImportError means broken install, let it propagate)
    mod = importlib.import_module(
        f"tokalign.algorithms.{algorithm_name}.{_BACKEND_MODULES['python']}"
    )
    backends.append(("python", mod.align))

    # Cython — optional, requires compiled .so
    try:
        mod = importlib.import_module(
            f"tokalign.algorithms.{algorithm_name}.{_BACKEND_MODULES['cython']}"
        )
        backends.append(("cython", mod.align))
    except ImportError:
        pass

    # Numba — optional, requires both module and CUDA device
    try:
        mod = importlib.import_module(
            f"tokalign.algorithms.{algorithm_name}.{_BACKEND_MODULES['numba']}"
        )
        if is_gpu_available():
            backends.append(("numba", mod.align))
    except ImportError:
        pass

    return backends
