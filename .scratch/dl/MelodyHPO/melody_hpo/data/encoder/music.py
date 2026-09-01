"""Abstract base class for symbolic music encoders.

Defines the interface that all music encoders must implement: a ``range``
class attribute bounding the valid code space, and static ``encode`` /
``decode`` methods for converting between string tokens and integer codes.
"""

from abc import ABC, abstractmethod


class MusicCode(ABC):
    """Abstract base class for symbolic music encoders/decoders."""

    range: dict[str, int]

    @staticmethod
    @abstractmethod
    def encode(token: str) -> int: ...

    @staticmethod
    @abstractmethod
    def decode(code: int) -> str: ...
