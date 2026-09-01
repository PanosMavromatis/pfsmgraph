from . import seq
from . import glob
from . import data


def hello() -> str:
    return "Hello from segalign!"


__all__ = ["seq", "glob", "data", "hello"]
