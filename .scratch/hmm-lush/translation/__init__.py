"""A working Python rendering of the Lush ``SeqData`` implementation.

Subgoal 3 of goal 3, ``docs/plan/feat-dataseq-merge/TODO.md``. This is a reading
aid for the merge, not shipped code, and nothing outside ``.scratch/`` may import
it. The whole directory is deleted by goal 8.

It is **not a transliteration.** Per the decision logged in the plan, it
reproduces the original's *implementation decisions* in idiomatic Python rather
than its Lush constructs -- a literal rendering would preserve accidents of a
list language and bury the choices we are actually here to evaluate. Every place
the two diverge is marked ``DEVIATION`` with the reason.

The account of the original is ``../ACCOUNT.md``; read that first. The comparison
against the ``dl`` base is ``../COMPARISON.md``.

Run it against the tracked corpora to reproduce the account's appendix::

    python3 -m translation
"""

from .lush_reader import RawDataSyntaxError, read_raw_data
from .seq_state import SeqState
from .dsource_seq import BEGIN, END, FIRST_USER_CODE, DsourceSeq
from .format_sds import format_sds

__all__ = [
    "BEGIN",
    "END",
    "FIRST_USER_CODE",
    "DsourceSeq",
    "RawDataSyntaxError",
    "SeqState",
    "format_sds",
    "read_raw_data",
]
