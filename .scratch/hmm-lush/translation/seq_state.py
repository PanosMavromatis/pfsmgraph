"""``seq-state``: a sequence together with the model's reading of it.

This is the design idea from the original most worth carrying forward, and it is
an *annotation* idea rather than a container one. Three parallel arrays, and the
off-by-one between them is exact rather than defensive: a path through an HMM
that emits N symbols visits N+1 states, so ``path_states`` and ``path_entropy``
are indexed by the gaps *between* symbols while ``symbol_data`` is indexed by the
symbols themselves.

``hmm`` consumes exactly this object (``hmm-trainer.lsh:66``), so it is the seam
the model implementation will have to be modified against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SeqState"]


@dataclass
class SeqState:
    """One symbol stream plus a state path over it.

    DEVIATION: the original's ``alphabet`` slot is an array of raw C pointers
    (``-idx1- (-gptr-)``) shared by aliasing -- ``set-alphabet`` stores the
    caller's array, so freeing or reloading the source invalidates the derived
    state. Here it is an ordinary list, and Python's own reference semantics make
    the sharing explicit rather than incidental. The pointer array existed only
    so the compiled methods could carry it across the DH boundary; see ACCOUNT.md
    section 7.
    """

    name: str
    size: int
    symbol_data: list[int] = field(default_factory=list)
    path_states: list[int] = field(default_factory=list)
    path_entropy: list[float] = field(default_factory=list)
    alphabet: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.symbol_data:
            self.symbol_data = [0] * self.size
        if not self.path_states:
            self.path_states = [0] * (self.size + 1)
        if not self.path_entropy:
            self.path_entropy = [0.0] * (self.size + 1)

    def resize(self, new_size: int) -> None:
        """Grow or shrink all three arrays consistently, keeping the +1 offset."""
        self.size = new_size
        _resize(self.symbol_data, new_size, 0)
        _resize(self.path_states, new_size + 1, 0)
        _resize(self.path_entropy, new_size + 1, 0.0)

    def set_alphabet(self, alphabet: list[str]) -> None:
        self.alphabet = alphabet

    def decode(self) -> list[str]:
        """Symbol names for the whole stream.

        The original has no decode method; rendering goes through an inline
        ``(ptr-str (alphabet (symbol-data pos-i)))`` inside ``view-string``.
        Naming the operation is the smallest useful thing the translation adds.
        """
        return [self.alphabet[code] for code in self.symbol_data]

    def view_string(self) -> str:
        """Two interleaved lines: symbols above, bracketed states below.

        A faithful rendering of ``view-string``, which is the clearest statement
        in the original of what the +1 offset is for::

                a    b    a
             [0]  [2]  [1]  [0]
        """
        symbols, states = [], []
        head = f" [{self.path_states[0]}]"
        states.append(head)
        symbols.append(" " * len(head))
        for pos, code in enumerate(self.symbol_data):
            cell = f" {self.alphabet[code] if self.alphabet else code}"
            symbols.append(cell)
            states.append(" " * len(cell))
            cell = f" [{self.path_states[pos + 1]}]"
            states.append(cell)
            symbols.append(" " * len(cell))
        return f"\n{''.join(symbols)} \n\n{''.join(states)} \n"


def _resize(target: list, length: int, fill) -> None:
    del target[length:]
    target.extend([fill] * (length - len(target)))
