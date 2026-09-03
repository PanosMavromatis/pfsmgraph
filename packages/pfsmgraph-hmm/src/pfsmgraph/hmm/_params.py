"""The parameters of an arc-emission HMM, as a single frozen value.

[ADR 0017] decides that this package represents model parameters as one immutable
value rather than as a mutable model object with a working copy beside it, and
four properties of that sentence are part of the decision rather than
consequences of it. Three of them are visible here:

* **The buffers are read-only, not merely the bindings.** ``frozen=True`` stops
  an attribute being rebound and does nothing whatever about writing through the
  array reference it holds, so every array is marked ``writeable = False``. The
  omission raises nothing and warns about nothing, which is why it is stated in
  the ADR's Decision rather than left to a docstring.
* **Derived quantities are computed, never stored.** :attr:`state_p`,
  :attr:`state_entropies` and :attr:`entropy` are cached properties. In the Lush
  original they are slots kept correct by a manual ``update-entropy`` call after
  every mutation, and a path that mutates and forgets leaves a model whose
  entropy describes parameters it no longer has.
* **The vocabulary is held with the arrays.** ``output_p``'s third axis is
  indexed by symbol code, and a code means nothing without the table that
  assigned it: two different 25-symbol vocabularies produce parameter sets of
  identical shape and incompatible meaning.

The fourth -- algorithms take parameters rather than owning them -- is what keeps
this module free of a ``viterbi`` method.

The formulation is arc-emission (Mealy) per [ADR 0015]: a symbol is emitted while
*crossing* a transition, so the emission parameter is ``output_p[i, j, symbol]``,
indexed by source state, destination state and symbol, and never
``B[state, symbol]``. Every textbook and every library is the other one.

.. [ADR 0015] ``docs/design/adr/0015-arc-emission-mealy-formulation.md``
.. [ADR 0017] ``docs/design/adr/0017-frozen-parameter-object-for-hmm.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pfsmgraph.dataseq import RESERVED_SYMBOLS, USER_BASE, Vocabulary

from ._numeric import stationary_distribution

# Aliased on import because this module also exposes a property named ``entropy``
# -- the model entropy -- and the two would read as one name at a call site.
from ._numeric import entropy as _shannon_entropy

__all__ = ["HMMParams"]

#: How far a probability vector's sum may stand from 1 at construction.
#:
#: Chosen to sit between two known magnitudes rather than picked round.
#:
#: *Below* it: a vector normalized in float32 and widened to float64 here, whose
#: sum drifts by order ``sqrt(A) * eps32`` -- about ``1e-6`` over a symbol axis of
#: a hundred, and worse under adversarial rounding. That consumer is not
#: hypothetical: ADR 0017 anticipates revision 03's ``torch`` backend building
#: transient parameters and returning a new frozen value, and torch is float32 by
#: default. An earlier ``1e-6`` here would have rejected its output.
#:
#: *Above* it: the ``1e-4`` drift of a four-decimal ASCII print, which is how the
#: tracked ``.hmm`` fixtures store their matrices. That drift must be *repaired*
#: by renormalizing the rows -- which restores the hypothesis -- rather than
#: tolerated by widening this bound, which would only loosen the conclusion. See
#: the fixture note in ``docs/agents/core.md``.
SUM_TOL = 1e-5


def _frozen(values: object, name: str, ndim: int) -> np.ndarray:
    """Validate one parameter array structurally, then return it read-only."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}-D, got {array.ndim}-D with shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains a non-finite value; probabilities must be finite "
            f"(the +inf that means 'impossible' belongs to the bit domain, not here)"
        )
    if np.any(array < 0.0):
        raise ValueError(
            f"{name} contains a negative value, which is not a probability; "
            f"smallest is {float(array.min())!r}"
        )
    # Copy before freezing: asarray may have returned the caller's own array, and
    # making that read-only would mutate an object we do not own. Same reasoning
    # as ``dataseq``'s SequenceRecord.
    array = array.copy()
    array.setflags(write=False)
    return array


def _frozen_result(array: np.ndarray) -> np.ndarray:
    """Freeze a derived array on its way out of a cached property.

    Freezing the *inputs* alone does not deliver ADR 0017's claim that a whole
    class of staleness bug becomes unrepresentable. A cached property hands back
    the same object on every access, so a caller who writes into the array
    returned by :attr:`state_p` corrupts the value every later reader sees --
    which is the stored-and-stale failure the ADR removes, reintroduced one level
    out.
    """
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class HMMParams:
    """The three arc-emission arrays and the vocabulary that gives them meaning.

    :param init_state_p: ``(S,)``, the initial state distribution. Sums to 1.
    :param transition_p: ``(S, S)``, indexed by source and destination state.
        Every row sums to 1.
    :param output_p: ``(S, S, A)``, indexed by source state, destination state
        and symbol code, with ``A == vocabulary.size``. Every fibre
        ``output_p[i, j, :]`` on an arc that carries probability sums to 1.
    :param vocabulary: the mapping ``output_p``'s symbol axis was sized against,
        taken as ``dataseq``'s ``Vocabulary`` Protocol rather than the concrete
        ``SymbolTable``, matching how ``SequenceDataset`` types its own. Only
        ``size`` is read; no symbol string is ever looked at here.

    **The symbol axis spans the whole vocabulary, reserved block included, and
    the reserved fibres must be exactly zero.** The alternative -- sizing the
    axis to the user symbols alone and subtracting ``USER_BASE`` at every index
    -- fails silently on a documented ``dataseq`` path: ``encode`` with
    ``on_unknown="unk"`` puts ``UNK`` (code 1) into a record, and ``1 -
    USER_BASE`` is ``-5``, a negative index numpy accepts without complaint. The
    decode would then return a confident path built from some other symbol's
    emission probabilities. Sized to the whole vocabulary, the same record
    reaches ``output_p[i, j, 1]``, which is zero, and ``bits(0)`` is ``+inf``,
    which absorbs under addition -- so the path is reported impossible instead of
    wrong. The cost is ``6 * S**2`` dead entries, 120 KB at ``S = 50``, scaling
    with the state count rather than with the corpus.

    Equality is identity. Array-valued equality has no single right answer --
    exact or within a tolerance, and if a tolerance then whose -- and nothing in
    0.1.0 compares two parameter sets, so the question is left open rather than
    guessed at; adding ``__eq__`` later is not a breaking change.
    """

    init_state_p: np.ndarray
    transition_p: np.ndarray
    output_p: np.ndarray
    vocabulary: Vocabulary

    def __post_init__(self) -> None:
        init_state_p = _frozen(self.init_state_p, "init_state_p", 1)
        transition_p = _frozen(self.transition_p, "transition_p", 2)
        output_p = _frozen(self.output_p, "output_p", 3)

        size = init_state_p.shape[0]
        if size < 1:
            raise ValueError("init_state_p must name at least one state, got a size of 0")
        if transition_p.shape != (size, size):
            raise ValueError(
                f"transition_p must be ({size}, {size}) to match init_state_p, "
                f"got {transition_p.shape}"
            )
        if output_p.shape[:2] != (size, size):
            raise ValueError(
                f"output_p's first two axes must be ({size}, {size}) -- source and "
                f"destination state, since emission is on the arc (ADR 0015) -- "
                f"got {output_p.shape[:2]} from a full shape of {output_p.shape}"
            )

        n_symbols = output_p.shape[2]
        vocabulary_size = self.vocabulary.size
        if n_symbols != vocabulary_size:
            raise ValueError(
                f"output_p's symbol axis is {n_symbols} but the vocabulary has "
                f"{vocabulary_size} codes. The axis spans the whole vocabulary, "
                f"reserved block included, so a code indexes it directly with no "
                f"offset; sizing it to the {vocabulary_size - USER_BASE} user "
                f"symbols instead would make an UNK-bearing record index backwards "
                f"without error"
            )

        self._check_distributions(init_state_p, transition_p, output_p)

        object.__setattr__(self, "init_state_p", init_state_p)
        object.__setattr__(self, "transition_p", transition_p)
        object.__setattr__(self, "output_p", output_p)

    @staticmethod
    def _check_distributions(
        init_state_p: np.ndarray, transition_p: np.ndarray, output_p: np.ndarray
    ) -> None:
        """Every axis that must be a probability distribution, in index order."""
        total = float(init_state_p.sum())
        if abs(total - 1.0) > SUM_TOL:
            raise ValueError(
                f"init_state_p sums to {total!r}, not 1 (tolerance {SUM_TOL})"
            )

        row_sums = transition_p.sum(axis=1)
        bad_rows = np.flatnonzero(np.abs(row_sums - 1.0) > SUM_TOL)
        if bad_rows.size:
            # A row of zeros reaches here too, and that is deliberate rather than
            # an oversight to be exempted. HMMLIB-ACCOUNT.md section 5 records that
            # merge-states divides with safe-/, so merging two unreachable states
            # yields zeros -- meaning revision 04's topology search can construct
            # exactly this. Rejecting it makes that surface at construction, where
            # revision 04 has to decide what an unreachable state means, rather
            # than silently downstream in a stationary solve that has none.
            first = int(bad_rows[0])
            raise ValueError(
                f"{bad_rows.size} row(s) of transition_p do not sum to 1: row "
                f"{first} sums to {float(row_sums[first])!r}. Rows are "
                f"{bad_rows.tolist()}"
            )

        reserved = output_p[:, :, :USER_BASE]
        offenders = np.argwhere(reserved != 0.0)
        if offenders.size:
            i, j, code = (int(x) for x in offenders[0])
            raise ValueError(
                f"output_p assigns {float(output_p[i, j, code])!r} to the reserved "
                f"symbol {RESERVED_SYMBOLS[code]} (code {code}) on the arc "
                f"{i} -> {j}, and {len(offenders)} reserved entries are non-zero. "
                f"The reserved block must be exactly zero: it is what makes "
                f"emitting PAD/UNK/BOS/EOS/GAP/MSK impossible by arithmetic -- "
                f"bits(0) is +inf -- rather than by convention"
            )

        # Only arcs that carry probability. A fibre on a dead arc cannot reach the
        # recurrence at all, because bits(transition_p[i, j] == 0) is +inf and
        # absorbs under addition, so validating it would check something that
        # cannot change an answer.
        fibre_sums = output_p.sum(axis=2)
        live = transition_p > 0.0
        bad_arcs = np.argwhere(live & (np.abs(fibre_sums - 1.0) > SUM_TOL))
        if bad_arcs.size:
            i, j = (int(x) for x in bad_arcs[0])
            raise ValueError(
                f"output_p[{i}, {j}] sums to {float(fibre_sums[i, j])!r}, not 1, on "
                f"an arc carrying transition_p[{i}, {j}] = "
                f"{float(transition_p[i, j])!r}. {len(bad_arcs)} such arc(s); "
                f"fibres on arcs of probability 0 are not checked"
            )

    @property
    def n_states(self) -> int:
        """``S``. The number of states."""
        return int(self.init_state_p.shape[0])

    @property
    def n_symbols(self) -> int:
        """``A``. Derived from ``output_p``, never stored beside it (ADR 0017)."""
        return int(self.output_p.shape[2])

    @cached_property
    def state_p(self) -> np.ndarray:
        """``(S,)``. The stationary distribution of :attr:`transition_p`.

        The original's ``state-p`` slot, computed rather than stored. Raises
        ``ValueError`` for a reducible chain -- two closed communicating classes
        give ``(P.T - I)`` a null space of dimension 2, which replacing one row
        with the normalization cannot determine. That failure therefore surfaces
        on an attribute access, which is a consequence of ADR 0017 making this
        derived and is why the message names the cause.
        """
        return _frozen_result(stationary_distribution(self.transition_p))

    @cached_property
    def state_entropies(self) -> np.ndarray:
        """``(S,)``. Shannon entropy in bits of each state's emission marginal.

        The marginal is ``p_i(k) = sum_j transition_p[i, j] * output_p[i, j, k]``
        -- the distribution over symbols emitted when leaving state ``i``,
        marginalizing over which arc is taken. It is model-shaped, which is why
        the einsum lives here rather than in ``_numeric``.
        """
        marginal = np.einsum("ij,ijk->ik", self.transition_p, self.output_p)
        return _frozen_result(_shannon_entropy(marginal))

    @cached_property
    def entropy(self) -> float:
        """The model entropy: :attr:`state_entropies` weighted by :attr:`state_p`."""
        return float(self.state_p @ self.state_entropies)

    def __repr__(self) -> str:
        return (
            f"HMMParams(n_states={self.n_states}, n_symbols={self.n_symbols})"
        )
