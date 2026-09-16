"""The minimum description length criterion the topology search scores moves with.

Private to ``pfsmgraph.hmm``: nothing here is re-exported from the package's
``__init__``, and nothing outside this distribution may import it. Promotion to
a shared home is reconsidered if ``hseg`` ever scores segmentations the same
way; inventing a sixth distribution before a second consumer exists is not
warranted (master plan, revision ``04-hmm-v0.3.0``).

The two code-length primitives are ``int-code-length`` and ``comb-code-length``
(``.scratch/hmm-lush/Code/Utility/util.lsh:463-486``), read together with the DH
compiler's generated C (``Code/Utility/C/util.c:637-715``). The C is what the
reading is checked against: there is no Lush runtime anywhere in this
repository, so the generated code is the only machine-checked statement of what
these compute.

**The original accumulates in single precision, and this does not.** The
generated C declares ``flt`` locals -- Lush's single-precision type -- and casts
every ``C_log2`` result back to ``flt``, widening to ``real`` only at the return.
That both types are distinct is visible in the tracked C, which declares ``real``
and ``flt`` locals in the same file; that ``flt`` is float32 is Lush's
convention rather than anything tracked here. Measured, the difference does not
reach the oracle: a whole 8-state model description length moves 2.1e-4 bits
between float32 and float64 accumulation, two orders below the 0.009 bits the
data half already differs from the original's own logged value.

**Negative input raises here; the original returned 0.0.** Both primitives guard
their domain by returning zero, which maps nonsense input to a *cheap* score --
the same failure class as the ``1e100`` sentinel this revision declined, a wrong
answer that is also comparable, so a search would silently prefer the broken
model. ``m <= 1`` still returns ``0.0``, because that one is not a guard: a
composition into a single part carries no information, and a one-state model is
where the search actually starts.
"""

from __future__ import annotations

import numpy as np

__all__: list[str] = []

#: Rissanen's normalising constant, ``log2`` of which opens every integer code.
#: The original spells the ``2.865`` literal inline (``util.lsh:468``).
RISSANEN_C = 2.865


def _int_code_length(n):
    """Rissanen's universal prior for the integers, in bits.

    ``int-code-length`` (``util.lsh:463-473``): ``log2(2.865)`` plus the
    iterated logarithm ``log2 n + log2 log2 n + ...``.

    **The original codes ``n + 1``, not ``n``.** The increment at
    ``util.lsh:467`` has no counterpart in the usual statement of the prior, and
    it is what makes ``_int_code_length(0)`` finite -- ``log2(2.865)`` alone --
    rather than an iterated logarithm of zero. That matters because the model
    description length calls this on a state count, and the search starts from
    one state.

    **The termination rule is "while the term is positive", not "while the term
    exceeds 1".** The guard tests the value whose logarithm is about to be
    taken, so a term is added exactly when the previous value exceeded 1, which
    is exactly when the new term is above zero. The last term added is therefore
    often below 1 and never below 0. ``HMMLIB-ACCOUNT.md`` §8 states this as
    "while the term exceeds 1", which describes a different loop; the generated
    C (``util.c:645-661``) settles it.

    Raises ``ValueError`` for a negative ``n``, where the original returned
    ``0.0`` -- see the module docstring. ``n`` is a float in the original and
    stays one here, because ``update-model-dl`` calls this on the quantization
    resolution ``d`` as well as on a state count.
    """
    n = float(n)
    if not (n >= 0.0):
        raise ValueError(
            f"n must be non-negative, got {n}. The original returned 0.0 here, "
            f"which would make an invalid model cheaper to describe than a "
            f"valid one rather than reporting the error"
        )
    n += 1.0
    total = float(np.log2(RISSANEN_C))
    while n > 1.0:
        n = float(np.log2(n))
        total += n
    return total


def _comb_code_length(total, m):
    """Bits to code a composition of ``total`` into ``m`` non-negative parts.

    ``comb-code-length`` (``util.lsh:475-486``), which evaluates to
    ``log2(total + m) + log2(C(total + m - 1, m - 1))``. Called with
    ``total = d``, it is the cost of one probability vector quantized to
    ``1 / d`` -- which is why ``d`` appears both as the rounding grid and as an
    argument here.

    **The binomial is never formed.** ``C(total + m - 1, m - 1)`` overflows a
    float long before it overflows an int, so the original sums
    ``log2(total + i) - log2(i)`` term by term and this does the same. The terms
    are differenced elementwise and then reduced by one ascending
    :func:`numpy.add.accumulate`, the reduction this package uses wherever the
    order is contract; the original interleaves its addition and subtraction per
    ``i`` instead, so this is **not** bit-identical to it. Measured across
    ``m`` in ``{2, 6, 9, 13, 32, 51}`` at ``d = 10000`` the two agree to 1e-13
    bits, and a ``lgamma`` closed form agrees with both to 4e-11, so a later
    substitution is free if the cost ever matters. It does not today: the value
    depends only on ``(total, m)``, which is constant across arcs, so the model
    description length calls this twice per score however many states there are.

    ``m <= 1`` is ``0.0``: a composition into one part carries no information.
    A negative ``total`` raises -- see the module docstring.
    """
    total = float(total)
    m = int(m)
    if not (total >= 0.0):
        raise ValueError(
            f"total must be non-negative, got {total}. The original returned "
            f"0.0 here, which would make an invalid model cheaper to describe "
            f"than a valid one rather than reporting the error"
        )
    if m <= 1:
        return 0.0
    i = np.arange(1, m, dtype=np.float64)
    terms = np.log2(total + i) - np.log2(i)
    return float(np.log2(total + m) + np.add.accumulate(terms)[-1])
