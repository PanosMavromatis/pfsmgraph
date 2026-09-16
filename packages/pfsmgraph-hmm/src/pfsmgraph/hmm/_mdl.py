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

from pfsmgraph.dataseq import USER_BASE

from ._forward_backward import _description_length, _forward_backward
from ._numeric import safe_divide
from ._params import _check_codes

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


def _quantize(p, d):
    """Round each probability to a multiple of ``1 / d`` and renormalise the last axis.

    ``update-approx-init-state-p``, ``-transition-p`` and ``-output-p``
    (``hmm-trainer.lsh:269-341``) in one function: ``round-using``
    (``util.lsh:59-62``) is ``int(x * d + 0.5) / d``, which for the non-negative
    values here is ``floor``, so an exact half rounds up. Each vector is then
    divided by its own sum, taken ascending, through ``safe_divide``.

    The result is arrays, never an ``HMMParams``, and nothing is checked: a vector
    whose every entry is below ``1 / (2 d)`` rounds to all zeros and stays zero,
    which is how coarse precision can make a record impossible. ``d`` need not be
    an integer, as in the original.
    """
    p = np.asarray(p, dtype=np.float64)
    rounded = np.floor(p * d + 0.5) / d
    total = np.add.accumulate(rounded, axis=-1)[..., -1:]
    return safe_divide(rounded, total)


def _corpus_description_length(init_state_p, transition_p, output_p, records):
    """Bits over every record, from arrays rather than an ``HMMParams``.

    Arrays because :func:`_data_description_length` passes rounded parameters,
    which need not be a valid model: a row can round to all zeros.
    """
    bits_per_record = np.zeros(len(records), dtype=np.float64)
    for index, record in enumerate(records):
        _, _, scale = _forward_backward(
            init_state_p, transition_p, output_p, record.codes
        )
        bits_per_record[index] = _description_length(scale)
    return float(np.add.accumulate(bits_per_record)[-1]) if len(records) else 0.0


def _data_description_length(params, records, d):
    """The data description length at precision ``d``: ``update-data-dl``.

    The original runs its forward pass again over the rounded ``-r`` matrices
    (``hmm-trainer.lsh:346-402``); here the same corpus description length the EM
    loop uses is called on :func:`_quantize`'s arrays, so there is one forward
    pass, not two. It is the data half of revision 04's two-part score, where
    ``d`` is chosen against the model half; this function takes ``d`` as given.

    ``+inf`` when rounding leaves some record with no path, which the original
    reports as ``1e100`` through its ``-1`` sentinel (``update-total-dl``). The
    original's final ``bits`` of the last column's sum is 1 within rounding and is
    omitted, as it is in :func:`baum_welch`. Raises ``ValueError`` for a ``d`` that is
    not positive and finite, and for a code outside the symbol axis.
    """
    if not (np.isfinite(d) and d > 0):
        raise ValueError(f"d must be positive and finite, got {d}")
    records = list(records)
    _check_codes(params, records)
    return _corpus_description_length(
        _quantize(params.init_state_p, d),
        _quantize(params.transition_p, d),
        _quantize(params.output_p, d),
        records,
    )


def _model_description_length(params, d):
    """The model half of the two-part score at precision ``d``: ``update-model-dl``.

    ``hmm-trainer.lsh:402-427``::

        int_code_length(n_states)
        + int_code_length(d)
        + (1 + n_states) * comb_code_length(d, 1 + n_states)
        + n_non_zero_transitions * comb_code_length(d, 1 + n_symbols)

    read left to right as the original accumulates it. The ``(1 + n_states)``
    factor counts the vectors transmitted -- one initial distribution plus one
    transition row per state -- and the per-arc term pays for one emission fibre
    on every transition that survives quantization.

    **``n-states-r`` is not a rounded quantity**, despite four lines later
    ``transition-p-r`` being exactly that. It is ``(to-float n-states)``
    (``hmm-trainer.lsh:405``), so the ``-r`` suffix means *real* in one binding
    and *rounded* in the next, inside one ``let*``. Both are the state count.

    **Transitions are counted after quantization**, which is the point of the
    whole scheme: rounding at ``1 / d`` is what drives a small transition to
    exactly zero, and a sparse topology is cheaper to describe than a dense one
    only because of it. So ``d`` is simultaneously the rounding grid and an
    argument of the code, and it cannot be chosen independently of the topology
    it prices.

    **The symbol axis is the user symbols, not the whole vocabulary.** The
    original passes ``:model:alphabet-size``, its entire alphabet; here
    ``HMMParams`` sizes ``output_p`` over the full ``dataseq`` vocabulary and
    requires the six ADR 0011 reserved fibres to be *exactly zero*, so those
    codes carry no information and must not be paid for. Taking
    ``output_p.shape[-1]`` literally would charge every arc for six symbols the
    model is forbidden to emit. This is not a rounding detail: measured against
    the three tracked models it adds 12, 25 and **105 bits**, the last 24% of
    ``m008``'s model description length -- far more than the differences the
    search arbitrates between, and biased consistently toward sparsity.

    Checked against the ``model-dl`` column each tracked model's own
    ``_training_log`` recorded at its stored ``d``: 58.2207, 142.024 and 439.154
    bits, reproduced to within the log's six-significant-figure print.

    Raises ``ValueError`` for a ``d`` that is not positive and finite, as
    :func:`_data_description_length` does.
    """
    if not (np.isfinite(d) and d > 0):
        raise ValueError(f"d must be positive and finite, got {d}")
    n_states = params.n_states
    n_user_symbols = params.n_symbols - USER_BASE
    quantized = _quantize(params.transition_p, d)
    n_non_zero = int(np.count_nonzero(quantized))
    return (
        _int_code_length(n_states)
        + _int_code_length(d)
        + (1 + n_states) * _comb_code_length(d, 1 + n_states)
        + n_non_zero * _comb_code_length(d, 1 + n_user_symbols)
    )


def _total_description_length(params, records, d):
    """The two-part score: the data half plus the model half, in bits.

    ``update-total-dl`` (``hmm-trainer.lsh:430-433``), and **the one function the
    topology search calls**. Nothing else assembles a score from the pieces: a
    candidate move wins or loses by this number alone, so
    :func:`_data_description_length` and :func:`_model_description_length` are
    implementation of the criterion rather than parts of its interface.

    That is the whole reason for the seam. [PRD §8] registers *which* description
    length should score this search as an open research question -- exact NML for
    HMM classes is intractable, and the tractable route, factorised NML over the
    multinomial case, is a decision this project has not made. Answering it later
    must be a substitution of this function, not a rewrite of the search, which is
    also why the return is a bare ``float``: a one-part code has no data/model
    split, so a return type carrying those two fields would assert the very
    structure that may be replaced.

    **No ``1e100`` sentinel.** The original maps its ``-1`` log-zero sentinel to
    ``1e100`` so an impossible model sorts last rather than best. Here
    :func:`~._numeric.bits` gives ``+inf``, which already sorts last and absorbs
    under addition, so the mapping dissolves -- the same dissolution revision 02
    applied to ``safe->--log`` and revision 03 inherited. A ported ``1e100``
    would be a magic number that is also *comparable*, and a model scoring
    ``1e100 + 1`` would lose to one scoring ``1e100`` for no reason at all.

    The sum is data-then-model, as the original writes it. Checked against the
    ``_total_dl`` each tracked model stores -- a fourth oracle, at four decimals
    rather than the training log's ``%g``, and the only one that constrains both
    halves at once: reproduced to 0.013 bits, whose largest part is the data
    half's known offset at ``d = 29``.

    ``d`` is taken as given; choosing it is ``suggest-d``'s job. Both halves
    reject a ``d`` that is not positive and finite, the data half first.
    """
    return _data_description_length(params, records, d) + _model_description_length(
        params, d
    )


#: What ``update-total-dl`` (``hmm-trainer.lsh:430-433``) substitutes for the total of
#: an impossible model. :func:`_total_description_length` does **not** use it -- it
#: returns ``+inf``, which is correct wherever a score is only compared -- but
#: :func:`_minimize` does arithmetic on scores, and there the two part company. Its
#: parabolic fit subtracts function values, ``inf - inf`` is ``nan``, and ``nan``
#: fails the step-size guard so the step silently collapses to ``-tol1``: the search
#: creeps and converges *on an impossible d*. ``1e100 - 1e100`` is ``0``, which sends
#: the same iteration down the golden-section branch instead, as the original went.
#: So the sentinel is reintroduced here and only here, scoped to the one consumer
#: that differences scores.
_IMPOSSIBLE_TOTAL = 1e100

#: The golden-section fraction, ``(3 - sqrt 5) / 2`` (``util.lsh:124``).
_CGOLD = (3.0 - float(np.sqrt(5.0))) / 2.0

#: Brent's absolute tolerance floor, protecting a minimum at exactly zero (``util.lsh:125``).
_ZEPS = 1e-10

#: ``suggest-d``'s search interval and starting probe (``hmm-trainer.lsh:445``).
#: 3821 is the interior golden-section point of ``[1, 10000]``.
D_LOW, D_START, D_HIGH = 1.0, 3821.0, 10000.0


def _match_sign(a, b):
    """``|a|`` carrying the sign of ``b``, with zero counted positive (``util.lsh:77-82``)."""
    return abs(a) if b >= 0 else -abs(a)


def _minimize(f, a, x, b, tol):
    """Brent's method for a local minimum of ``f`` in ``[a, b]`` from ``x``.

    ``minimize`` (``util.lsh:118-227``), itself Numerical Recipes' ``brent``:
    golden-section steps with parabolic interpolation, returning ``(x, f(x))``.
    Transliterated statement for statement, including its operation order, because
    the probe sequence -- not only the answer -- is what reproduces the original's
    choice of ``d`` (see :func:`_suggest_d`).

    **One difference from Numerical Recipes is kept: there is no ``ITMAX``.** The
    original loops until the bracket closes, and so does this. Brent's bracket
    shrinks every iteration, so it terminates, but it is not bounded by a count.

    **This finds a local minimum, and that is the original's behaviour rather than a
    defect of the port.** Nothing here requires ``f(x)`` to be below ``f(a)`` and
    ``f(b)``, so ``x`` is not a bracketing triple, and on a function with more than
    one basin the search keeps whichever one its first steps slide into.

    Never compiled -- ``util.c`` has no ``C_minimize`` -- so its ``(-float-)``
    declarations were inert and it ran in the interpreter's doubles, unlike the
    code-length primitives above.
    """
    u = 0.0
    v = w = x
    fx = f(x)
    fv = fw = fx
    xm = (a + b) / 2
    tol1 = tol * abs(x) + _ZEPS
    tol2 = 2 * tol1
    d = 0.0
    e = 0.0
    while abs(x - xm) > tol2 - (b - a) / 2:
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = (a - x) if x >= xm else (b - x)
                d = _CGOLD * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = _match_sign(tol1, xm - x)
        else:
            e = (a - x) if x >= xm else (b - x)
            d = _CGOLD * e
        u = (x + d) if abs(d) >= tol1 else (x + _match_sign(tol1, d))
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
        xm = (a + b) / 2
        tol1 = tol * abs(x) + _ZEPS
        tol2 = 2 * tol1
    return x, fx


def _minimize_int(f, a, x, b):
    """The integer minimizing ``f`` near Brent's real answer (``util.lsh:108-116``).

    Runs :func:`_minimize` at a relative tolerance of ``1e-2``, then compares the
    floor of its answer with the next integer up and keeps the lower. **A tie goes
    to the larger integer**: the original tests ``f(floor) < f(floor + 1)``
    strictly. Returns ``(n, f(n))`` with ``n`` a float.
    """
    real_x, _ = _minimize(f, a, x, b, 1e-2)
    lower = float(np.floor(real_x))
    upper = lower + 1.0
    f_lower = f(lower)
    f_upper = f(upper)
    return (lower, f_lower) if f_lower < f_upper else (upper, f_upper)


def _suggest_d(params, records):
    """The quantization resolution ``d`` the original would choose: ``suggest-d``.

    ``hmm-trainer.lsh:441-446``: minimise :func:`_total_description_length` over
    integer ``d`` by :func:`_minimize_int` on ``[1, 10000]`` from 3821. ``d`` is the
    rounding grid *and* an argument of the model code, so a coarser grid makes the
    model cheaper to describe and the data dearer, and this is where that trade is
    struck. It is re-run after every accepted move.

    **It reproduces the original's choice, including where the original chose badly.**
    The data half is not monotone in ``d`` -- rounding thresholds make it jump -- so
    the total has several basins, and Brent keeps the one it slides into from 3821.
    On all three tracked models this returns the stored ``d`` exactly (29, 3, 4), and
    on two of them that is the global integer minimum. On ``m001_0001_001``, the
    one-state model every search starts from, it is not: ``d = 13`` scores 3187.91
    bits against ``d = 29``'s 3198.37, **10.46 bits lower**. Kept deliberately, so
    the port stays checkable against the fixtures; whether the search loop should
    use this or a global scan is that loop's design decision, not this function's.

    An impossible ``d`` is scored ``1e100`` inside the search rather than ``+inf``
    -- see :data:`_IMPOSSIBLE_TOTAL` for why the difference is not cosmetic here.
    Returns ``d`` as a float with an integer value.
    """
    records = list(records)

    def objective(d):
        total = _total_description_length(params, records, d)
        return _IMPOSSIBLE_TOTAL if total == np.inf else total

    d, _ = _minimize_int(objective, D_LOW, D_START, D_HIGH)
    return d
