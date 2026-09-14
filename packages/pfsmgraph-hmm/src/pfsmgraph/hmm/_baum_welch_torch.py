"""The Baum-Welch E-step by reverse-mode differentiation, in torch.

An optional backend of the ``baum_welch`` call, held to the numpy reference in
:mod:`._forward_backward` within a stated tolerance rather than bit for bit
[ADR 0020]. It shares no code with that reference: numpy writes β and ξ out,
and this module writes only the forward pass and lets autograd derive the rest.
That the two agree is the check the backend exists to provide.

**The counts are gradients.** For any positive weight ``x`` in the forward pass,
``x · ∂ln P/∂x = ∂ln P/∂ln x`` is the expected number of times it is used
(Eisner 2016). The leaves are ``init_state_p`` and the arc table
``w[i, j, u] = transition_p[i, j] * output_p[i, j, present[u]]``, not the three
parameter arrays, so ``w · ∂w`` is ``Σ_t ξ_t(i, j)`` over the positions emitting
``present[u]``: the emission counts, per symbol present. The transition counts
are then their sum over ``u``, which is how the reference forms them too, so
``Σ_k emission_counts[i, j, k] == transition_counts[i, j]`` holds by
construction and the M-step's fibres sum to 1. Taking ``T · ∂T`` and ``O · ∂O``
separately does not: near ``1e-321`` the two round a whole subnormal unit apart,
which is 0.2% of the value, and ``HMMParams`` rejects the fibre. Leaves are
probabilities rather than logarithms because ``exp(log p)`` is not bit-exactly
``p``.

**The scale factors are constants to autograd.** Each ``q_t`` is detached, and
the backward pass starts from ``ln Σ_j`` of the last unnormalised column, not
from ``Σ ln q_t``. The gradient is the same, since dividing by a constant
leaves a log-derivative unchanged, but the graph then holds only ``+``, ``*``
and division by constants, so every count is a sum of nonnegative terms.
Differentiating through ``q_t`` instead subtracts near-equal terms, and where a
true count is near zero the rounding left behind can be negative: down to
``-3e-19`` on EM trajectories whose parameters had fallen to ``1e-50``.

**The arithmetic follows ADR 0020 as far as torch allows.** The pass is scaled,
not in log space, so gradients stay finite on unreachable states. The arc table
is built by numpy exactly as the reference builds it; the reductions over states
are torch's to order, and are where the tolerance comes from. The reported
description length is ``bits`` of the scale factors on the host, as for every
other backend. Measured against the reference, every count agrees within
``N · eps · max(1, count)`` and the description length within
``N · eps · max(1, bits)``; the tests hold both to four times that.

**float64 on CPU.** No environment chooses a device (ADR 0021), and device
placement belongs to the batching subgoal.

**Nothing raises, as for every kernel.** An impossible sequence has a zero scale
factor, after which every column is zeros rather than ``nan``; the description
length is ``+inf``, no backward pass runs, and every count is zero, as the
reference returns. So is every count of an empty record.

.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

import numpy as np
import torch

from ._numeric import bits

__all__: list[str] = []


def _forward(init, w, sym):
    """The scaled forward pass: ``(scale, log_last)``.

    ``init`` is ``(S,)`` and ``w`` the ``(S, S, U)`` arc table, both tensors;
    ``sym`` indexes ``w``'s last axis per position. ``scale`` is ``(N + 1,)``
    with ``scale[0] = 1``, as the reference's is. ``log_last`` is ``ln Σ_j`` of
    the last unnormalised column, in the autograd graph: with the scale factors
    detached it differs from ``ln P`` by a constant, so its gradient is
    ``∂ln P``, and its value is not used.
    """
    alpha = init
    column = init
    factors = [torch.ones((), dtype=torch.float64)]
    for u in sym:
        column = (alpha[:, None] * w[:, :, u]).sum(dim=0)
        q = column.detach().sum()
        alpha = column / q if q > 0 else torch.zeros_like(column)
        factors.append(q)
    return torch.stack(factors), torch.log(column.sum())


def _e_step(init_state_p, transition_p, output_p, codes):
    """``((init_counts, transition_counts, emission_counts), bits)`` for one record.

    The signature of every ``baum_welch`` backend. Arrays in and out are numpy
    float64; the counts have the reference's shapes, ``(S,)``, ``(S, S)`` and
    ``(S, S, A)``.
    """
    init = np.asarray(init_state_p, dtype=np.float64)
    transition = np.asarray(transition_p, dtype=np.float64)
    output = np.asarray(output_p, dtype=np.float64)
    present, sym = np.unique(np.asarray(codes), return_inverse=True)
    w = transition[:, :, np.newaxis] * output[:, :, present]

    init_leaf = torch.tensor(init, requires_grad=True)
    w_leaf = torch.tensor(w, requires_grad=True)
    scale, log_last = _forward(init_leaf, w_leaf, sym.tolist())
    total = float(np.add.accumulate(bits(scale.numpy()))[-1])

    init_counts = np.zeros_like(init)
    transition_counts = np.zeros_like(transition)
    emission_counts = np.zeros_like(output)
    if present.size == 0 or not np.isfinite(total):
        return (init_counts, transition_counts, emission_counts), total

    log_last.backward()
    per_symbol = w * w_leaf.grad.numpy()
    init_counts = init * init_leaf.grad.numpy()
    emission_counts[:, :, present] = per_symbol
    transition_counts = np.add.accumulate(per_symbol, axis=2)[:, :, -1]
    return (init_counts, transition_counts, emission_counts), total
