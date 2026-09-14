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
the backward pass starts from ``ln Σ_j`` of each record's last unnormalised
column, not from ``Σ ln q_t``. The gradient is the same, since dividing by a
constant leaves a log-derivative unchanged, but the graph then holds only ``+``,
``*`` and division by constants, so every count is a sum of nonnegative terms.
Differentiating through ``q_t`` instead subtracts near-equal terms, and where a
true count is near zero the rounding left behind can be negative: down to
``-3e-19`` on EM trajectories whose parameters had fallen to ``1e-50``.

**Every record has its own leaves.** The batch's counts are returned per record,
as every ``baum_welch`` kernel returns them, so the leaves are ``(B, S)`` and
``(B, S, S, U)`` copies and one backward pass over the sum of the records'
losses leaves each record's gradient in its own slice. A padded step carries α
and the last column unchanged through ``torch.where``, so padding is off the
graph rather than weighted by zero. An impossible or empty record's loss term is
replaced by ``ln 1`` *before* the logarithm: masking it afterwards would still
send ``0 · (1/0) = nan`` through the logarithm's backward.

**The arithmetic follows ADR 0020 as far as torch allows.** The pass is scaled,
not in log space, so gradients stay finite on unreachable states. The arc table
is built by numpy exactly as the reference builds it; the reductions over states
are torch's to order, and are where the tolerance comes from. The reported
description length is ``bits`` of the scale factors on the host, as for every
other backend. Measured against the reference, every count agrees within
``N · eps · max(1, count)`` and the description length within
``N · eps · max(1, bits)``; the tests hold both to four times that.

**float64 on ``device``, the CPU when ``None``.** No environment chooses a device
(ADR 0021): the caller names one, :func:`_device` probes it before any work, and
nothing falls back. Measured on an NVIDIA L4, the tolerance is unchanged there.

**Nothing raises, as for every kernel.** An impossible sequence has a zero scale
factor, after which its columns are zeros rather than ``nan``; its description
length is ``+inf`` and every count is zero, as the reference returns. So is every
count of an empty record.

.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

import numpy as np
import torch

from ._backends import BackendUnavailableError
from ._numeric import bits

__all__: list[str] = []


def _device(device):
    """The ``torch.device`` for ``baum_welch``'s ``device=``, probed by allocation.

    ``None`` is the CPU. A name torch cannot parse raises ``ValueError``; a device
    that parses but cannot hold a float64 tensor here -- an ordinal past the last
    GPU, a backend this torch build lacks -- raises
    :class:`BackendUnavailableError` with torch's own message.
    """
    if device is None:
        return torch.device("cpu")
    try:
        parsed = torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"device {device!r} is not a torch device name: {exc}") from None
    try:
        torch.empty(0, dtype=torch.float64, device=parsed)
    except Exception as exc:
        raise BackendUnavailableError(
            f"backend 'torch' cannot allocate on device {device!r}: {exc}"
        ) from None
    return parsed


def _e_step_batch(init_state_p, transition_p, output_p, codes, lengths, device=None):
    """Per-record ``(counts, bits)`` for a padded batch, as gradients.

    The signature of every ``baum_welch`` backend. ``codes`` ``(B, L)`` and
    ``lengths`` ``(B,)`` are ``pad_collate``'s; arrays in and out are numpy
    float64, and the counts have shapes ``(B, S)``, ``(B, S, S)`` and
    ``(B, S, S, A)``. ``device`` is a ``torch.device``, a device name, or ``None``
    for the CPU.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device or "cpu")
    codes = np.asarray(codes)
    lengths = np.asarray(lengths, dtype=np.int64)
    batch, width = codes.shape
    init = np.asarray(init_state_p, dtype=np.float64)
    transition = np.asarray(transition_p, dtype=np.float64)
    output = np.asarray(output_p, dtype=np.float64)
    size = init.shape[0]
    present, sym = np.unique(codes, return_inverse=True)
    sym = sym.reshape(codes.shape)
    w = transition[:, :, np.newaxis] * output[:, :, present]

    init_leaf = torch.tensor(np.broadcast_to(init, (batch, size)), device=dev, requires_grad=True)
    w_leaf = torch.tensor(np.broadcast_to(w, (batch, *w.shape)), device=dev, requires_grad=True)
    live = torch.as_tensor(np.arange(width)[np.newaxis, :] < lengths[:, np.newaxis], device=dev)
    sym_t = torch.as_tensor(sym, device=dev)
    rows = torch.arange(batch, device=dev)

    alpha = init_leaf
    last = init_leaf
    factors = [torch.ones(batch, dtype=torch.float64, device=dev)]
    for t in range(width):
        column = (alpha[:, :, None] * w_leaf[rows, :, :, sym_t[:, t]]).sum(dim=1)
        q = column.detach().sum(dim=1)
        on = live[:, t]
        possible = on & (q > 0)
        q_safe = torch.where(possible, q, torch.ones_like(q))
        scaled = torch.where(possible[:, None], column / q_safe[:, None], torch.zeros_like(column))
        alpha = torch.where(on[:, None], scaled, alpha)
        last = torch.where(on[:, None], column, last)
        factors.append(torch.where(on, q, torch.ones_like(q)))
    scale = torch.stack(factors, dim=1).cpu().numpy()
    total = np.add.accumulate(bits(scale), axis=1)[:, -1]

    init_counts = np.zeros((batch, size), dtype=np.float64)
    transition_counts = np.zeros((batch, size, size), dtype=np.float64)
    emission_counts = np.zeros((batch, size, size, output.shape[2]), dtype=np.float64)
    trainable = (lengths > 0) & np.isfinite(total)
    if not trainable.any():
        return (init_counts, transition_counts, emission_counts), total

    keep = torch.as_tensor(trainable, device=dev)
    mass = last.sum(dim=1)
    torch.log(torch.where(keep, mass, torch.ones_like(mass))).sum().backward()

    per_symbol = w[np.newaxis] * w_leaf.grad.cpu().numpy()
    per_symbol[~trainable] = 0.0
    init_counts = init[np.newaxis] * init_leaf.grad.cpu().numpy()
    init_counts[~trainable] = 0.0
    emission_counts[:, :, :, present] = per_symbol
    transition_counts = np.add.accumulate(per_symbol, axis=3)[:, :, :, -1]
    return (init_counts, transition_counts, emission_counts), total
