"""The Baum-Welch E-step by reverse-mode differentiation, in torch.

An optional backend of the ``baum_welch`` call, held to the numpy reference in
:mod:`._forward_backward` within a stated tolerance rather than bit for bit
[ADR 0020]. It shares no code with that reference: numpy writes β and ξ out,
and this module writes only the forward pass and lets autograd derive the rest.
That the two agree is the check the backend exists to provide.

**The counts are gradients.** For any parameter ``p`` of the model,
``p · ∂ln P/∂p = ∂ln P/∂ln p``, and for the three arrays that is exactly the
expected count ``run-add`` accumulates: ``γ₀`` for ``init_state_p``,
``Σ_t ξ_t(i, j)`` for ``transition_p[i, j]``, and that sum restricted to the
positions emitting ``k`` for ``output_p[i, j, k]`` (Eisner 2016). The leaves are
the probabilities as given, not their logarithms, because ``exp(log p)`` is not
bit-exactly ``p`` and would move the model before the forward pass began; and a
dead arc contributes ``0 · finite = 0`` with no ``log(0)`` leaf.

**The arithmetic follows ADR 0020 as far as torch allows.** The pass is scaled,
not in log space, so gradients stay finite on unreachable states. The arc table
``w`` is an elementwise product, which is exactly rounded, so it holds numpy's
bits; the reductions over states are torch's to order, and are where the
tolerance comes from. Logarithms are taken of the scale factors only: ``ln`` in
the graph, whose gradient is the count in nats with no conversion, and the
reported description length through :func:`~._numeric.bits` on the host, as for
every other backend.

**float64 on CPU.** No environment chooses a device (ADR 0021), and device
placement belongs to the batching subgoal.

**Nothing raises, as for every kernel.** An impossible sequence has a zero scale
factor; the forward pass guards the division so the later columns are zeros
rather than ``nan``, the description length is ``+inf``, no backward pass runs,
and every count is zero, as the reference returns.

.. [ADR 0020] ``docs/design/adr/0020-scaled-probability-domain-forward-backward.md``
"""

from __future__ import annotations

import numpy as np
import torch

from ._numeric import bits

__all__: list[str] = []


def _forward(init_state_p, transition_p, output_p, codes):
    """The scaled forward pass over tensors: ``(scale, log_likelihood)``.

    ``scale`` is ``(N + 1,)`` with ``scale[0] = 1``, as the reference's is;
    ``log_likelihood`` is ``Σ ln scale`` over the non-zero factors, a 0-d tensor
    in the autograd graph.
    """
    present, sym = np.unique(np.asarray(codes), return_inverse=True)
    w = transition_p[:, :, None] * output_p[:, :, torch.from_numpy(present)]
    alpha = init_state_p
    factors = [torch.ones((), dtype=torch.float64)]
    for u in sym.tolist():
        column = (alpha[:, None] * w[:, :, u]).sum(dim=0)
        q = column.sum()
        alive = q > 0
        alpha = torch.where(alive, column / torch.where(alive, q, 1.0), 0.0)
        factors.append(q)
    scale = torch.stack(factors)
    log_likelihood = torch.log(torch.where(scale > 0, scale, 1.0)).sum()
    return scale, log_likelihood


def _e_step(init_state_p, transition_p, output_p, codes):
    """``((init_counts, transition_counts, emission_counts), bits)`` for one record.

    The signature of every ``baum_welch`` backend. Arrays in and out are numpy
    float64; the counts have the reference's shapes, ``(S,)``, ``(S, S)`` and
    ``(S, S, A)``.
    """
    leaves = [
        torch.tensor(np.asarray(p, dtype=np.float64), requires_grad=True)
        for p in (init_state_p, transition_p, output_p)
    ]
    scale, log_likelihood = _forward(*leaves, codes)
    total = float(np.add.accumulate(bits(scale.detach().numpy()))[-1])
    if len(codes) == 0 or not np.isfinite(total):
        return tuple(np.zeros_like(p, dtype=np.float64) for p in (init_state_p, transition_p, output_p)), total
    log_likelihood.backward()
    counts = tuple(
        np.asarray(p, dtype=np.float64) * leaf.grad.numpy()
        for p, leaf in zip((init_state_p, transition_p, output_p), leaves)
    )
    return counts, total
