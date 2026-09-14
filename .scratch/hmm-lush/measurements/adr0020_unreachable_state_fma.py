"""Autograd at a state unreachable everywhere, and numba multiply-add contraction.

Evidence for ADR 0020, the unreachable-state half of the "autograd" bullet and
the "contraction" bullet. Needs numpy, torch and numba. The contraction check
is only meaningful on a CPU with FMA. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/adr0020_unreachable_state_fma.py
"""
import numpy as np
import torch
from numba import njit

print("== autograd with a state that is unreachable at every position ==")
init = np.array([0.5, 0.5, 0.0])
trans = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
out = np.full((3, 3, 2), 0.5)
codes = [0, 1, 1, 0]
for domain in ("scaled", "log"):
    with np.errstate(divide="ignore"):
        th = [torch.tensor(np.log(x), requires_grad=True) for x in (init, trans, out)]
    if domain == "scaled":
        a = th[0].exp(); ll = 0.0
        for k in codes:
            a = a @ (th[1].exp() * th[2].exp()[:, :, k]); q = a.sum(); a = a / q; ll = ll + q.log()
    else:
        la = th[0]
        for k in codes:
            la = torch.logsumexp(la[:, None] + th[1] + th[2][:, :, k], dim=0)
        ll = torch.logsumexp(la, dim=0)
    ll.backward()
    g = th[1].grad.numpy()
    print(f"  {domain:6s}: logP {ll.item():.6f}  nan grads in transition: {int(np.isnan(g).sum())} of 9")

print("== does numba fuse a*b*c + d into FMA? (this CPU has FMA) ==")


@njit
def acc(x, y, z):
    s = 0.0
    for i in range(x.size):
        s += x[i] * y[i] * z[i]
    return s


rng = np.random.default_rng(1)
diff = 0
for _ in range(2000):
    x, y, z = rng.random((3, 8))
    s = 0.0
    for i in range(8):
        s = s + float(x[i]) * float(y[i]) * float(z[i])   # Python floats: never fused
    diff += acc(x, y, z) != s
print(f"  numba vs unfused Python: {diff} of 2000 sums differ")


@njit(fastmath=True)
def acc_fast(x, y, z):
    s = 0.0
    for i in range(x.size):
        s += x[i] * y[i] * z[i]
    return s


diff = 0
for _ in range(2000):
    x, y, z = rng.random((3, 8))
    s = 0.0
    for i in range(8):
        s = s + float(x[i]) * float(y[i]) * float(z[i])
    diff += acc_fast(x, y, z) != s
print(f"  numba fastmath=True vs unfused Python: {diff} of 2000 sums differ")
