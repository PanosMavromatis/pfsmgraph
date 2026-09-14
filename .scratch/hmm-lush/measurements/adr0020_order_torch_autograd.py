"""Bit-exactness across summation orders, torch vs numpy, and autograd at dead arcs.

Evidence for ADR 0020, the "order" and "autograd" bullets. Needs numpy and torch
(the CUDA row appears only with a device). Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/adr0020_order_torch_autograd.py
"""
import numpy as np
import torch

rng = np.random.default_rng(20260914)


def rand_model(S, A, zeros=0.0):
    init = rng.dirichlet(np.ones(S))
    trans = rng.dirichlet(np.ones(S), size=S)
    if zeros:
        mask = rng.random((S, S)) < zeros
        mask[np.arange(S), np.arange(S)] = False
        trans = np.where(mask, 0.0, trans)
        trans /= trans.sum(1, keepdims=True)
    out = rng.dirichlet(np.ones(A), size=(S, S))
    return init, trans, out


def scaled_blas(init, trans, out, codes):
    a = init.copy(); qs = []
    for k in codes:
        a = a @ (trans * out[:, :, k]); q = a.sum(); a = a / q; qs.append(q)
    return np.array(qs)


def scaled_loop(init, trans, out, codes):
    """What a compiled kernel does: serial ascending i, then ascending j."""
    S = len(init); a = init.copy(); qs = []
    for k in codes:
        nxt = np.zeros(S)
        for j in range(S):
            acc = 0.0
            for i in range(S):
                acc += a[i] * trans[i, j] * out[i, j, k]
            nxt[j] = acc
        q = 0.0
        for j in range(S):
            q += nxt[j]
        a = nxt / q; qs.append(q)
    return np.array(qs)


print("== does summation order change the bits? (scale factors Q_t) ==")
for S in (4, 16, 64):
    m = rand_model(S, 6); codes = rng.integers(0, 6, 60)
    b, l = scaled_blas(*m, codes), scaled_loop(*m, codes)
    print(f"  S={S:3d} BLAS vs ascending loop: {np.mean(b != l):5.1%} of Q_t differ, "
          f"max |d bits| {np.max(np.abs(np.log2(b) - np.log2(l))):.1e}")

print("== torch float64 vs numpy, same ops ==")
m = rand_model(32, 6); codes = rng.integers(0, 6, 60)
ref = scaled_blas(*m, codes)
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    I, T, O = (torch.tensor(x, dtype=torch.float64, device=dev) for x in m)
    a = I.clone(); qs = []
    for k in codes:
        a = a @ (T * O[:, :, int(k)]); q = a.sum(); a = a / q; qs.append(q)
    got = torch.stack(qs).cpu().numpy()
    print(f"  {dev:4s}: {np.mean(got != ref):5.1%} of Q_t differ from numpy, max rel {np.max(np.abs(got / ref - 1)):.1e}")


def torch_loglik(theta_i, theta_t, theta_o, codes, domain):
    if domain == "scaled":
        I, T, O = theta_i.exp(), theta_t.exp(), theta_o.exp()
        a = I; ll = 0.0
        for k in codes:
            a = a @ (T * O[:, :, k]); q = a.sum(); a = a / q; ll = ll + q.log()
        return ll
    la = theta_i
    for k in codes:
        la = torch.logsumexp(la[:, None] + theta_t + theta_o[:, :, k], dim=0)
    return torch.logsumexp(la, dim=0)


def numpy_counts(init, trans, out, codes):
    """Explicit scaled alpha/beta/xi, as run-add does: expected transition counts."""
    N, S = len(codes), len(init)
    al = np.zeros((N + 1, S)); be = np.zeros((N + 1, S)); Q = np.ones(N + 1)
    al[0] = init
    for t, k in enumerate(codes, 1):
        al[t] = al[t - 1] @ (trans * out[:, :, k]); Q[t] = al[t].sum(); al[t] /= Q[t]
    be[N] = 1.0 / Q[N]
    for t in range(N - 1, -1, -1):
        be[t] = (trans * out[:, :, codes[t]]) @ be[t + 1] / Q[t]
    C = np.zeros((S, S))
    for t in range(N):
        C += al[t][:, None] * trans * out[:, :, codes[t]] * be[t + 1][None, :]
    return C


print("== autograd count identity: d logP / d log a_ij == expected count, with 40% dead arcs ==")
init, trans, out = rand_model(6, 5, zeros=0.4)
init[1] = 0.0; init /= init.sum()           # a state that cannot start
codes = rng.integers(0, 5, 30)
C = numpy_counts(init, trans, out, codes)
for domain in ("scaled", "log"):
    with np.errstate(divide="ignore"):
        th = [torch.tensor(np.log(x), dtype=torch.float64, requires_grad=True) for x in (init, trans, out)]
    ll = torch_loglik(*th, [int(k) for k in codes], domain)
    ll.backward()
    g = th[1].grad.numpy()
    live = trans > 0
    nan = np.isnan(g).sum()
    print(f"  {domain:6s}: logP {ll.item():.12f}  nan grads {nan}  "
          f"max |grad - count| on live arcs {np.nanmax(np.abs(g - C)[live]):.1e}  "
          f"on dead arcs grad={np.unique(g[~live])}")
