"""Compare scaled-probability vs log-space forward passes: accuracy and speed.

Evidence for ADR 0020, the "accuracy" and "speed" bullets. Standalone: imports
numpy only, nothing from ``packages/``. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/adr0020_accuracy_speed.py

Timings are host-specific. The ADR records the host they were taken on.
"""
import itertools, math, time
from fractions import Fraction
import numpy as np


def logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    finite = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        r = np.log(np.sum(np.exp(x - finite), axis=axis, keepdims=True)) + finite
    return np.squeeze(r, axis=axis) if axis is not None else r.item()

LN2 = math.log(2.0)


def rand_model(rng, S, A, zeros=0.0):
    init = rng.dirichlet(np.ones(S))
    trans = rng.dirichlet(np.ones(S), size=S)
    if zeros:
        mask = rng.random((S, S)) < zeros
        mask[np.arange(S), rng.integers(0, S, S)] = False
        trans = np.where(mask, 0.0, trans)
        trans /= trans.sum(1, keepdims=True)
    out = rng.dirichlet(np.ones(A), size=(S, S))
    return init, trans, out


def bits_exact(init, trans, out, codes):
    """Exact -log2 P(seq) over every state path, in rationals."""
    S = len(init)
    I = [Fraction(x) for x in init]
    T = [[Fraction(x) for x in r] for r in trans]
    O = [[[Fraction(x) for x in c] for c in r] for r in out]
    total = Fraction(0)
    for path in itertools.product(range(S), repeat=len(codes) + 1):
        p = I[path[0]]
        for t, k in enumerate(codes):
            p *= T[path[t]][path[t + 1]] * O[path[t]][path[t + 1]][k]
            if p == 0:
                break
        total += p
    if total == 0:
        return math.inf
    # -log2 of a rational, exactly enough: split numerator/denominator
    return -(math.log2(total.numerator) - math.log2(total.denominator))


def scaled(init, trans, out, codes):
    alpha = init.copy()
    dl = 0.0
    for k in codes:
        alpha = alpha @ (trans * out[:, :, k])
        q = alpha.sum()
        if q == 0:
            return math.inf
        alpha /= q
        dl += -np.log2(q)
    return dl


def logspace_e(init, trans, out, codes):
    with np.errstate(divide="ignore"):
        la = np.log(init)
        arc = np.log(trans[:, :, None] * out)
    for k in codes:
        la = logsumexp(la[:, None] + arc[:, :, k], axis=0)
    return -logsumexp(la) / LN2


def logspace_2(init, trans, out, codes):
    with np.errstate(divide="ignore"):
        la = np.log2(init)
        arc = np.log2(trans[:, :, None] * out)
    for k in codes:
        la = np.logaddexp2.reduce(la[:, None] + arc[:, :, k], axis=0)
    return -np.logaddexp2.reduce(la)


rng = np.random.default_rng(20260914)
print("== accuracy vs exact enumeration (S=3, A=4, N=7), error in bits ==")
errs = {"scaled": [], "log_e": [], "log_2": []}
for _ in range(200):
    m = rand_model(rng, 3, 4, zeros=0.3)
    codes = rng.integers(0, 4, 7)
    ex = bits_exact(*m, codes)
    if not math.isfinite(ex):
        continue
    for name, f in (("scaled", scaled), ("log_e", logspace_e), ("log_2", logspace_2)):
        errs[name].append(abs(f(*m, codes) - ex) / ex)
for k, v in errs.items():
    v = np.array(v)
    print(f"  {k:7s} n={len(v)} median rel err {np.median(v):.2e}  max {v.max():.2e}")

print("== long sequence drift (S=16, A=8, N=100000): each vs scaled ==")
m = rand_model(rng, 16, 8)
codes = rng.integers(0, 8, 100_000)
s = scaled(*m, codes)
e = logspace_e(*m, codes)
b = logspace_2(*m, codes)
print(f"  total {s:.6f} bits; log_e - scaled {e - s:+.3e}; log_2 - scaled {b - s:+.3e}")
print(f"  relative: {abs(e - s) / s:.2e}, {abs(b - s) / s:.2e}")

print("== impossible sequence ==")
init = np.array([1.0, 0.0]); trans = np.array([[1.0, 0.0], [0.0, 1.0]])
out = np.zeros((2, 2, 2)); out[0, 0, 0] = 1.0; out[1, 1, 1] = 1.0
for name, f in (("scaled", scaled), ("log_e", logspace_e), ("log_2", logspace_2)):
    with np.errstate(all="ignore"):
        print(f"  {name:7s} {f(init, trans, out, np.array([0, 1]))}")

print("== speed, numpy per-timestep vectorised, N=400, forward only (ms) ==")
for S in (5, 16, 64, 160):
    m = rand_model(rng, S, 12)
    codes = rng.integers(0, 12, 400)
    row = []
    for f in (scaled, logspace_e, logspace_2):
        f(*m, codes[:5])
        t0 = time.perf_counter(); reps = 0
        while time.perf_counter() - t0 < 0.5:
            f(*m, codes); reps += 1
        row.append((time.perf_counter() - t0) / reps * 1e3)
    print(f"  S={S:4d} scaled {row[0]:8.2f}  log_e {row[1]:8.2f}  log_2 {row[2]:8.2f}  (log_e/scaled {row[1]/row[0]:.1f}x)")
