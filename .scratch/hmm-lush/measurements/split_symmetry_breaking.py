"""Compare ways of making a split state's twins stop being identical.

Evidence for axis 2 of `feat/hmm-split-state` (goal 3), bound for the ADR that also
records axis 1. Every arm carries `init_state_p` (axis 1, decided): the split state
and its twin share `init[s] / 2` and every other state keeps its own.

The arms differ only in how the twins are told apart:

- `lush`: exact halves inbound, outbound rows copied, and every fibre leaving either
  twin redrawn with `rand_p_vector(..., 0.1)`, as `hmm-param.lsh:190-207` does.
- `inbound(w)`: each predecessor `j` splits its arc into the state as
  `(1/2 + w * U(-1/2, 1/2)) * T[j, s]`, the remainder to the twin; outbound rows and
  every fibre are copied. `w = 0.01` is the user's `0.099 / 0.101` example.
- `inbound(w)+seed(e)`: as `inbound(w)`, and where the twins have a single distinct
  predecessor row -- the case the perturbation cannot inform -- the twins' outbound
  fibres become `(1 - e) * learned + e * rand_p_vector(...)`. Other splits are
  identical to `inbound(w)`.
- `inbound(w)+min(n)`: a control, not a candidate. As `inbound(w)`, but EM runs `n`
  cycles before `run-converge`'s stop rule may fire. An inbound split starts at exactly
  the incumbent's likelihood, so its first batches move the description length by
  under `run-converge`'s 0.1 bits and the rule can stop before the twins separate;
  this arm tests whether that, rather than the design, is what holds it back.

A predecessor *row* is distinct when it can differ between the twins' entries: every
`j != s` with `T[j, s] > 0`, plus the twins' own row once if `T[s, s] > 0`, since the
twin's row is a copy of it. A split with exactly one is `single`.

Incumbents: the three tracked `set02a_200` models, and models trained here on
`set11a_dInt` (1449 symbols over 25) from a fixed seed at 1, 4 and 8 states, which is
less toy-sized in its alphabet. Each split re-converges with `baum_welch`'s default
`run-converge` stop, as `try-split` does, and is scored with `_suggest_d` and
`_total_description_length`. Reported per split: total description length against the
incumbent's, cycles, and the twins' final gap (max absolute difference between their
outbound rows and fibres). The data half is reported beside the total because the
total also moves with `d`: twins left exactly identical still score differently
across arms when their inbound fractions round differently.

Imports private modules, as the other scripts here do; nothing under `packages/`
imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/split_symmetry_breaking.py [--quick]

`--quick` runs one seed on the `set02a_200` incumbents only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, baum_welch
from pfsmgraph.hmm._mdl import _data_description_length, _suggest_d, _total_description_length
from pfsmgraph.hmm._numeric import rand_p_vector

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params  # noqa: E402

QUICK = "--quick" in sys.argv
SEEDS = (0,) if QUICK else (0, 1, 2)
BACKEND = "cython"


def distinct_predecessor_rows(p: HMMParams, s: int) -> int:
    col = p.transition_p[:, s] > 0
    return int(col.sum() - col[s]) + int(col[s])


def split(p: HMMParams, s: int, arm: str, rng: np.random.Generator) -> HMMParams:
    S, U = p.n_states, p.n_symbols - USER_BASE
    arm = arm.split("+min(")[0]
    kind, _, rest = arm.partition("(")
    w = e = 0.0
    if kind.startswith("inbound"):
        w = float(rest.split(")")[0])
        if "seed(" in arm:
            e = float(arm.split("seed(")[1].rstrip(")"))

    init = np.append(p.init_state_p, 0.5 * p.init_state_p[s])
    init[s] = init[S]

    T = np.zeros((S + 1, S + 1))
    T[:S, :S] = p.transition_p
    for j in range(S):
        frac = 0.5 + w * rng.uniform(-0.5, 0.5) if w else 0.5
        T[j, s] = frac * p.transition_p[j, s]
        T[j, S] = p.transition_p[j, s] - T[j, s]
    T[S] = T[s]  # the twin's outbound row copies the (already split) row of s

    O = np.zeros((S + 1, S + 1, p.n_symbols))
    O[:S, :S] = p.output_p
    O[:S, S] = O[:S, s]
    O[S] = O[s]
    if kind == "lush":
        for j in range(S + 1):  # the original's draw order
            O[s, j, USER_BASE:] = rand_p_vector(U, 0.1, rng)
            O[S, j, USER_BASE:] = rand_p_vector(U, 0.1, rng)
    elif e and distinct_predecessor_rows(p, s) == 1:
        for t in (s, S):
            for j in range(S + 1):
                O[t, j, USER_BASE:] = (1 - e) * O[t, j, USER_BASE:] + e * rand_p_vector(U, 0.1, rng)
    return HMMParams(init, T, O, p.vocabulary)


def twin_gap(p: HMMParams, s: int) -> float:
    t = p.n_states - 1
    return float(max(np.abs(p.output_p[s] - p.output_p[t]).max(),
                     np.abs(p.transition_p[s] - p.transition_p[t]).max()))


def score(p: HMMParams, records) -> tuple[float, float]:
    d = _suggest_d(p, records)
    return _total_description_length(p, records, d), _data_description_length(p, records, d)


def converge(q: HMMParams, records, arm: str):
    if "+min(" not in arm:
        r = baum_welch(q, records, backend=BACKEND)
        return r.params, r.cycles
    n = int(arm.split("+min(")[1].rstrip(")"))
    forced = baum_welch(q, records, backend=BACKEND, max_cycles=n, change_bits=1e-12, patience=10**9)
    r = baum_welch(forced.params, records, backend=BACKEND)
    return r.params, forced.cycles + r.cycles


def set11a_incumbents():
    path = Path(".scratch/hmm-lush/Training/set11a_dInt/set11a_dInt.sds/0.seq")
    raw = np.array(path.read_text().split(), dtype=np.int64)
    A = int(raw.max()) + 1
    records = [SequenceRecord(raw + USER_BASE)]
    vocab = SymbolTable([f"s{i}" for i in range(A)])
    rng = np.random.default_rng(20260916)
    out = []
    for S in (1, 4, 8):
        O = np.zeros((S, S, USER_BASE + A))
        O[..., USER_BASE:] = rng.dirichlet(np.ones(A), size=(S, S))
        start = HMMParams(rng.dirichlet(np.ones(S)), rng.dirichlet(np.ones(S), size=S), O, vocab)
        out.append((f"set11a S={S}", baum_welch(start, records, backend=BACKEND).params, records))
    return out


ARMS = (
    "lush",
    "inbound(0.01)",
    "inbound(0.1)",
    "inbound(0.01)+seed(0.01)",
    "inbound(0.1)+seed(0.01)",
    "inbound(0.1)+min(200)",
)


def main():
    record = [load_corpus_record()]
    incumbents = [(m.removesuffix(".hmm"), load_params(FIXTURES / m), record) for m in SAVED_MODELS]
    if not QUICK:
        incumbents += set11a_incumbents()
    print(f"backend {BACKEND}; seeds {SEEDS}; arms {', '.join(ARMS)}")
    print("dDL / dData = candidate total / data description length minus the incumbent's (negative is better)")
    for name, inc, records in incumbents:
        base, base_data = score(inc, records)
        print(f"\n{name}: S={inc.n_states}, incumbent total {base:.1f} bits, data {base_data:.1f}")
        best = {arm: np.inf for arm in ARMS}
        for s in range(inc.n_states):
            preds = distinct_predecessor_rows(inc, s)
            for arm in ARMS:
                cells = []
                for seed in SEEDS:
                    rng = np.random.default_rng([seed, s, ARMS.index(arm)])
                    params, cycles = converge(split(inc, s, arm, rng), records, arm)
                    total, data = score(params, records)
                    best[arm] = min(best[arm], total - base)
                    cells.append((total - base, data - base_data, cycles, twin_gap(params, s)))
                dls = [c[0] for c in cells]
                print(f"  s={s} preds={preds} {'single' if preds == 1 else '      '} {arm:26s} "
                      f"dDL min {min(dls):8.1f} mean {np.mean(dls):8.1f} | dData "
                      f"{'/'.join(f'{c[1]:.1f}' for c in cells):>20s} | cycles "
                      f"{'/'.join(str(c[2]) for c in cells):>12s} | gap "
                      f"{'/'.join(f'{c[3]:.0e}' for c in cells)}", flush=True)
        print("  best split per arm (what suggest-split would adopt): "
              + "; ".join(f"{a} {best[a]:.1f}" for a in ARMS))


if __name__ == "__main__":
    main()
