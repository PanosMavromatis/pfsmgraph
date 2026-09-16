"""Split the cost of one candidate topology move into building it and scoring it.

Evidence for the parameter representation (`exp/hmm-param-representation`, goal 1).
The question is whether *building* a candidate model -- new arrays, `HMMParams`
validation and freezing, the stationary solve behind `state_p` -- is a cost worth
choosing a storage layout for. The search builds a candidate for every move it
tries and rejects most of them, so if building were expensive, over-allocating
and slicing, or an edge list, could pay for itself.

The workload is split- and merge-shaped, without a search loop: from an incumbent
of `S` states, build a candidate of `S + 1` (a partner for state 0, inbound and
initial mass halved, outbound fibres randomised at noise 0.1, as `split-state`
does) or `S - 1` (state 1 folded into 0, rows weighted by `state_p`, as
`merge-states` does). The corpus is `set11a_dInt`, 1449 symbols over 25, the one
the master plan's 62,500-entry figure was measured on. Scoring is what the search
does next: `_suggest_d`, and a `baum_welch` re-convergence.

Two tables. The first is on the `python` backend, the default. The second repeats
the ratio on `cython`, since a faster scorer is what could make building a larger
share. `_suggest_d` has no `backend=` and runs the numpy forward pass in both.

Imports private modules (`_mdl`, `_numeric`), as the ADR 0020 scripts do; nothing
under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/param_representation_move_cost.py

and add `--cython-only` to skip the first table, which takes a few minutes. Timings are host-specific, and the header names the host they were taken on. The
conclusion rests on a ratio of four to five orders of magnitude, not on the figures.
"""

from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path

import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, backends, baum_welch
from pfsmgraph.hmm._mdl import _corpus_description_length, _suggest_d, _total_description_length
from pfsmgraph.hmm._numeric import rand_p_vector

A = 25
CORPUS = Path(".scratch/hmm-lush/Training/set11a_dInt/set11a_dInt.sds/0.seq")

raw = np.array(CORPUS.read_text().split(), dtype=np.int64)
assert raw.min() >= 0 and raw.max() < A, (raw.min(), raw.max())
RECORDS = [SequenceRecord(raw + USER_BASE)]
VOCAB = SymbolTable([f"s{i}" for i in range(A)])
rng = np.random.default_rng(20260916)


def random_params(S):
    out = np.zeros((S, S, USER_BASE + A))
    out[:, :, USER_BASE:] = rng.dirichlet(np.ones(A), size=(S, S))
    return HMMParams(
        rng.dirichlet(np.ones(S)), rng.dirichlet(np.ones(S), size=S), out, VOCAB
    )


def split_arrays(p, k):
    """`split-state`'s shape: a partner for `k` with its inbound and initial mass halved."""
    S = p.n_states
    init = np.append(p.init_state_p, p.init_state_p[k] / 2)
    init[k] /= 2
    t = np.zeros((S + 1, S + 1))
    t[:S, :S] = p.transition_p
    t[:S, S] = t[:S, k] / 2
    t[:S, k] /= 2
    t[S, :] = t[k, :]
    o = np.zeros((S + 1, S + 1, p.n_symbols))
    o[:S, :S] = p.output_p
    o[:S, S] = o[:S, k]
    for j in range(S + 1):
        o[S, j, USER_BASE:] = rand_p_vector(A, 0.1, rng)
    return init, t, o


def merge_arrays(p, i, j):
    """`merge-states`' shape: `j` folded into `i`, outbound rows weighted by `state_p`."""
    w = p.state_p[[i, j]]
    w = w / w.sum()
    keep = [s for s in range(p.n_states) if s != j]
    init = p.init_state_p.copy()
    init[i] += init[j]
    init = init[keep]
    t = p.transition_p.copy()
    t[:, i] += t[:, j]
    t[i] = w[0] * t[i] + w[1] * t[j]
    t = t[np.ix_(keep, keep)]
    t /= t.sum(axis=1, keepdims=True)
    o = p.output_p.copy()
    o[:, i] = o[:, i] + o[:, j]
    o[i] = w[0] * o[i] + w[1] * o[j]
    o = o[np.ix_(keep, keep)]
    user = o[..., USER_BASE:]
    total = user.sum(axis=-1, keepdims=True)
    o[..., USER_BASE:] = np.divide(user, total, out=np.zeros_like(user), where=total > 0)
    return init, t, o


def timed(fn, reps):
    samples = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result


def table_python():
    print("\n[1] python backend: where one move's time goes")
    print(
        f"{'S':>3} {'move':>5} | {'arrays':>8} {'HMMParams':>9} {'state_p':>8} | "
        f"{'fwd pass':>8} {'total@d':>8} {'suggest_d':>9} | "
        f"{'bw/cycle':>8} {'bw conv':>8} {'cycles':>6} | {'build/score':>11}"
    )
    for S in (5, 16, 50):
        incumbent = baum_welch(random_params(S), RECORDS, max_cycles=20).params
        for move in ("split", "merge"):
            if move == "split":
                make = lambda: split_arrays(incumbent, 0)  # noqa: E731
            else:
                make = lambda: merge_arrays(incumbent, 0, 1)  # noqa: E731
            t_arrays, arrays = timed(make, 30)
            t_ctor, cand = timed(lambda: HMMParams(*arrays, VOCAB), 30)
            t_with_state_p, _ = timed(lambda: HMMParams(*arrays, VOCAB).state_p, 30)
            t_state_p = t_with_state_p - t_ctor
            t_fwd, _ = timed(
                lambda: _corpus_description_length(
                    cand.init_state_p, cand.transition_p, cand.output_p, RECORDS
                ),
                5,
            )
            t_total, _ = timed(lambda: _total_description_length(cand, RECORDS, 100.0), 5)
            t_suggest, _ = timed(lambda: _suggest_d(cand, RECORDS), 1)
            t0 = time.perf_counter()
            result = baum_welch(cand, RECORDS, max_cycles=500)
            t_bw = time.perf_counter() - t0
            build = t_arrays + t_ctor + t_state_p
            score = t_bw + t_suggest
            print(
                f"{S:>3} {move:>5} | {t_arrays * 1e3:6.2f}ms {t_ctor * 1e3:7.2f}ms "
                f"{t_state_p * 1e3:6.2f}ms | {t_fwd * 1e3:6.1f}ms {t_total * 1e3:6.1f}ms "
                f"{t_suggest:8.2f}s | {t_bw / max(result.cycles, 1) * 1e3:6.1f}ms "
                f"{t_bw:7.2f}s {result.cycles:>6} | {build / score:11.2e}"
            )


def table_cython():
    if not any(s.name == "cython" and s.available for s in backends("baum_welch")):
        print("\n[2] cython backend unavailable on this host; skipped")
        return
    print("\n[2] cython backend: does a faster scorer make building matter?")
    print(
        f"{'S':>3} | {'build':>8} | {'cycle py':>9} {'cycle cy':>9} {'speedup':>7} | "
        f"{'cy cycle / build':>16} | {'suggest_d':>9}"
    )
    for S in (5, 50, 150):
        incumbent = random_params(S)
        t_build, cand = timed(
            lambda: (lambda a: HMMParams(*a, VOCAB))(split_arrays(incumbent, 0)), 20
        )
        t0 = time.perf_counter()
        r_py = baum_welch(cand, RECORDS, max_cycles=10, backend="python")
        t_py = (time.perf_counter() - t0) / max(r_py.cycles, 1)
        baum_welch(cand, RECORDS, max_cycles=10, backend="cython")  # warm
        t0 = time.perf_counter()
        r_cy = baum_welch(cand, RECORDS, max_cycles=10, backend="cython")
        t_cy = (time.perf_counter() - t0) / max(r_cy.cycles, 1)
        t_suggest, _ = timed(lambda: _suggest_d(cand, RECORDS), 1)
        print(
            f"{S:>3} | {t_build * 1e3:6.2f}ms | {t_py * 1e3:7.1f}ms {t_cy * 1e3:7.1f}ms "
            f"{t_py / t_cy:6.1f}x | {t_cy / t_build:15.0f}x | {t_suggest:8.2f}s"
        )


if __name__ == "__main__":
    import sys

    print(
        f"host: {platform.processor() or platform.machine()}, "
        f"python {platform.python_version()}, numpy {np.__version__}; "
        f"corpus {raw.size} symbols over {A}"
    )
    if "--cython-only" not in sys.argv:
        table_python()
    table_cython()
