"""How the cost of one `suggest-merge` round grows with model size, and where it goes.

Evidence for the scored primitives (`feat/hmm-scored-primitives`, goal 5). The master
plan asks for the all-pairs merge cost to be recorded, because a merge round tries
every pair of states and is therefore quadratic in their number, and that cost is
what the deferred alignment seed attacks at its input rather than in its inner loop.

For each `S`, an incumbent is a seeded random start converged by `baum_welch` on
`cython`. The script times one whole `_suggest_merge` round on `cython`, then times
the same pairs again phase by phase: building the candidate (`_merge_states`),
re-converging it (`baum_welch`), choosing `d` (`_suggest_d`, which takes no
`backend=` and runs the numpy forward pass), and the final total. The phase sums
should add up to the round, which is the check that no phase was missed.

The corpus is `set11a_dInt`, 1449 symbols over 25 as one record, the one
`param_representation_move_cost.py` and the master plan's figures were measured on.

Imports private modules, as the other measurement scripts do; nothing under
`packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/merge_round_cost.py

and add `--quick` to stop at `S = 8`. Timings are host-specific, and the header names
the host they were taken on; the conclusion rests on the slope and the phase shares.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable
from pfsmgraph.hmm import HMMParams, backends, baum_welch
from pfsmgraph.hmm._mdl import _suggest_d, _total_description_length
from pfsmgraph.hmm._numeric import closed_classes
from pfsmgraph.hmm._topology import _merge_states
from pfsmgraph.hmm._trials import _suggest_merge

A = 25
CORPUS = Path(".scratch/hmm-lush/Training/set11a_dInt/set11a_dInt.sds/0.seq")
BACKEND = "cython"

raw = np.array(CORPUS.read_text().split(), dtype=np.int64)
assert raw.min() >= 0 and raw.max() < A, (raw.min(), raw.max())
RECORDS = [SequenceRecord(raw + USER_BASE)]
VOCAB = SymbolTable([f"s{i}" for i in range(A)])


def incumbent(S):
    rng = np.random.default_rng(20260917 + S)
    out = np.zeros((S, S, USER_BASE + A))
    out[:, :, USER_BASE:] = rng.dirichlet(np.ones(A), size=(S, S))
    start = HMMParams(rng.dirichlet(np.ones(S)), rng.dirichlet(np.ones(S), size=S), out, VOCAB)
    return baum_welch(start, RECORDS, backend=BACKEND).params


def phases(params):
    """The same pairs `_suggest_merge` tries, timed phase by phase."""
    spent = dict.fromkeys(("build", "baum_welch", "suggest_d", "total"), 0.0)
    transient = closed_classes(params.transition_p) < 0
    for a, b in combinations(range(params.n_states), 2):
        if transient[a] and transient[b]:
            continue
        t = time.perf_counter()
        candidate = _merge_states(params, a, b)
        spent["build"] += time.perf_counter() - t
        t = time.perf_counter()
        converged = baum_welch(candidate, RECORDS, backend=BACKEND).params
        spent["baum_welch"] += time.perf_counter() - t
        t = time.perf_counter()
        d = _suggest_d(converged, RECORDS)
        spent["suggest_d"] += time.perf_counter() - t
        t = time.perf_counter()
        _total_description_length(converged, RECORDS, d)
        spent["total"] += time.perf_counter() - t
    return spent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="stop at S = 8")
    sizes = (2, 4, 8) if parser.parse_args().quick else (2, 4, 8, 12, 16)

    assert any(s.name == BACKEND and s.available for s in backends("baum_welch")), (
        f"backend {BACKEND!r} is unavailable: {backends('baum_welch')}"
    )
    print(f"host: {platform.node()} · {platform.processor() or platform.machine()} · "
          f"{os.cpu_count()} CPUs · numpy {np.__version__}")
    print(f"corpus: {CORPUS.parent.name}, {len(raw)} symbols over {A}, one record; backend {BACKEND}\n")
    print(f"{'S':>3} {'pairs':>5} {'round s':>8} {'s/pair':>7} {'phases s':>8}"
          f" {'build':>6} {'EM':>6} {'d':>6} {'total':>6}")

    rows = []
    for S in sizes:
        params = incumbent(S)
        t = time.perf_counter()
        moves = _suggest_merge(params, RECORDS, backend=BACKEND)
        round_s = time.perf_counter() - t
        spent = phases(params)
        summed = sum(spent.values())
        pairs = len(moves)
        per_pair = round_s / pairs if pairs else float("nan")
        shares = {k: v / summed if summed else float("nan") for k, v in spent.items()}
        rows.append((S, pairs, round_s))
        print(f"{S:>3} {pairs:>5} {round_s:>8.2f} {per_pair:>7.3f} {summed:>8.2f}"
              f" {shares['build']:>6.1%} {shares['baum_welch']:>6.1%}"
              f" {shares['suggest_d']:>6.1%} {shares['total']:>6.1%}")

    fit = [(S, r) for S, pairs, r in rows if pairs]
    if len(fit) >= 2:
        slope_s = np.polyfit(np.log([S for S, _ in fit]), np.log([r for _, r in fit]), 1)[0]
        pair_counts = [p for _, p, _ in rows if p]
        slope_p = np.polyfit(np.log(pair_counts), np.log([r for _, r in fit]), 1)[0]
        print(f"\nlog-log slope of round time: {slope_s:.2f} against S, {slope_p:.2f} against pairs")


if __name__ == "__main__":
    main()
