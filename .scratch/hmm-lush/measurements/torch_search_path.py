"""Whether a topology search on `backend="torch"` leaves the path the bit-identical phases take.

Evidence for ADR 0024 (`feat/hmm-convergence-backend`, goal 5), whose section 4 says torch
"can change the search's path" by inference from ADR 0023's Open item: near the optimum a
candidate's total steps by 58-71 bits when EM stops slightly elsewhere and the best `d`
moves, and torch's E-step is held only to ADR 0020's tolerance. This measures it on the
tracked fixtures instead, with `score_backend="cython"` on both sides, so the scan and every
convergence check are bit-identical and only the E-step differs.

Part 1, rounds: one `_suggest_move` round from each saved model (`m001_0001_001`,
`m001_0005_005`, `m008_0001_008`: 1, 5 and 8 states), seed `SeedSequence(1, spawn_key=(1,
0))`, split floor 200. Candidates are matched by identity `(kind, states, trial)`, so the
comparison is between the same candidate trained on each backend: its `d`, the EM cycles
it took to converge, unrounded data length and total, and which backend scored it lower;
then the round's cycles summed over its candidates, and whether the ranking and the winner
agree.

Part 2, path: `_search` from the one-state model over the corpus's vocabulary, seed
`SeedSequence(20260917)` as `test_search.py` uses, for the 3 rounds the Lush training log
of `m001_0005_005` records, reporting each round's move on both backends, its EM cycles and
which scored lower, the first round that differs, and the best total each search returns.

The corpus is `set02a_200` as one record (N = 1268). Torch runs on the CPU, since `_search`
takes no `device=`, at about 200 ms per EM cycle against Cython's 0.5, so the whole run
takes an hour or more; `--part rounds` or `--part path` runs one half, and `--models`
names which saved models the rounds use.

Imports private modules and the test suite's fixture reader, as the other measurement
scripts do; nothing under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/torch_search_path.py
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, load_corpus_record, load_params  # noqa: E402

from pfsmgraph.hmm._search import _search  # noqa: E402
from pfsmgraph.hmm._trials import _suggest_move  # noqa: E402

MODELS = ("m001_0001_001.hmm", "m001_0005_005.hmm", "m008_0001_008.hmm")
ROUND_SEED = np.random.SeedSequence(1, spawn_key=(1, 0))
SEARCH_SEED = np.random.SeedSequence(20260917)
BACKENDS = ("cython", "torch")


def _identity(move):
    return move.kind, move.states, move.trial


def _lower(a, b):
    """Which backend scored lower: ``cython``, ``torch`` or ``tie``."""
    return "tie" if a == b else "cython" if a < b else "torch"


def _rounds(records, models):
    for name in models:
        params = load_params(FIXTURES / name)
        ranked = {}
        for backend in BACKENDS:
            start = time.perf_counter()
            ranked[backend] = _suggest_move(
                params, records, seed=ROUND_SEED, split_min_cycles=200,
                backend=backend, score_backend="cython",
            )
            print(f"{name} S={params.n_states} {backend}: {len(ranked[backend])} moves in "
                  f"{time.perf_counter() - start:.0f} s", flush=True)
        ours = {_identity(m): m for m in ranked["torch"]}
        d_changed = cycles_changed = 0
        worst_data = worst_total = 0.0
        cycles = {"cython": 0, "torch": 0}
        lower = {"cython": 0, "torch": 0, "tie": 0}
        for reference in ranked["cython"]:
            move = ours[_identity(reference)]
            a, b = reference.result, move.result
            if a is None or b is None:
                print(f"  {_identity(reference)}: impossible on "
                      f"{'both' if a is None and b is None else 'one backend only'}")
                continue
            d_changed += a.d != b.d
            cycles_changed += a.cycles != b.cycles
            cycles["cython"] += a.cycles
            cycles["torch"] += b.cycles
            lower[_lower(a.total_bits, b.total_bits)] += 1
            worst_data = max(worst_data, abs(a.data_bits - b.data_bits))
            worst_total = max(worst_total, abs(a.total_bits - b.total_bits))
            print(f"  {_identity(reference)}: d {a.d:g}/{b.d:g}, cycles {a.cycles}/{b.cycles}, "
                  f"data {a.data_bits - b.data_bits:+.3g}, total {a.total_bits:.4f}/{b.total_bits:.4f}, "
                  f"lower {_lower(a.total_bits, b.total_bits)}")
        order = [_identity(m) for m in ranked["cython"]] == [_identity(m) for m in ranked["torch"]]
        finite = sorted(m.total_bits for m in ranked["cython"] if m.total_bits < np.inf)
        gap = finite[1] - finite[0] if len(finite) > 1 else float("nan")
        winners = ranked["cython"][0], ranked["torch"][0]
        print(f"  cycles to converge, summed over candidates: cython {cycles['cython']}, "
              f"torch {cycles['torch']}; lower total: cython {lower['cython']}, "
              f"torch {lower['torch']}, tie {lower['tie']}; winners' totals "
              f"{winners[0].total_bits:.4f}/{winners[1].total_bits:.4f}, lower "
              f"{_lower(winners[0].total_bits, winners[1].total_bits)}")
        print(f"  summary: d differs in {d_changed}, cycles in {cycles_changed}, "
              f"max |data| {worst_data:.3g} bits, max |total| {worst_total:.3g} bits; "
              f"same ranking {order}, same winner "
              f"{_identity(ranked['cython'][0]) == _identity(ranked['torch'][0])}, "
              f"cython winner's lead {gap:.3g} bits", flush=True)


def _path(records):
    vocabulary = load_params(FIXTURES / "m001_0001_001.hmm").vocabulary
    results = {}
    for backend in BACKENDS:
        start = time.perf_counter()
        results[backend] = _search(
            records, start=vocabulary, seed=SEARCH_SEED, max_rounds=3, patience=3,
            backend=backend, score_backend="cython",
        )
        print(f"path {backend}: {len(results[backend].rounds)} rounds in "
              f"{time.perf_counter() - start:.0f} s", flush=True)
    a, b = results["cython"], results["torch"]
    print(f"  start: total {a.start.total_bits:.4f}/{b.start.total_bits:.4f}, d {a.start.d:g}/{b.start.d:g}, "
          f"cycles {a.start.cycles}/{b.start.cycles}")
    first = None
    for x, y in zip(a.rounds, b.rounds):
        same = _identity(x.move) == _identity(y.move)
        first = first if first is not None or same else x.index
        print(f"  round {x.index}: {_identity(x.move)}/{_identity(y.move)}, "
              f"total {x.move.total_bits:.4f}/{y.move.total_bits:.4f}, "
              f"d {x.move.result.d:g}/{y.move.result.d:g}, "
              f"cycles {x.move.result.cycles}/{y.move.result.cycles}, "
              f"lower {_lower(x.move.total_bits, y.move.total_bits)}, same move {same}")
    print(f"  first differing round: {first}; stop {a.stop}/{b.stop}; best total "
          f"{a.best.total_bits:.4f}/{b.best.total_bits:.4f}, lower "
          f"{_lower(a.best.total_bits, b.best.total_bits)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("all", "rounds", "path"), default="all")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    arguments = parser.parse_args()
    part = arguments.part
    print(f"host: {platform.processor() or platform.machine()}, {os.cpu_count()} CPUs, "
          f"python {platform.python_version()}")
    records = [load_corpus_record()]
    if part in ("all", "rounds"):
        _rounds(records, arguments.models)
    if part in ("all", "path"):
        _path(records)


if __name__ == "__main__":
    main()
