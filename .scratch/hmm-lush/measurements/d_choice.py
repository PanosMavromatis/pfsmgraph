"""How the topology search should choose the quantization resolution `d`.

Evidence for ADR 0023 section 5 (`feat/hmm-search-loop`, goal 2). `_suggest_d` reproduces
the original's `suggest-d`, Brent's method on `[1, 10000]` from 3821, and keeps the local
minimum it slides into. This script measures three things on the three tracked `set02a_200`
models, with the corpus as one record:

1. **The shape of `total(d)`.** Every integer `d` in `[1, 300]`, then a log grid to 10000:
   how many local minima there are, where the global one is, and how the data and model
   halves move at large `d`.
2. **The assumption the bounded scan rests on.** The minimum over `d <= 300` of
   `data(d) - unrounded`, where `unrounded` is the data length of the parameters as trained.
   The scan stops correctly only while that is non-negative.
3. **The bounded exact scan against Brent.** The scan tries `d = 1, 2, ...` and stops once
   `unrounded + model_lower_bound(d)` reaches the best total, where the bound is monotone in
   `d`. Reported: the `d` each chooses, its total, and each one's evaluation count.

The model bound is `int(S) + int(d) + (1 + S)·comb(d, 1 + S)`, plus `S·comb(d, 1 + A)` once
`d >= S`: every transition row's largest entry is at least `1/S`, so `p·d >= 1` and rounding
cannot zero it, which leaves at least `S` live arcs.

Every total here runs the numpy forward pass, since the description-length functions take
no `backend=` yet, so one evaluation costs about 30-37 ms and the run takes a few minutes.

Imports private modules and the test suite's fixture reader, as the other measurement
scripts do; nothing under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/d_choice.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params  # noqa: E402

from pfsmgraph.dataseq import USER_BASE  # noqa: E402
from pfsmgraph.hmm._mdl import (  # noqa: E402
    D_HIGH,
    D_LOW,
    D_START,
    _comb_code_length,
    _corpus_description_length,
    _data_description_length,
    _int_code_length,
    _minimize_int,
    _model_description_length,
    _total_description_length,
)

EXHAUSTIVE_UP_TO = 300


def model_lower_bound(params, d):
    """A lower bound on the model half, monotone in ``d``."""
    s, a = params.n_states, params.n_symbols - USER_BASE
    bound = _int_code_length(s) + _int_code_length(d) + (1 + s) * _comb_code_length(d, 1 + s)
    return bound + (s * _comb_code_length(d, 1 + a) if d >= s else 0.0)


def bounded_scan(params, records, unrounded):
    """Returns ``(d, total, evaluations)``: the global integer minimum if the assumption holds."""
    best_d, best, evaluations = None, np.inf, 0
    for d in range(int(D_LOW), int(D_HIGH) + 1):
        if unrounded + model_lower_bound(params, float(d)) >= best:
            break
        total = _total_description_length(params, records, float(d))
        evaluations += 1
        if total < best:
            best_d, best = d, total
    return best_d, best, evaluations


def main():
    records = [load_corpus_record()]
    grid = sorted(set(np.round(np.logspace(np.log10(EXHAUSTIVE_UP_TO), 4, 40)).astype(int).tolist()))
    for name in SAVED_MODELS:
        params = load_params(FIXTURES / name)
        unrounded = _corpus_description_length(
            params.init_state_p, params.transition_p, params.output_p, records
        )

        brent_calls = []

        def objective(d, params=params, calls=brent_calls):
            calls.append(d)
            return _total_description_length(params, records, d)

        brent_d, _ = _minimize_int(objective, D_LOW, D_START, D_HIGH)
        brent_total = _total_description_length(params, records, brent_d)

        started = time.perf_counter()
        small = {
            d: _total_description_length(params, records, float(d))
            for d in range(1, EXHAUSTIVE_UP_TO + 1)
        }
        per_evaluation = (time.perf_counter() - started) / EXHAUSTIVE_UP_TO
        large = {d: _total_description_length(params, records, float(d)) for d in grid}
        global_d = min(small, key=small.get)
        local_minima = sum(
            1
            for d in range(2, EXHAUSTIVE_UP_TO)
            if small[d] < small[d - 1] and small[d] <= small[d + 1]
        )
        slack = min(
            _data_description_length(params, records, float(d)) - unrounded
            for d in range(1, EXHAUSTIVE_UP_TO + 1)
        )
        scan_d, scan_total, scan_evaluations = bounded_scan(params, records, unrounded)

        print(f"{name}: S={params.n_states}, {per_evaluation * 1000:.1f} ms per total")
        print(
            f"  shape: {local_minima} local minima on [1, {EXHAUSTIVE_UP_TO}], global d={global_d} "
            f"({small[global_d]:.3f}); best beyond {EXHAUSTIVE_UP_TO}: {min(large.values()):.3f}"
        )
        print(f"  assumption: min data(d) - unrounded over d <= {EXHAUSTIVE_UP_TO} = {slack:+.4f} bits")
        print(
            f"  brent: d={brent_d:.0f} total={brent_total:.3f} in {len(brent_calls)} evaluations, "
            f"{brent_total - small[global_d]:.3f} bits above the global minimum"
        )
        print(f"  bounded scan: d={scan_d} total={scan_total:.3f} in {scan_evaluations} evaluations")
        for d in (global_d, 300, 1000, 3821, 10000):
            print(
                f"    d={d}: data={_data_description_length(params, records, float(d)):.3f} "
                f"model={_model_description_length(params, float(d)):.3f}"
            )


if __name__ == "__main__":
    main()
