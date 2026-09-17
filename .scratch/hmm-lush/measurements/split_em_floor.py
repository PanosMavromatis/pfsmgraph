"""How many EM cycles a split candidate needs before its score can be trusted.

Evidence for ADR 0023 section 6 (`feat/hmm-search-loop`, goal 2), which sizes the minimum
EM budget ADR 0022 section 4 obliges the search to give a split. For every split trial
`(state, trial)` of the tracked `set02a_200` models, the same candidate (one generator per
trial, keyed `SeedSequence(ENTROPY, spawn_key=(state, trial))`) is re-converged by
`_try_split` under each floor in `FLOORS`, on `cython`, with the corpus as one record.

Each cell reads `floor:Δtotal/Δdata(cycles)` against the 800-cycle run, which stands in for
"enough". A data difference of about 10 bits or more means the twins had not separated when
EM stopped. `cycles` above the floor means the ordinary stopping rule carried the run on by
itself.

`d` is chosen by `_suggest_d` (Brent) inside `_try_split`, so a total can move by more than
its data half through Brent's local minima as well as through rounding. The spread of totals
is therefore re-measured once the search chooses `d` by the bounded scan (ADR 0023, Open).

Imports private modules and the test suite's fixture reader, as the other measurement
scripts do; nothing under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/split_em_floor.py

which takes about ten minutes on a 4-vCPU host, or add `--quick` for the one- and five-state
models alone. Model directory names may be given instead, e.g. `m008_0001_008.hmm`.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, SAVED_MODELS, load_corpus_record, load_params  # noqa: E402

from pfsmgraph.hmm._trials import _try_split  # noqa: E402

FLOORS = (0, 30, 60, 100, 200, 400, 800)
ENTROPY = 20260917
TRIALS_PER_STATE = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    models = args.models or (SAVED_MODELS[:2] if args.quick else SAVED_MODELS)

    records = [load_corpus_record()]
    reference = FLOORS[-1]
    for name in models:
        params = load_params(FIXTURES / name)
        for state in range(params.n_states):
            for trial in range(TRIALS_PER_STATE):
                runs = {}
                for floor in FLOORS:
                    rng = np.random.default_rng(np.random.SeedSequence(ENTROPY, spawn_key=(state, trial)))
                    started = time.perf_counter()
                    result = _try_split(
                        params, records, state, rng=rng, min_cycles=floor, backend="cython"
                    )
                    runs[floor] = (result, time.perf_counter() - started)
                ref = runs[reference][0]
                cells = " ".join(
                    f"{floor}:{result.total_bits - ref.total_bits:+.2f}/"
                    f"{result.data_bits - ref.data_bits:+.2f}({result.cycles}c,{seconds:.1f}s)"
                    for floor, (result, seconds) in runs.items()
                )
                print(
                    f"{name} split ({state}, {trial}): total at {reference}={ref.total_bits:.2f} | {cells}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
