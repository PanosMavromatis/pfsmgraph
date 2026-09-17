"""Why a split candidate's total moves far more than its data length with EM's budget.

Evidence for ADR 0023's Open item on an acceptance margin (`feat/hmm-search-loop`, goal 4).
`split_em_floor.py` found that on `m008_0001_008`, at floors of 200 cycles or more, one
candidate's total moved by up to 72 bits while its unrounded data length moved by at most
2.4, and that choosing `d` by the bounded scan rather than Brent left all 16 trials
bit-identical. This decomposes the two largest cases at floors 200, 400 and 800: the `d`
the scan chose, the arcs surviving quantization, the rounded data length, the model length
and the total.

The answer it gives: the best integer `d` steps (6 to 5, 5 to 4) as EM moves the parameters
slightly, and each step changes the model half by one `d`-increment of its cost, 58-71 bits
at nine states, while the rounded data length moves by 2-3. The total is a step function of
the converged parameters, so rankings closer than one step are decided by where EM stopped.

Imports private modules and the test suite's fixture reader, as the other measurement
scripts do; nothing under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/split_total_steps.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, load_corpus_record, load_params  # noqa: E402

from pfsmgraph.hmm._mdl import _data_description_length, _model_description_length, _quantize  # noqa: E402
from pfsmgraph.hmm._trials import _try_split  # noqa: E402

ENTROPY = 20260917
CASES = ((2, 0), (0, 0))
FLOORS = (200, 400, 800)


def main():
    params = load_params(FIXTURES / "m008_0001_008.hmm")
    records = [load_corpus_record()]
    for state, trial in CASES:
        for floor in FLOORS:
            rng = np.random.default_rng(np.random.SeedSequence(ENTROPY, spawn_key=(state, trial)))
            result = _try_split(
                params, records, state, rng=rng, min_cycles=floor,
                backend="cython", score_backend="cython",
            )
            arcs = np.count_nonzero(_quantize(result.params.transition_p, result.d))
            rounded = _data_description_length(result.params, records, result.d, backend="cython")
            model = _model_description_length(result.params, result.d)
            print(
                f"split ({state}, {trial}) floor {floor}: d={result.d:.0f} arcs={arcs} "
                f"data(d)={rounded:.2f} model={model:.2f} total={result.total_bits:.2f} "
                f"unrounded={result.data_bits:.2f}"
            )


if __name__ == "__main__":
    main()
