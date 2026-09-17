"""Where one round of the topology search spends its time, before and after ADR 0024 section 2.

Evidence for ADR 0024 (`feat/hmm-convergence-backend`, goal 4). A profile of one
`_suggest_move` round on `m001_0005_005` found 84% of it in the numpy forward pass that
`baum_welch`'s convergence check ran whatever `backend=` said. Since section 2 the check
runs on `score_backend`. The profile's original script was an untracked session file, so
this one rewrites it with the same setup: the `set02a_200` corpus as one record (N = 1268),
`backend="cython"`, `score_backend="cython"`, `split_min_cycles=200`, two split trials per
state, and seed `SeedSequence(1, spawn_key=(1, 0))`, so 10 merges and 10 splits.

"before" rebuilds the pre-section-2 behaviour by patching `baum_welch`'s check to ignore
its phase, which is exactly the one call site section 2 changed; "after" is the code as it
stands. Each configuration is timed once plainly and once under `cProfile`, which reports
the cumulative time in the numpy `_forward_backward` and in `_corpus_step`. Both must rank
the same moves with bit-identical totals, since the phases are bit-identical (ADR 0020).

Imports private modules and the test suite's fixture reader, as the other measurement
scripts do; nothing under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/search_round_profile.py

Timings are host-specific, and the header names the host; the conclusion rests on the
share of the round each configuration spends in the numpy forward pass.
"""

from __future__ import annotations

import cProfile
import os
import platform
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, load_corpus_record, load_params  # noqa: E402

import pfsmgraph.hmm._baum_welch as baum_welch_module  # noqa: E402
from pfsmgraph.hmm import _forward_backward  # noqa: E402
from pfsmgraph.hmm._mdl import _corpus_description_length  # noqa: E402
from pfsmgraph.hmm._trials import _suggest_move  # noqa: E402

MODEL = "m001_0005_005.hmm"
SEED = np.random.SeedSequence(1, spawn_key=(1, 0))


def _round(params, records):
    return _suggest_move(
        params, records, seed=SEED, split_min_cycles=200,
        backend="cython", score_backend="cython",
    )


def _numpy_check(init_state_p, transition_p, output_p, records, *, backend="python"):
    """The check before ADR 0024 section 2: the reference phase whatever it is asked."""
    return _corpus_description_length(init_state_p, transition_p, output_p, records)


def _measure(label, params, records):
    start = time.perf_counter()
    moves = _round(params, records)
    wall = time.perf_counter() - start

    profiler = cProfile.Profile()
    profiler.enable()
    profiled = _round(params, records)
    profiler.disable()
    stats = pstats.Stats(profiler)
    total = stats.total_tt

    def cumulative(module, name):
        for (filename, _, function), row in stats.stats.items():
            if function == name and filename == module.__file__:
                return row[1], row[3]  # call count, cumulative seconds
        return 0, 0.0

    fb_calls, fb_time = cumulative(_forward_backward, "_forward_backward")
    step_calls, step_time = cumulative(baum_welch_module, "_corpus_step")
    print(
        f"{label}: {len(moves)} moves in {wall:.2f} s ({total:.2f} s under cProfile); "
        f"numpy _forward_backward {fb_time:.2f} s over {fb_calls} calls "
        f"({100 * fb_time / total:.0f}%); _corpus_step {step_time:.2f} s over {step_calls} calls",
        flush=True,
    )
    assert [m.total_bits for m in moves] == [m.total_bits for m in profiled]
    return moves


def main():
    print(
        f"host: {platform.processor() or platform.machine()}, {os.cpu_count()} CPUs, "
        f"python {platform.python_version()}"
    )
    params, records = load_params(FIXTURES / MODEL), [load_corpus_record()]

    real = baum_welch_module._corpus_description_length
    baum_welch_module._corpus_description_length = _numpy_check
    try:
        before = _measure("before (check on numpy)", params, records)
    finally:
        baum_welch_module._corpus_description_length = real
    after = _measure("after (check on score_backend)", params, records)

    same = [(m.kind, m.states, m.trial, m.total_bits) for m in before] == [
        (m.kind, m.states, m.trial, m.total_bits) for m in after
    ]
    print(f"same ranked moves and bit-identical totals: {same}")
    assert same


if __name__ == "__main__":
    main()
