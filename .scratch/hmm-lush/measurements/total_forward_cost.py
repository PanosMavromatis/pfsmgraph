"""What the forward pass inside the total description length costs, per backend.

Evidence for ADR 0023 section 7 (`feat/hmm-search-loop`, goal 2). `merge_round_cost.py`
found `_suggest_d` taking 53-62% of every trial, because each of its evaluations runs
`_corpus_description_length`, which calls the numpy `_forward_backward` directly and takes
no `backend=`. The compiled kernels already exist in `_backends._TABLE["forward_backward"]`
and are held bit-exact to numpy, so passing a backend through is plumbing, not a new kernel.

For the smallest and largest tracked `set02a_200` models, this times one forward-backward
over the corpus as one record (N = 1268) per backend that runs here, and checks that every
backend's scale factors, the only output the description length reads, are bit-identical to
numpy's. Timings are host-specific; the conclusion rests on the ratio and the identity.

Imports private modules and the test suite's fixture reader, as the other measurement
scripts do; nothing under `packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/total_forward_cost.py
"""

from __future__ import annotations

import platform
import sys
import timeit

sys.path.insert(0, "packages/pfsmgraph-hmm/tests")
from _lush_fixtures import FIXTURES, load_corpus_record, load_params  # noqa: E402

from pfsmgraph.hmm._backends import _resolve, _status  # noqa: E402

MODELS = ("m001_0001_001.hmm", "m008_0001_008.hmm")


def main():
    print(f"host: {platform.processor() or platform.machine()}, python {platform.python_version()}")
    record = load_corpus_record()
    available = [status.name for status in _status("forward_backward") if status.available]
    for name in MODELS:
        params = load_params(FIXTURES / name)
        arrays = (params.init_state_p, params.transition_p, params.output_p, record.codes)
        reference = None
        for backend in available:
            kernel = _resolve("forward_backward", backend)
            scale = kernel(*arrays)[2]
            reference = scale.tobytes() if reference is None else reference
            number, total = timeit.Timer(lambda kernel=kernel: kernel(*arrays)).autorange()
            print(
                f"{name} S={params.n_states} {backend}: {1000 * total / number:.2f} ms, "
                f"scale bit-identical to python: {scale.tobytes() == reference}",
                flush=True,
            )


if __name__ == "__main__":
    main()
