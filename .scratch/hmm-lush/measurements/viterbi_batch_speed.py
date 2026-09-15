"""Time `viterbi_batch` against the per-record `viterbi` loop, per backend.

Evidence for the batched decode (`feat/hmm-batched-decode`, goal 5). For every
backend `backends("viterbi_batch")` reports available, each cell of a
`B x S` grid times one padded decode of `B` ragged records against the loop
`[viterbi(params, r, backend=b) for r in records]`, after asserting the two agree
bit for bit, so no figure can come from a wrong answer.

Imports the public `pfsmgraph.hmm` and `pfsmgraph.dataseq` only; nothing under
`packages/` imports this. Run from the repo root with

    uv run python .scratch/hmm-lush/measurements/viterbi_batch_speed.py

Timings are host-specific, and the header names the host they were taken on.
"""

import platform
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable

from pfsmgraph.hmm import HMMParams, backends, viterbi, viterbi_batch

BATCHES = (1, 16, 256)
STATES = (5, 64, 160)
LENGTHS = (200, 400)
N_SYMBOLS = 8
REPEATS = 3


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def host_header():
    import os

    lines = [f"CPU: {cpu_model()}, {os.cpu_count()} logical CPUs"]
    try:
        import numba

        lines.append(f"numba threads: {numba.get_num_threads()}")
    except ImportError:
        lines.append("numba: not installed")
    try:
        from numba import cuda

        if cuda.is_available():
            device = cuda.get_current_device()
            name = device.name.decode() if isinstance(device.name, bytes) else device.name
            lines.append(f"GPU: {name}")
        else:
            lines.append("GPU: none detected")
    except ImportError:
        lines.append("GPU: numba-cuda not installed")
    versions = []
    for dist in ("numpy", "numba", "numba-cuda", "pfsmgraph-hmm"):
        try:
            versions.append(f"{dist} {version(dist)}")
        except PackageNotFoundError:
            versions.append(f"{dist} absent")
    lines.append(", ".join(versions))
    return "\n".join(lines)


def model(rng, size):
    vocabulary = SymbolTable([f"s{i}" for i in range(N_SYMBOLS)])
    transition_p = rng.random((size, size))
    transition_p /= transition_p.sum(axis=1, keepdims=True)
    output_p = np.zeros((size, size, vocabulary.size))
    user = rng.random((size, size, N_SYMBOLS))
    output_p[:, :, USER_BASE:] = user / user.sum(axis=2, keepdims=True)
    init_p = rng.random(size)
    return HMMParams(init_p / init_p.sum(), transition_p, output_p, vocabulary)


def records(rng, count):
    return [
        SequenceRecord(
            rng.integers(USER_BASE, USER_BASE + N_SYMBOLS, int(rng.integers(*LENGTHS) + 1)).astype(np.int32)
        )
        for _ in range(count)
    ]


def best_of(call):
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    print(host_header())
    print(f"records ragged in [{LENGTHS[0]}, {LENGTHS[1]}], {N_SYMBOLS} user symbols, best of {REPEATS}\n")
    available = [s.name for s in backends("viterbi_batch") if s.available]
    print(f"{'backend':>12} | {'B':>4} | {'S':>4} | {'batch ms':>10} | {'loop ms':>10} | {'loop/batch':>10}")
    print("-" * 66)
    rng = np.random.default_rng(20260915)
    for size in STATES:
        params = model(rng, size)
        for count in BATCHES:
            recs = records(rng, count)
            for backend in available:
                batched = viterbi_batch(params, recs, backend=backend)  # warm-up, compiles
                looped = [viterbi(params, r, backend=backend) for r in recs]
                for a, b in zip(batched, looped):
                    assert np.array_equal(a.states, b.states), (backend, count, size)
                    assert np.float64(a.total_bits).tobytes() == np.float64(b.total_bits).tobytes()
                t_batch = best_of(lambda: viterbi_batch(params, recs, backend=backend))
                t_loop = best_of(lambda: [viterbi(params, r, backend=backend) for r in recs])
                print(
                    f"{backend:>12} | {count:>4} | {size:>4} | {t_batch * 1e3:>10.1f} | "
                    f"{t_loop * 1e3:>10.1f} | {t_loop / t_batch:>9.2f}x",
                    flush=True,
                )


if __name__ == "__main__":
    main()
