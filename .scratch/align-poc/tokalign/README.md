# tokalign

Sequence alignment library for arbitrary symbolic sequences with multi-character token support.

Unlike traditional bioinformatics alignment tools that operate on single-character amino acids or nucleotides, tokalign works with **any string symbols** — multi-character tokens like `"alpha"`, `"REST"`, `"V7"`, or even full words. This makes it suitable for aligning tokenized text, symbolic sequences, or any domain where the alignment alphabet extends beyond single characters.

## Status

This project is in early development (v0.1.0). The core type system is implemented and the first algorithm has Phase 0, Phase 1, and Phase 2 complete.

**What exists today:**

- `Alphabet` — bidirectional string-to-integer mapping; indices 0–2 reserved for neural-net interop (padding, BOS, EOS), gap at index 3, user symbols from 4
- `ScoringMatrix` — 2D numpy scoring matrix constructed from human-readable `(str, str) -> float` dicts, with affine gap penalty support (`gap_open` + `gap_extend`)
- `AlignmentResult` — alignment output with score, aligned sequences, identity calculation, and a `format()` method for readable multi-character symbol display
- `_backends` — centralised backend discovery with CUDA availability detection
- Test infrastructure with automatic backend discovery via `_backends`
- A plugin development guide for building algorithms through a four-phase lifecycle

**Algorithms:**

| Algorithm | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|---------|
| Needleman-Wunsch (global, affine) | done | done | done | — |

## Architecture

Each alignment algorithm progresses through four phases, starting with a formalization and then producing three executable backends:

0. **Formalization** (`FORMALIZATION.md`) — language-agnostic pseudocode specification adapted to tokalign's model
1. **Pure Python** (`_python.py`) — correctness-first reference implementation, mechanically translated from the formalization
2. **Cython** (`_cython.pyx`) — compiled for performance
3. **Numba CUDA** (`_numba.py`) — GPU-parallelized for scale

All three backends expose an identical `align()` function signature. Tests are parametrized across backends automatically — adding a new backend runs the full test suite against it with no extra configuration.

### Encode-at-the-boundary pattern

Symbols stay as human-readable strings at the API surface. At each `align()` entry point, the `Alphabet` encodes them to integer arrays for the DP computation:

```
User strings → Alphabet.encode_pair() → integer arrays → DP kernel → Alphabet.decode() → result strings
```

This separation keeps DP loops fast (integer-only) and makes the Python-to-Cython-to-GPU translation mechanical.

### Package layout

```
src/tokalign/
├── _types.py          # Alphabet, ScoringMatrix, AlignmentResult
├── _backends.py       # Backend discovery with CUDA availability check
├── scoring.py         # Scoring matrix construction & I/O
├── algorithms/
│   ├── _registry.py   # Maps algorithm names → backend implementations
│   └── <algorithm>/   # One directory per algorithm
│       ├── FORMALIZATION.md
│       ├── _python.py
│       ├── _cython.pyx
│       └── _numba.py
└── _ext/              # Compiled extension modules
```

## Installation

Requires Python 3.10+.

```bash
# Install with uv (recommended)
uv sync
uv pip install -e .

# Or with pip
pip install -e .
```

### Optional dependencies

```bash
# Development (testing + Cython compilation)
uv sync --extra dev

# Benchmarking (matplotlib for scaling plots)
uv sync --extra bench

# GPU support
uv sync --extra gpu
```

## Usage

```python
from tokalign._types import Alphabet, ScoringMatrix

# Define an alphabet of multi-character symbols
alphabet = Alphabet(symbols=("alpha", "beta", "gamma", "delta"))

# Create a simple identity scoring matrix
scoring = ScoringMatrix.identity(alphabet, match=2.0, mismatch=-1.0)

# Encode sequences to integer arrays
encoded = alphabet.encode(["alpha", "beta", "gamma"])
decoded = alphabet.decode(encoded)  # ["alpha", "beta", "gamma"]
```

## Running tests

```bash
uv run pytest                   # Full suite across all available backends
uv run pytest tests/ -v         # Verbose output
```

Cython backends require compilation first:

```bash
uv run python setup.py build_ext --inplace
```

## Benchmarking

Compare backend performance for a specific algorithm:

```bash
uv run python benchmarks/run_benchmark.py needleman_wunsch
```

The script auto-discovers available backends, runs timing benchmarks across sequence lengths (10–5000), and saves results to `benchmarks/results/<algorithm>/`:
- `results.md` — markdown table with mean +/- std timings
- `results.json` — raw data for programmatic use
- `scaling.png` — log-log scaling plot

Optional flags:
- `--memory` — include peak memory profiling (requires `memory-profiler`)
- `--lengths 10,100,1000` — custom sequence lengths
- `--reps 10` — number of timed repetitions per configuration
- `--seed 42` — random seed for reproducible sequence generation

The script will refuse to run if any backend is stale (prerequisite has a newer modification time) or if fewer than two backends are available.

## Claude Code plugin workflow

Plugin management for tokalign (and other projects) lives in a separate shared repo, [`claude-plugin-tools`][cpt]. This repo carries only the project-specific state:

- `dev/plugin-config.sh` — enabled plugins plus a self-locating `PLUGIN_DIR` pointing at `dev/plugins/` (committed; manage with `~/claude-plugin-tools/set-enabled.sh` rather than editing by hand).
- `.envrc` — exports `PLUGIN_CONFIG` so the tools scripts know where to find the config.
- `dev/plugins/` — populated by `~/claude-plugin-tools/pull.sh`, gitignored.

One-time setup on a new machine:

```bash
git clone git@github.com:PanosMavromatis/claude-plugin-tools.git ~/claude-plugin-tools
cd /path/to/tokalign
source .envrc                                  # or `direnv allow` if direnv is set up
~/claude-plugin-tools/pull.sh                  # clones enabled plugins into dev/plugins/
```

Daily use:

```bash
~/claude-plugin-tools/claude.sh                # launch claude with enabled plugins
~/claude-plugin-tools/claude.sh --resume       # trailing args forwarded to claude
```

See the tools repo's README for the full design (flag vs env-var, direnv pairing, migration notes).

[cpt]: https://github.com/PanosMavromatis/claude-plugin-tools

## GPU dev surface

Phase 3 (Numba CUDA) development runs on a GCE GPU dev VM with an L4 GPU, driven by the `tokalign-devbox` Docker image built from `Dockerfile.dev`. Operational details live in:

- [`docs/ops/`](docs/ops/README.md) — entry-point cheatsheet: every `dev/` script in calling order, with links into the runbooks below
- [`docs/ops/gpu-dev-vm-runbook.md`](docs/ops/gpu-dev-vm-runbook.md) — VM lifecycle, container bind-mounts, first-shell setup
- [`docs/ops/gce-vm-github-workflow.md`](docs/ops/gce-vm-github-workflow.md) — Mac-side SSH key setup, agent forwarding, GitHub auth on the VM, daily `git`/`gh` workflow
- [`docs/ops/gpu-dev-docker-runbook.md`](docs/ops/gpu-dev-docker-runbook.md) — image build, Artifact Registry push, architecture caveats
- [`docs/ops/gce-vm-docker-workflow.md`](docs/ops/gce-vm-docker-workflow.md) — Docker engine + NVIDIA Container Toolkit install on the VM, daily container lifecycle
- [`docs/ops/gce-vm-nvidia-checklist.md`](docs/ops/gce-vm-nvidia-checklist.md) — reference for the four-layer GPU stack (driver / runtime / NVRTC / Numba) and end-to-end verification

The motivating decisions are [ADR 0016](docs/decisions/adr/0016-phase-3-dev-on-gce-gpu-vm.md), [ADR 0017](docs/decisions/adr/0017-gpu-accelerator-and-region.md), and [ADR 0018](docs/decisions/adr/0018-artifact-registry-location.md).

### Configuration

`dev/env.sh` is sourced by every script under `dev/{ar,docker,gpu-vm,ssh}/` and sets the GCP project, region, registry, image names, and VM identifiers used across them (`PROJECT_ID`, `REGION`, `ZONE`, `REPO`, `REPO_LOCATION`, `DEV_IMAGE`, `RUN_IMAGE`, `VM_NAME`, `MACHINE_TYPE`, `ACCELERATOR`, `FALLBACK_ZONES`, plus the derived `REGISTRY` / `DEV_IMAGE_URI` / `RUN_IMAGE_URI`). Every value uses `${VAR:-default}`, so a one-off override flows through without editing the file:

```bash
REGION=us-west4 ./dev/gpu-vm/start.sh
```

Edit `dev/env.sh` if your defaults need to change permanently.

### Scripts

Thin wrappers over the `gcloud` and `docker` commands from the runbooks — read those first for context before running these.

**VM lifecycle** (run on the Mac):

```bash
./dev/gpu-vm/create.sh         # One-time: create tokalign-gpu-dev in us-central1-c
./dev/gpu-vm/start.sh          # Start the VM
./dev/gpu-vm/stop.sh           # Stop the VM (GPU-attached VMs bill while running)
./dev/gpu-vm/delete.sh         # Permanently delete the VM and its boot disk
```

**Capacity-error recovery** (when the primary zone is out of L4):

```bash
./dev/gpu-vm/create-with-fallback.sh         # First-time create: try each FALLBACK_ZONES entry in turn
TARGET_ZONE=us-west4-a ./dev/gpu-vm/migrate-zone.sh   # Existing VM: snapshot the boot disk and recreate it in TARGET_ZONE
```

**Image build + run**:

```bash
./dev/docker/local/build.sh    # Mac: buildx cross-build of tokalign-devbox for linux/amd64
./dev/docker/vm/run.sh         # VM: run the container with GPU passthrough and bind-mounts
```

## Decisions

Architectural decisions for both `tokalign` and the `tokalign-dev` plugin live together in [`docs/decisions/adr/`](docs/decisions/adr/). The two projects share a single ADR log by design: the plugin exists to enforce the package's architecture, so many decisions (particularly ADRs 0004–0015 on the algorithm lifecycle) simultaneously define what the package must do and what the plugin must guide. A split log would artificially separate decisions that are genuinely coupled.

## License

TBD
