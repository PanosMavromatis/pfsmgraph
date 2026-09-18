# Changelog

Changes to `pfsmgraph-hmm`, newest first. The public API is the names in
`pfsmgraph.hmm.__all__`; anything else, including every module whose name begins with an
underscore, is out of contract and may change in any release. The decisions behind each
change are in [`docs/design/adr/`](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/README.md).

## 0.3.0 — unreleased

**Topology search ships in this release, and there is no supported way to call it.**
Growing and shrinking a model by state split and merge, scored by minimum description
length, is implemented in the private modules `_search`, `_trials`, `_topology` and `_mdl`.
None of it is exported: `__all__` is the same ten names 0.2.0 shipped. It imports and runs,
but nothing promises that it will keep working, and its signatures may change without
notice. This is deliberate. `__all__` can widen in a later release at no cost to anyone,
but it can never narrow, and the scoring criterion is still an open research question
([ADR 0025](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/0025-topology-search-not-exported.md)).

### Added

- `baum_welch(..., score_backend=)` names the backend of the forward pass that the
  convergence check runs. Before this, the check always ran the pure-numpy pass, whatever
  `backend=` said. It takes the same names as `backend=` except `"torch"`, which it
  refuses, and it defaults to `"python"`. Every choice returns bit-identical results, so
  the keyword changes speed and never an answer.
- `baum_welch(..., min_cycles=)` holds the convergence rule off until at least that many
  cycles have run. It is for a start known to sit near a point where the rule stops too
  early, such as a newly split state. The default is `0`, which is the 0.2.0 rule.

### Fixed

- `HMMParams.state_p` now always raises `ValueError` for a transition matrix with more than
  one closed communicating class. In 0.2.0 the error depended on the linear solve coming
  out singular. A probability that is not exactly representable, such as 2/3, could keep
  it just off singular, and the solve then returned one of the chain's stationary
  distributions with no error. That happened in 18 of 254 random reducible chains. The
  classes are now found from which transitions are non-zero, before any solve, and the
  error message names them.
- `HMMParams.state_p` is exactly `0.0` on transient states. Before, it held solver rounding
  of up to `1.8e-14` in magnitude, some of it negative.

## 0.2.0 — 2026-09-16

### Added

- `backend=` on every public call, and `backends()`, `BackendStatus` and
  `BackendUnavailableError` to see which backends can run
  ([ADR 0021](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/0021-runtime-backend-selection.md)).
  The backends are pure Python/numpy, Cython, Numba CPU-parallel and Numba CUDA, and they
  return bit-identical results on one machine. The default is `"python"` whatever is
  installed. A backend that cannot run raises and never falls back to another.
- `viterbi_batch`, which decodes many records padded together. Row `i` of its result is
  bit-identical to `viterbi` on record `i`.
- `baum_welch` and `BaumWelchResult`: Baum-Welch training over a fixed topology, keeping
  each record's counts separate, with `batch_size=` and a `log=` text stream. It also has a
  `torch` backend, with `device=`, which is held to the reference within a measured
  tolerance rather than exactly.
- Optional extras: `cpu-parallel` (numba), `gpu` (numba-cuda, not on macOS) and `torch`.

### Changed

- The package is now published as platform wheels with the Cython kernels compiled: twenty
  wheels covering Linux x86_64 and aarch64 (glibc 2.17 and later), macOS arm64 and Windows
  x86_64, on CPython 3.10 to 3.14. It is built by GitHub Actions and published through
  PyPI Trusted Publishing with PEP 740 attestations. Other platforms build from the sdist,
  which needs a C compiler.

### Fixed

- The wheel now contains the license text. 0.1.0's wheel declared `License-Expression: MIT`
  but did not include the `LICENSE` file.

## 0.1.0 — 2026-09-13

The first release: frozen arc-emission model parameters and their Viterbi decode.

- `HMMParams`, a frozen parameter value. Emission is on the arc, `output_p[i, j, symbol]`,
  with a symbol axis that spans the whole `pfsmgraph-dataseq` vocabulary, reserved codes
  included. Derived quantities such as `state_p` are computed when read, never stored
  ([ADR 0015](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/0015-arc-emission-mealy-formulation.md),
  [ADR 0017](https://github.com/PanosMavromatis/pfsmgraph/blob/main/docs/design/adr/0017-frozen-parameter-object-for-hmm.md)).
- `viterbi` and `ViterbiPath`: the most probable state path of one record, as a
  description length in bits, with ties going to the lowest state index.
- `ImpossibleSequenceError`, a `ValueError`, raised for a record that no path can emit.
- A pure `py3-none-any` wheel; no backend could be selected yet.
