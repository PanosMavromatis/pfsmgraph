# 0008. Build backends are per-package; meson-python for compiled members

- **Status:** Accepted — but see [ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md),
  which suspends the meson-python half of this decision in practice until the first
  Cython kernel lands.
- **Date:** 2026-06-29
- **Source:** PRD §6, §6.1 — decisions D7, D8

## Context

The three-phase lifecycle of [ADR 0002](0002-three-phase-algorithm-lifecycle.md) applies
**only where dynamic programming is involved**. That makes the family heterogeneous in
build needs, and the heterogeneity is structural rather than incidental:

| Package | DP / wavefront | Compiled extensions | GPU mechanism |
|---|---|---|---|
| `dataseq` | none expected | none expected | stock PyTorch `DataLoader`; no subclassing |
| `align` | pervasive | Cython + Numba CUDA | `numba-cuda`, optional extra |
| `hmm` | Baum-Welch core only | Cython (partial; topology search likely plain Python) | `numba-cuda`, optional extra |
| `hseg` | unknown — TBD | possibly none | n/a |
| `dl` | none | none | PyTorch |

Two packages need a compiler, a build system, and a rebuild-on-edit story. Three do not
and should not pay for one.

The proof-of-concept used `setuptools.build_meta` (`setuptools>=75`, `cython>=3.0`,
`numpy>=2.1` at build time, `language_level="3"`, directives as function-scoped
`@cython` decorators in the `.pyx`). Bringing that code into the workspace is the
natural moment to reconsider, and it is also the *cheapest possible* moment: the entire
compiled surface is one example Needleman-Wunsch extension with zero shipped algorithms.

## Decision

**Two decisions, at different levels.**

**D7 — the principle: each package declares its own build backend.** The family is
build-heterogeneous by design. There is no family-wide backend, and none is sought. The
uv workspace ([ADR 0006](0006-single-repository-as-a-uv-workspace.md)) permits this
directly: each member's `pyproject.toml` carries its own `[build-system]`.

**D8 — the choice: meson-python for compiled members, hatchling for pure-Python ones.**
`align` and the Baum-Welch slice of `hmm` use meson-python; `dataseq`, `hseg`, and `dl`
use hatchling. The build-backend choice applies to the *compiled members*, not to the
family.

For `align`, `setup.py` is replaced by a `meson.build`; the extension target is
`pfsmgraph.align.algorithms.needleman_wunsch._cython`, with the source path
package-relative under `packages/pfsmgraph-align/`.

Operationally, meson-python editable installs require **`ninja` on `PATH`** in the
development environment for the rebuild-on-import hook, and the compiled members
additionally need a C compiler. `ninja` belongs in the root `dev` group rather than in
build isolation, because meson-python's isolated build environment is deleted after the
initial build while the rebuild hook keeps running.

## Consequences

### Positive

- **The dev loop improves for exactly the code that is hardest to debug.**
  meson-python's editable installs auto-rebuild compiled code on import. This advantage
  recurs for every kernel cythonized — Needleman-Wunsch, Smith-Waterman, Hirschberg,
  banded, the `hmm` Baum-Welch core — and again for every wavefront pass, so it
  compounds across the whole of
  [ADR 0002](0002-three-phase-algorithm-lifecycle.md) phase 2.
- **Pure-Python members stay trivial to build.** hatchling needs no compiler, no build
  system, and no `ninja`, so contributors touching only `dataseq` or `dl` need none of
  that toolchain.
- **Ecosystem alignment.** meson-python is where the scientific-Python compiled
  ecosystem has standardized: NumPy, SciPy, scikit-*. Problems encountered are problems
  others have already had.

### Negative / costs

- **Two build systems in one repository**, with two sets of conventions and failure
  modes to know.
- **The compiled members gain a `meson.build` that must be kept in sync** with the
  Python package layout by hand.
- **`ninja` and a C compiler become development prerequisites** for anyone working on
  `align` or `hmm`.
- **The namespace interaction was underestimated, and it bit.** See below.

## Alternatives considered

- **Stay on `setuptools.build_meta`.** Workable, and empirically shown not to have the
  namespace problem (see Evidence), but rejected on the dev-loop and ecosystem grounds
  above. Notably it was rejected *by preference, not by necessity*.
- **One backend for the whole family.** Rejected as D7: it would either impose a
  compiler toolchain on three packages that need none, or deny it to the two that do.
- **Defer the backend switch until later.** Rejected: now is the cheapest switch point,
  because there is almost nothing to port. Every algorithm added raises the cost.

## Evidence

- **setuptools editable does not propagate a `.pyx` edit.** Verified empirically: a
  stale `.so` persisted across a *fresh process* until a manual
  `build_ext --inplace` was run. Under meson-python the rebuild happens on import.
- **A namespace concern was considered and withdrawn — on evidence that turned out not
  to generalize.** setuptools was empirically shown to editable-install a Cython
  extension and a pure-Python sibling across two namespace workspace members with no
  shadowing (no `pfsmgraph/__init__.py`; `[tool.setuptools.packages.find]` with
  `where=["src"], namespaces=true`). That result is real, but it is a result *about
  setuptools*. meson-python's editable install works by a different mechanism — a
  `sys.meta_path` finder — and does **not** compose the same way. This is the substance
  of [ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md), and it is the reason
  the meson-python half of this decision is currently suspended.

## Open

`dataseq`'s backend follows from the merge
([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)): the
`dl`-derived base is presumed pure-Python (hatchling), but this is unconfirmed until the
merge lands. `hseg`'s backend depends on whether hierarchical segmentation has its own
DP recurrence or is pure orchestration over `align` — also unconfirmed.
