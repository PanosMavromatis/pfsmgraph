# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**There is no code yet.** The repository contains only `README.md` and `docs/design/PRD.md`. The PRD is the authoritative design document; read it before scaffolding anything. No `pyproject.toml`, no `uv.lock`, no `packages/`, no tests, and no ADRs exist — all of these are still to be created, and the PRD (§11, "Scaffolding notes") specifies how.

Consequently there are no build, lint, or test commands to document yet. Once the workspace exists, the toolchain is **uv** (workspace-based) with **pytest**; add the concrete commands to this file at that point.

## Architecture

`pfsmgraph` is a family of five independently publishable Python packages sharing one PEP 420 namespace, developed in a single repo as a **uv workspace** under `packages/`.

```
                  dataseq          (base — no intra-family dependencies)
                     │
                   align
                  ╱  │  ╲
              hseg  hmm  dl
```

| Distribution | Import | Role |
|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; base layer, PyTorch `Dataset`-compatible |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment (DP-heavy, compiled) |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Baum-Welch, topology search via state merge/split; translated from an existing Lush implementation |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models; `rnn` and `transformer` are plain submodules of one distribution |

The family coheres because `align`/`hseg` are **interpretability instruments for the outputs of `hmm`/`dl`**, and because alignment is a training accelerant for HMM topology search — not merely because the topics are adjacent.

**Implementation order** (`dataseq` → `hmm` → `align` → `hseg`) deliberately differs from **release order**, which must follow the dependency graph since a package cannot publish before its dependencies exist on PyPI.

## Invariants

These constrain any code written here. They are inherited from the proof-of-concept and are non-negotiable without amending the PRD.

- **No `pfsmgraph/__init__.py` anywhere.** The `pfsmgraph` level is a PEP 420 implicit namespace; each distribution contributes exactly one regular subpackage beneath it. An `__init__.py` at that level breaks every other package's imports.
- **Encode at the boundary.** Multi-character string symbols are mapped to integers at the entry point of every public call; all inner computation is integer-only; results decode back to strings at exit. This is what makes Cython and CUDA backends mechanical to write — they never touch string types.
- **Fixed reserved symbol block** in `dataseq`, not configurable: `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6. `PAD` must be 0 because PyTorch's zero-fill idioms (`pad_sequence`, `torch.zeros()` buffers) would otherwise silently mean something other than "absent". Encoding is **strict by default** — unseen symbols raise; `UNK` fallback is explicit opt-in.
- **Three-phase algorithm lifecycle**, applied in order wherever dynamic programming appears: pure Python (correctness) → Cython (performance) → Numba CUDA anti-diagonal wavefront (scale).
- **One parameterized test suite per algorithm**, run automatically against every available backend, so backend equivalence is enforced rather than assumed.
- **Build backends are per-package, not family-wide.** meson-python for compiled members (`align`, the Baum-Welch core of `hmm`); hatchling for pure-Python members. meson-python editable installs need `ninja` present for rebuild-on-import.
- **"GPU" means two unrelated things.** `numba-cuda` for the DP packages, `torch` for `dl`. Do not unify these into one `[gpu]` extra.

## Workspace footgun

During development a `{ workspace = true }` path source satisfies *any* version constraint, so a missing or wrong bound in `[project.dependencies]` never fails locally — it only breaks a pip user after publish. Keep published lower bounds honest and review them on every breaking change.

The `0.0.0` placeholder releases already on PyPI are intentionally dependency-free; do not add dependency declarations to them.

## Design docs

- `docs/design/PRD.md` — packaging, naming, and distribution architecture; the source for the initial ADR set (§9).
- `docs/design/adr/` — to be created. Numbering starts fresh at 0001; do not import earlier ADR numbering.
