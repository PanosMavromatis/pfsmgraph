# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**The workspace is scaffolded and `uv sync`s cleanly; no algorithms are implemented.** In place: the uv workspace root `pyproject.toml` (virtual — no `[project]` table), `uv.lock`, all five `packages/*` members with their own `pyproject.toml`, the (currently dormant) `meson.build` files for `align` and `hmm`, and one empty `pfsmgraph/<pkg>/__init__.py` per member (plus `dl/rnn/` and `dl/transformer/`). Not yet present: any implementation code, any tests, any ADRs (`docs/design/adr/` exists but is empty). The proof-of-concept alignment/HMM code has not been moved in. The PRD remains the authoritative design document.

**`align` and `hmm` are temporarily on hatchling, not meson-python.** meson-python's editable-install import hook injects a `sys.meta_path` finder that claims the entire `pfsmgraph` PEP 420 namespace and shadows the other distributions, so `import pfsmgraph.dataseq` (and `hseg`, `dl`) fails after `uv sync`. Neither package has compiled code yet — the `meson.build` extension blocks are dormant `if fs.exists()` guards — so the switch to meson-python is deferred to when the first `.pyx` lands, at which point the namespace/editable interaction must be solved (non-editable install of the compiled members, a single combined compiled distribution, or an upstream fix). The `meson.build` files and the revert recipe are kept in `packages/pfsmgraph-{align,hmm}/pyproject.toml`. This qualifies the PRD §6.1 note, whose "namespace is fine" evidence was gathered with *setuptools* editable, which composes; meson-python's finder does not.

Still to do, in PRD order (§11): author the initial ADR set (§9); then implement `dataseq` (merge of three existing implementations, `dl` version as base — §3.5); then `hmm` (Lush translation); then `align`, then `hseg`.

## Commands

Toolchain: **uv** (workspace) + **pytest**. Requires `uv` and Python ≥ 3.10.

- `uv sync` — create/refresh the venv; installs all five members editable (plain `.pth`) plus the `dev` group (`pytest`).
- `uv run pytest` — run the suite (no tests exist yet).
- `uv build --package pfsmgraph-<pkg>` — build one member's sdist + wheel.
- `uv lock` — refresh `uv.lock` (committed; one lockfile for the whole family).

When `align`/`hmm` move back to meson-python: re-add `meson-python`, `cython`, and `ninja` to the root `dev` group (`ninja` must be on `PATH` for rebuild-on-import), and the compiled members will additionally need a C compiler.

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
- **Build backends are per-package, not family-wide.** meson-python for compiled members (`align`, the Baum-Welch core of `hmm`); hatchling for pure-Python members. meson-python editable installs need `ninja` present for rebuild-on-import. *(Currently `align`/`hmm` are on hatchling too, pending their first `.pyx` — see "Current state".)*
- **"GPU" means two unrelated things.** `numba-cuda` for the DP packages, `torch` for `dl`. Do not unify these into one `[gpu]` extra.

## Workspace footgun

During development a `{ workspace = true }` path source satisfies *any* version constraint, so a missing or wrong bound in `[project.dependencies]` never fails locally — it only breaks a pip user after publish. Keep published lower bounds honest and review them on every breaking change.

The `0.0.0` placeholder releases already on PyPI are intentionally dependency-free; do not add dependency declarations to them.

## Design docs

- `docs/design/PRD.md` — packaging, naming, and distribution architecture; the source for the initial ADR set (§9).
- `docs/design/adr/` — exists but empty; ADRs still to be written. Numbering starts fresh at 0001; do not import earlier ADR numbering.
