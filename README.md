# pfsmgraph

A composable ecosystem of Python packages for modeling symbolic data sequences. Probabilistic finite-state models (PFSMs) are the unifying core, bridging sequence alignment, hierarchical segmentation, HMMs (Baum-Welch), deep learning (RNNs, Transformers), interpretability, and graph operations.

> **Status: scaffolding.** The workspace layout, package boundaries, and build backends are in place. No algorithms are implemented yet — every package is an empty namespace subpackage. See [`docs/design/PRD.md`](docs/design/PRD.md) for the design; decision records will live in [`docs/design/adr/`](docs/design/adr/) (none written yet).

## Packages

`pfsmgraph` is a family of five independently publishable packages sharing one PEP 420 namespace (`import pfsmgraph.<pkg>`), developed together as a [uv](https://docs.astral.sh/uv/) workspace.

| Distribution | Import | Role | Depends on | Build backend |
|---|---|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; PyTorch `Dataset`-compatible base layer | — | hatchling |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment (DP-heavy, compiled) | `dataseq` | meson-python&nbsp;† |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation | `dataseq`, `align` | hatchling |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Baum-Welch, topology search via state merge/split | `dataseq`, `align` | meson-python&nbsp;† |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models (`rnn`, `transformer` submodules) | `dataseq`, `align` | hatchling |

† `align` and `hmm` are meson-python by design (they get Cython + CUDA kernels), but are **temporarily on hatchling** until the first kernel lands — meson-python's editable-install hook currently shadows the shared namespace. See [`packages/pfsmgraph-align/pyproject.toml`](packages/pfsmgraph-align/pyproject.toml).

```
                  dataseq          (base — no intra-family dependencies)
                     │
                   align
                  ╱  │  ╲
              hseg  hmm  dl
```

`align` and `hseg` are interpretability instruments for the outputs of `hmm` and `dl`; alignment also accelerates HMM topology search. That is what makes the family a family rather than a bundle of adjacent topics.

## Repository layout

```
pyproject.toml                 # uv workspace root (virtual — not a package)
docs/design/PRD.md             # authoritative design document
docs/design/adr/               # decision records (to be written)
packages/
├── pfsmgraph-dataseq/
├── pfsmgraph-align/           # + meson.build
├── pfsmgraph-hseg/
├── pfsmgraph-hmm/             # + meson.build
└── pfsmgraph-dl/
```

Each member has its own `pyproject.toml` and build backend. Sources use a `src/` layout under `src/pfsmgraph/<pkg>/`; there is deliberately **no `pfsmgraph/__init__.py`** at any level — the namespace is implicit (PEP 420).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync                             # venv + all five members editable + dev tools
uv run pytest                       # run the test suite (none yet)
uv build --package pfsmgraph-align  # build one distribution
uv lock                             # refresh uv.lock (committed; one per family)
```

Once `align`/`hmm` return to meson-python, a C compiler and `ninja` (for rebuild-on-import) are also needed; `uv sync` will provide `ninja` via the dev group.

## Publishing

Release order follows the dependency graph: `dataseq` → `align` → {`hseg`, `hmm`, `dl`}; a package cannot publish before its dependencies exist on PyPI. All six names — the five packages plus the bare `pfsmgraph` umbrella — are claimed with dependency-free `0.0.0` placeholders. See PRD §4 and §11.

## License

MIT — see [`LICENSE`](LICENSE).
