# pfsmgraph

A composable ecosystem of Python packages for modeling symbolic data sequences. Probabilistic finite-state models (PFSMs) are the unifying core, bridging sequence alignment, hierarchical segmentation, HMMs (Baum-Welch), deep learning (RNNs, Transformers), interpretability, and graph operations.

> **Status: one package released, one begun, three scaffolded.** The workspace layout, package boundaries, and build backends are in place. `pfsmgraph-dataseq` — the base layer every other member depends on — is implemented, tested, documented at [`docs/api/dataseq/`](docs/api/dataseq/README.md), and released at 0.1.0. `pfsmgraph-hmm` has begun: its migrated numeric helpers are in place and tested, and it now exports the frozen parameter value `HMMParams` together with the Viterbi decode over it — the project's first dynamic-programming kernel, checked against the original implementation's own saved decodes. The other three are still empty namespace subpackages, and all four unreleased names still hold dependency-free `0.0.0` placeholders. See [`docs/design/PRD.md`](docs/design/PRD.md) for the design and [`docs/design/adr/`](docs/design/adr/README.md) for the decision records, which are authoritative.

## Packages

`pfsmgraph` is a family of five independently publishable packages sharing one PEP 420 namespace (`import pfsmgraph.<pkg>`), developed together as a [uv](https://docs.astral.sh/uv/) workspace.

| Distribution | Import | Role | Depends on | Build backend |
|---|---|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; PyTorch `Dataset`-compatible base layer&nbsp;‡ | — | meson-python&nbsp;† |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment (DP-heavy, compiled) | `dataseq` | meson-python |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation | `dataseq`, `align` | meson-python&nbsp;† |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Baum-Welch, topology search via state merge/split | `dataseq`, `align` | meson-python |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models (`rnn`, `transformer` submodules) | `dataseq`, `align` | meson-python&nbsp;† |

† `align` and `hmm` are meson-python because they get Cython + CUDA kernels. The other three are marked because they are on it for a reason unrelated to building: meson-python's editable install injects a meta-path finder that replaces `pfsmgraph.__path__`, so **a member left on a plain `.pth` is shadowed by any sibling's finder and stops importing**. Finders chain, so the fix is for every member to have one — pure ones included. Measured 2026-09-04; the decision supersedes ADR 0012 and overrides ADR 0008's per-package backends. See [`packages/pfsmgraph-dataseq/pyproject.toml`](packages/pfsmgraph-dataseq/pyproject.toml), which carries the argument the other four point at.

‡ The only member with an implementation. Its public API is documented at [`docs/api/dataseq/`](docs/api/dataseq/README.md); the other four rows describe intent, not code.

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
docs/design/adr/               # decision records (authoritative)
docs/api/                      # API documentation, one subdirectory per package
packages/
├── pfsmgraph-dataseq/         # + meson.build
├── pfsmgraph-align/           # + meson.build
├── pfsmgraph-hseg/            # + meson.build
├── pfsmgraph-hmm/             # + meson.build
└── pfsmgraph-dl/              # + meson.build
```

Each member has its own `pyproject.toml`, and all five share the meson-python backend. Sources use a `src/` layout under `src/pfsmgraph/<pkg>/`; there is deliberately **no `pfsmgraph/__init__.py`** at any level — the namespace is implicit (PEP 420).

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync                             # venv + all five members editable + dev tools
uv run pytest                       # run the suite (280: 74 dataseq, 165 hmm, 41 root)
uv build --package pfsmgraph-align  # build one distribution
uv lock                             # refresh uv.lock (committed; one per family)
```

All five members build through meson-python, so `ninja` (for rebuild-on-import) is needed; `uv sync` provides it via the dev group. The dev group alone is **not** enough — the editable loader bakes an absolute `ninja` path at build time rather than consulting `PATH`, so every member is also listed under `[tool.uv] no-build-isolation-package` at the workspace root. A C compiler joins the list when `align`/`hmm` get their first `.pyx`; until then nothing here is compiled.

Because `uv sync` installs every member **editable**, imports resolve to `packages/*/src/` — so a feature can be exercised locally the moment it is written, with nothing to publish or reinstall. Two gitignored directories exist for that: [`.notebooks/`](.notebooks/README.md) is the workbench and [`.data/`](.data/README.md) holds its inputs. Each tracks only a `.gitignore` and a `README.md`; everything else written there is ignored. Nothing under `packages/` may import or read from either.

## Publishing

Release order follows the dependency graph: `dataseq` → `align` → {`hseg`, `hmm`, `dl`}; a package cannot publish before its dependencies exist on PyPI. `pfsmgraph-dataseq` is released at 0.1.0; the other five names — the four remaining packages plus the bare `pfsmgraph` umbrella — are still dependency-free `0.0.0` placeholders. See PRD §4 and §11.

Releases run through the repo-root `justfile`, which requires [just](https://just.systems) (`brew install just`): `just release <version> [package]` runs test → build → `twine check` → preflight → upload → tag, defaulting to `pfsmgraph-dataseq`. `just` alone lists every recipe, and [`docs/ops/release.md`](docs/ops/release.md) is the runbook.

## License

MIT — see [`LICENSE`](LICENSE).
