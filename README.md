# pfsmgraph

A composable ecosystem of Python packages for modeling symbolic data sequences. Probabilistic finite-state models (PFSMs) are the unifying core, bridging sequence alignment, hierarchical segmentation, HMMs (Baum-Welch), deep learning (RNNs, Transformers), interpretability, and graph operations.

> **Status: two packages released, three scaffolded.** The workspace layout, package boundaries, and build backends are in place. `pfsmgraph-dataseq` — the base layer every other member depends on — is implemented, tested, documented at [`docs/api/dataseq/`](docs/api/dataseq/README.md), and released at 0.1.0. `pfsmgraph-hmm` is implemented through Baum-Welch and released at **0.2.0** (2026-09-16). It exports the frozen parameter value `HMMParams`, the Viterbi decode over it — one record at a time or many padded together — and Baum-Welch training over a fixed topology, each with a choice of backend across [ADR 0002](docs/design/adr/0002-three-phase-algorithm-lifecycle.md)'s four lifecycle phases plus torch. The decode was checked against the original implementation's own saved decodes; the compiled phases are held to the numpy reference **bit for bit**, not to a tolerance. 0.2.0 is this family's first release of platform wheels — twenty of them, cp310–cp314 across four platforms — because it is the first version whose public API reaches a compiled kernel, and a pure wheel would report `cython ✗` to every pip user. Topology search by state merge and split is a later revision. The other three are still empty namespace subpackages, and all three unreleased names still hold dependency-free `0.0.0` placeholders. See [`docs/design/PRD.md`](docs/design/PRD.md) for the design and [`docs/design/adr/`](docs/design/adr/README.md) for the decision records, which are authoritative.

## Packages

`pfsmgraph` is a family of five independently publishable packages sharing one PEP 420 namespace (`import pfsmgraph.<pkg>`), developed together as a [uv](https://docs.astral.sh/uv/) workspace.

| Distribution | Import | Role | Depends on | Build backend |
|---|---|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; PyTorch `Dataset`-compatible base layer&nbsp;‡ | — | meson-python&nbsp;† |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment (DP-heavy, compiled) | `dataseq` | meson-python |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation | `dataseq`, `align` | meson-python&nbsp;† |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Arc-emission HMMs: parameters and Viterbi decode (0.1.0); Baum-Welch and merge/split topology search to follow&nbsp;‡ | `dataseq` | meson-python |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models (`rnn`, `transformer` submodules) | `dataseq`, `align` | meson-python&nbsp;† |

† `align` and `hmm` are meson-python because they get Cython + CUDA kernels. The other three are marked because they are on it for a reason unrelated to building: meson-python's editable install injects a meta-path finder that replaces `pfsmgraph.__path__`, so **a member left on a plain `.pth` is shadowed by any sibling's finder and stops importing**. Finders chain, so the fix is for every member to have one — pure ones included. Measured 2026-09-04 and recorded as [ADR 0018](docs/design/adr/0018-family-wide-meson-python-build-backend.md), which supersedes ADR 0012 and overrides ADR 0008's per-package backends. The ADR carries the measurements and the rejected alternatives; [`packages/pfsmgraph-dataseq/pyproject.toml`](packages/pfsmgraph-dataseq/pyproject.toml) carries the short form the other four members point at.

‡ Implemented and released. Public APIs are documented at [`docs/api/dataseq/`](docs/api/dataseq/README.md) and [`docs/api/hmm/`](docs/api/hmm/README.md); the other three rows describe intent, not code. The Depends-on column shows declared dependencies ([ADR 0019](docs/design/adr/0019-declared-dependencies-follow-imports.md)), so `hmm` gains `align` only when it imports it.

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
dp-compile.toml                # manifest for the dp-compile plugin (see docs/agents/claude.md)
.claude/settings.json          # marketplace + enabled plugins (see docs/agents/claude.md)
.github/workflows/             # CI: suite on push/PR, cibuildwheel release (first CI, 2026-09-16)
docs/design/PRD.md             # authoritative design document
docs/design/adr/               # decision records (authoritative)
docs/design/algorithms/        # per-algorithm formalizations, one directory each
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
uv run pytest                       # run the suite (1418: 74 dataseq, 1292 hmm, 52 root)
uv build --package pfsmgraph-align  # build one distribution
uv lock                             # refresh uv.lock (committed; one per family)
```

All five members build through meson-python, so `ninja` (for rebuild-on-import) is needed; `uv sync` provides it via the dev group. The dev group alone is **not** enough — the editable loader bakes an absolute `ninja` path at build time rather than consulting `PATH`, so every member is also listed under `[tool.uv] no-build-isolation-package` at the workspace root. A C compiler is needed too, as of 2026-09-09: `hmm` carries the first `.pyx` (`_viterbi_cython.pyx`). `align` has no compiled code yet.

Because `uv sync` installs every member **editable**, imports resolve to `packages/*/src/` — so a feature can be exercised locally the moment it is written, with nothing to publish or reinstall. Two gitignored directories exist for that: [`.notebooks/`](.notebooks/README.md) is the workbench and [`.data/`](.data/README.md) holds its inputs. Each tracks only a `.gitignore` and a `README.md`; everything else written there is ignored. Nothing under `packages/` may import or read from either.

## Publishing

Release order follows declared dependencies ([ADR 0019](docs/design/adr/0019-declared-dependencies-follow-imports.md)): a package cannot publish before the family members it imports exist on PyPI, so `dataseq` goes first and `hmm`, which imports only `dataseq`, can follow it directly. `pfsmgraph-dataseq` is released at 0.1.0 and `pfsmgraph-hmm` at 0.2.0; the other four names — the three remaining packages plus the bare `pfsmgraph` umbrella — are still dependency-free `0.0.0` placeholders. See PRD §4 and §11.

Releases run through the repo-root `justfile`, which requires [just](https://just.systems) (`brew install just`): `just release <version> [package]` runs test → build → `twine check` → preflight → upload → tag, defaulting to `default_package`, the member under development, so an omitted argument can never reach a published package. Since 2026-09-16 that is one of **two** release paths: a member whose shipped artifact this machine cannot build — today `pfsmgraph-hmm`, whose wheels GitHub Actions builds — releases through `just release-ci <version> [package]`, which pushes the tag that triggers the workflow's publish job. `just` refuses the wrong path in either direction. `just` alone lists every recipe, and [`docs/ops/release.md`](docs/ops/release.md) is the runbook.

## License

MIT — see [`LICENSE`](LICENSE).
