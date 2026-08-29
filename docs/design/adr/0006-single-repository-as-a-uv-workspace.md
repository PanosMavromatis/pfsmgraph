# 0006. One repository, structured as a uv workspace

- **Status:** Accepted
- **Date:** 2026-06-29
- **Source:** PRD §5 — decision D5

## Context

Five independently publishable packages ([ADR 0005](0005-namespace-prefix-and-pep-420-layout.md))
have to live somewhere. The default assumption for independently published packages is
one repository each, and this project has a precedent for splitting: an earlier two-repo
split separated the public library from its Claude Code development plugin.

But that split was driven by **audience and access control** — a public library versus
internal tooling, with different permissions. That rationale does not transfer. The five
packages here serve a single audience (the project author, building one ecosystem) and
share a tight internal dependency graph in which `dataseq` and `align` sit beneath
everything else ([ADR 0009](0009-dataseq-as-the-base-layer.md)).

With N repositories, every early API change to `align` becomes a release-and-re-pin
dance across repositories: publish `align`, then bump the pin in `hseg`, `hmm`, and `dl`
separately. During the period when the API is least stable, that cost is paid most
often.

The blocking concern was whether a workspace would leak into what consumers receive.

## Decision

**A single repository, structured as a uv workspace, with each package an independently
publishable member under `packages/`.**

```
pyproject.toml        # virtual root: [tool.uv.workspace] members = ["packages/*"]
uv.lock               # one lockfile for the whole family
packages/pfsmgraph-<pkg>/pyproject.toml
packages/pfsmgraph-<pkg>/src/pfsmgraph/<pkg>/
```

The root is a **virtual** workspace root — it has no `[project]` table and is not itself
distributable. Sources use a `src/` layout.

Each consumer declares both the real published dependency and the development-time
redirect:

```toml
[project]
dependencies = ["pfsmgraph-dataseq>=0.1"]   # what pip users receive

[tool.uv.sources]
pfsmgraph-dataseq = { workspace = true }     # what the author resolves locally
```

## Consequences

### Positive

- **Pip users never know a workspace existed.** This is the decisive resolution (§5.3):
  a uv workspace is a **development-time construct only**, living entirely in
  `[tool.uv.workspace]` and `[tool.uv.sources]`, which build backends ignore. Published
  wheels are built from the standard `[project]` table, so consumer metadata carries an
  ordinary `pfsmgraph-align>=0.1`, and `pip install pfsmgraph-hseg` resolves from PyPI
  exactly as normal. The workspace changes how the author resolves and builds locally,
  not what consumers receive.
- **Atomic cross-package commits.** An API change in `align` and its propagation to
  `hseg`, `hmm`, and `dl` land in one commit, reviewable as one change. This is the
  single largest benefit while the family co-evolves.
- **One lockfile means all members resolve against one consistent set of transitive
  pins.**
- **Per-package build backends still work.** The workspace does not impose a family-wide
  build backend ([ADR 0008](0008-per-package-build-backends.md)).

### Negative / costs

- **The version-bound footgun, which is the serious one.** During development a
  `{ workspace = true }` path source satisfies *any* version constraint. A missing or
  wrong lower bound in `[project.dependencies]` therefore **never fails locally** — it
  only breaks a pip user after publish, in an environment the author is not in. Published
  lower bounds must be set deliberately when `dataseq` and `align` get their first real
  releases, and reviewed on every breaking change. This is recorded in `CLAUDE.md` as a
  standing hazard.
- **The single lockfile becomes a constraint if two members ever need conflicting
  transitive versions** — a real possibility given that `numba-cuda` and `torch` both
  care about CUDA versions ([ADR 0004](0004-gpu-backends-and-optional-dependency-strategy.md)).
- **The workspace does not solve compiled-extension editable-install friction.** That
  remains per-package build configuration, and is currently unsolved
  ([ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md)).
- **Release order still has to be managed by hand** — a package cannot publish before
  its dependencies exist on PyPI, so `dataseq` → `align` → {`hseg`, `hmm`, `dl`} is
  mandatory regardless of how atomic the commits are.

## Alternatives considered

- **One repository per package.** Rejected: it optimizes for independent lifecycles the
  family does not have, and imposes the release-and-re-pin cycle during the period of
  maximum API churn. It also has no answer for propagating a breaking change coherently.
- **A single monolithic distribution.** Rejected at
  [ADR 0005](0005-namespace-prefix-and-pep-420-layout.md).
- **A non-workspace monorepo with manual path installs.** Rejected: the workspace gives
  exactly this with one lockfile and no bespoke tooling.

## Open

Splitting a package into its own repository later remains available if and when its
lifecycle genuinely diverges. Nothing in this decision forecloses that; the published
artifacts would be unchanged.
