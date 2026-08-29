# 0012. `align` and `hmm` are temporarily on hatchling, not meson-python

- **Status:** Accepted (temporary) — qualifies
  [ADR 0008](0008-per-package-build-backends.md) rather than reversing it. Expires when
  the first Cython kernel lands.
- **Date:** 2026-08
- **Source:** repository state; `CLAUDE.md` "Current state"; the revert recipe kept in
  `packages/pfsmgraph-align/pyproject.toml` and `packages/pfsmgraph-hmm/pyproject.toml`

## Context

[ADR 0008](0008-per-package-build-backends.md) puts `align` and `hmm` on meson-python.
When that was applied to the scaffolded workspace, the workspace stopped importing.

**meson-python's editable-install import hook injects a `sys.meta_path` finder that
claims the entire `pfsmgraph` PEP 420 namespace** and shadows the other distributions.
After `uv sync`, `import pfsmgraph.dataseq` — and likewise `hseg` and `dl` — fails,
because the finder installed on behalf of `pfsmgraph-align` answers for the whole
`pfsmgraph` prefix and does not know about its siblings.

A second, independent instance of the same problem: **two meson-python editable
installs also conflict with each other**, so this is not resolved by having only one
compiled member.

This is a direct collision between two decisions that are individually sound:
[ADR 0005](0005-namespace-prefix-and-pep-420-layout.md)'s single shared namespace, and
[ADR 0008](0008-per-package-build-backends.md)'s meson-python editable installs. It
affects development only — a non-editable install of built wheels composes correctly —
but development is where the workspace lives
([ADR 0006](0006-single-repository-as-a-uv-workspace.md)).

The mitigating fact: **neither package has compiled code yet.** The `meson.build`
extension blocks are dormant, guarded by `if fs.exists()`. So meson-python is currently
paying its entire cost and delivering none of its benefit — the rebuild-on-import loop
has nothing to rebuild.

## Decision

**`align` and `hmm` use hatchling for now.** Their `meson.build` files are retained
unchanged, and the revert recipe is kept as a comment in each package's
`pyproject.toml`. The root `dev` group correspondingly omits `meson-python`, `cython`,
and `ninja`.

The switch to meson-python is **deferred to the moment the first `.pyx` lands**, at
which point the namespace/editable interaction must actually be solved. Candidate
resolutions, none yet chosen:

1. **Non-editable install of the compiled members** — accepts a manual reinstall step in
   the dev loop for `align`/`hmm`, which is close to the setuptools status quo ante.
2. **A single combined compiled distribution** — one distribution owning all compiled
   code, so only one meson-python finder ever exists. Costs a package boundary.
3. **An upstream fix** — a meson-python editable finder that defers to other finders for
   prefixes it does not own. Correct, but not on this project's schedule.

**Revert recipe** (kept verbatim in each package's `pyproject.toml`) — in both
`packages/pfsmgraph-align/pyproject.toml` and `packages/pfsmgraph-hmm/pyproject.toml`:

```toml
[build-system]
requires = ["meson-python", "cython>=3.0", "numpy>=2.1"]
build-backend = "mesonpy"
```

then drop the `[tool.hatch.*]` block from each, and restore `meson-python`, `cython`,
and `ninja` to the workspace-root `dev` group. `ninja` must be on `PATH` for the
rebuild-on-import hook, and the compiled members will additionally need a C compiler.
The Needleman-Wunsch extension target is
`pfsmgraph.align.algorithms.needleman_wunsch._cython`
([ADR 0008](0008-per-package-build-backends.md)).

## Consequences

### Positive

- **`uv sync` produces a working, fully importable workspace.** All five members install
  editable via plain `.pth` files and all five import.
- **Nothing is lost.** With no compiled code present, meson-python would produce a
  pure-Python wheel anyway, so hatchling is equivalent for these two packages today.
- **The decision, the evidence, and the exit condition are recorded together** — this
  ADR exists because the deviation was previously visible only in a `pyproject.toml`
  comment, where it would have read as an unexplained inconsistency with
  [ADR 0008](0008-per-package-build-backends.md).

### Negative / costs

- **The repository does not match its own stated architecture**, and every document
  describing the build backends needs a footnote. `README.md` and `CLAUDE.md` both carry
  one.
- **The real problem is deferred, not solved**, and it lands at exactly the moment the
  first kernel is being written — when the meson-python dev loop is most wanted.
- **The `meson.build` files are unexercised** for as long as this holds, so they may
  have drifted or been wrong all along; that will only be discovered on revert.

## Alternatives considered

- **Keep meson-python and accept the broken imports.** Rejected: `import
  pfsmgraph.dataseq` failing after `uv sync` makes the workspace unusable, which defeats
  [ADR 0006](0006-single-repository-as-a-uv-workspace.md) entirely.
- **Solve the namespace problem now.** Rejected as premature: all three candidate
  resolutions carry real costs, and choosing between them without a compiled kernel to
  evaluate against would be guessing. The information needed to choose arrives with the
  first `.pyx`.
- **Revert to setuptools, which was shown to compose across namespace members.**
  Genuinely tempting — see the Evidence in
  [ADR 0008](0008-per-package-build-backends.md) — but rejected for now: it would
  reintroduce the stale-`.so` dev loop that motivated leaving setuptools, and hatchling
  is simpler for packages that currently compile nothing.
- **Drop the shared namespace.** Rejected outright — it is the foundation of
  [ADR 0005](0005-namespace-prefix-and-pep-420-layout.md).

## Evidence

**This qualifies PRD §6.1's withdrawn namespace concern, whose evidence does not
generalize.** §6.1 records that setuptools was empirically shown to editable-install a
Cython extension and a pure-Python sibling across two namespace workspace members with
no shadowing. That finding is correct — but it is a finding *about setuptools*, whose
editable install composes across namespace members. meson-python's works by a different
mechanism, a `sys.meta_path` finder, and does not compose. The two experiments look
alike and give opposite results, which is precisely why this needs recording rather than
remembering.
