# Scratch — dataseq merge working area

**Temporary. Deleted by the last goal of `docs/plan/feat-dataseq-merge/TODO.md`.**

The three existing `dataseq` implementations are imported here to be read side by
side before anything is merged into `packages/pfsmgraph-dataseq/`. Nothing in this
directory is part of the distribution, and nothing outside it may import from it.

## Why the leading dot

It is load-bearing, not cosmetic. `uv run pytest` has no `testpaths` configured, so
its rootdir walk reaches everywhere; a half-translated Lush file that fails at import
would surface as a *collection error* and fail the run before any test executes. The
dot matches pytest's default `norecursedirs` entry `.*`, so this tree is never walked
— with no configuration to add now or remove later. `_scratch/` would not work: the
default list contains `_darcs`, not `_*`.

The directory also sits outside `packages/`, which the workspace glob
`members = ["packages/*"]` would otherwise claim as a member and fail on for want of
a `pyproject.toml`. And `.scratch` is not a name `.gitignore` already swallows
(`lib/`, `build/`, `var/`, `share/`), so this code commits normally.

## Layout

| Directory | Implementation | Language |
|---|---|---|
| `dl/` | The `dl` version — the merge base, per PRD §3.5 | Python |
| `hmm-lush/` | The earlier `hmm` implementation, plus its Python translation | Lush → Python |
| `py-rudimentary/` | The rudimentary third implementation | Python |

## Provenance

Filled in as each implementation is imported. Record where it came from and at what
revision, so a claim about "what the original did" stays checkable after this
directory is gone.

| Implementation | Source | Revision / date | Imported |
|---|---|---|---|
| `dl` | _tbd_ | _tbd_ | _tbd_ |
| `hmm-lush` | _tbd_ | _tbd_ | _tbd_ |
| `py-rudimentary` | _tbd_ | _tbd_ | _tbd_ |

## Before deleting

The cleanup goal requires deciding how this code is retained **first**. A squash
merge collapses the commit that adds it and the commit that deletes it into nothing,
losing it from `main` entirely. Retention needs a merge commit, a tag on the
pre-deletion SHA, or a branch left unmerged.
