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

**"`dl`" is a slot in this repository's package family, not the name of the source
project.** The merge base comes from **MelodyHPO**, a standalone and now defunct
project that built dataset handling and DL models together without packaging them;
the `dl` label originates in the `pfsmgraph` umbrella structure and was applied
retroactively. Anywhere the plan or the branch doc says "the `dl` implementation",
the artefact is `.scratch/dl/MelodyHPO/melody_hpo/data/`.

## Provenance

Filled in as each implementation is imported. Record where it came from and at what
revision, so a claim about "what the original did" stays checkable after this
directory is gone.

| Implementation | Source | Revision / date | Imported |
|---|---|---|---|
| `dl` | `github.com/PanosMavromatis/MelodyHPO`, branch `main` | `5f423118fa7cda0d7ca347ef0f112a326cca819c` — 2026-03-23, "Automate doc_paths construction in minicorps layer definitions" | 2026-08-31 |
| `hmm-lush` | _tbd_ | _tbd_ | _tbd_ |
| `py-rudimentary` | _tbd_ | _tbd_ | _tbd_ |

The MelodyHPO working tree was copied whole, so it also carries a checkout of a
*second* repository at `MelodyHPO/data/MelodyData` —
`github.com/PanosMavromatis/MelodyData` at `abe762513eeeba0745425b76b3eb5f6409eadf65`
(2026-03-11), cloned with `--filter=blob:none`. That is ingested corpus data, not
MelodyHPO's own content, and is not tracked here; it is recorded because `data/` is
where `melody_hpo/data/` reads from, so a claim about the container's input format is
checkable against it while the copy is on disk.

The copy was `main` with one uncommitted file, `notebooks/explore/gpt2.ipynb`. That
is outside the imported scope, so nothing tracked here differs from `5f42311`.

## Two edits made to the imported tree

Both are renames, both reversible, and both were necessary rather than tidying.

- **`MelodyHPO/.git` → `MelodyHPO/.git-disabled`.** A directory containing `.git` is an
  *embedded repository*: git declines to descend into it and offers a gitlink instead.
  The failure mode is silent — `git add .scratch/dl/MelodyHPO/melody_hpo/data/data.py`
  staged nothing and exited `0`, and `git status` showed one collapsed
  `?? .scratch/dl/MelodyHPO/` line — so the branch would have merged with none of the
  merge base actually in it. The history is still readable with
  `git --git-dir=.git-disabled log`, which is how the revision above was captured.
- **`MelodyHPO/CLAUDE.md` → `MelodyHPO/CLAUDE.md.orig`.** Claude Code loads a nested
  `CLAUDE.md` when it reads files beside it, so a defunct project's agent instructions
  would have entered sessions working in this repository. Being untracked does not
  prevent that; presence on disk is what does it. The suffix keeps it readable while
  making it inert.

## What is tracked, and why so little

`.scratch/dl/.gitignore` is deny-by-default: the MelodyHPO tree is 2.2 GB on disk and
33 files (≈62 KB) of it are tracked. The exclusions carry their reasons inline in that
file rather than here. Two are worth knowing without opening it: `.env` stays ignored
by the repo-root rules, and `data/` (5.4 MB of ingested CSVs) is deliberately not
vendored, since it belongs to MelodyData rather than to MelodyHPO.

Both `.venv` trees, the caches, and `LLMsFS/` account for essentially all of the 2.2 GB.
None of it is needed to read the merge base; **do not walk the tree indiscriminately.**

## Before deleting

The cleanup goal requires deciding how this code is retained **first**. A squash
merge collapses the commit that adds it and the commit that deletes it into nothing,
losing it from `main` entirely. Retention needs a merge commit, a tag on the
pre-deletion SHA, or a branch left unmerged.
