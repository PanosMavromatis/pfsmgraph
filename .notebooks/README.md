# `.notebooks/` — local workbench

A scratch area for exercising the workspace packages by hand: write a throwaway
script or notebook, import `pfsmgraph.dataseq`, and see what a new feature
actually does before it is released.

**Everything in this directory is gitignored except this file and
`.gitignore`.** That is deliberate — it is a workbench, not a source tree. Write
freely; nothing here will follow you into a commit.

## You are not installing from PyPI

`uv sync` installs all five members **editable** (a plain `.pth`), so
`pfsmgraph.dataseq` resolves to the working tree, not to a wheel:

```
$ uv run python -c "import pfsmgraph.dataseq as d; print(d.__file__)"
/Users/…/pfsmgraph/packages/pfsmgraph-dataseq/src/pfsmgraph/dataseq/__init__.py
```

Because the install is editable, an edit under `packages/` is live at the next
interpreter start — there is nothing to reinstall, and nothing to publish. This
is what makes pre-release checking possible at all.

`uv run` walks up to the workspace root to find the environment, so it works
from inside this directory. Use `uv run python foo.py`, not a bare `python`,
unless you have activated `.venv` yourself.

## A worked example

Run verbatim from this directory. Output is pasted from the run, per
[ADR 0013](../docs/design/adr/0013-api-documentation-layout-and-tooling.md).

```python
from pfsmgraph.dataseq import SymbolTable, SequenceDataset, pad_collate

corpus = [["the", "cat", "sat"], ["the", "dog"]]
table = SymbolTable.from_sequences(corpus)
print("size:", table.size, "| encode:", table.encode(["the", "cat"]))

data = SequenceDataset.from_symbols(corpus, table)
print("lengths:", data.lengths)

batch = pad_collate([data[0], data[1]])
for key, value in batch.items():
    print(f"{key}:\n{value}")
```

```
size: 10 | encode: [6 7]
lengths: (3, 2)
codes:
[[6 7 8]
 [6 9 0]]
lengths:
[3 2]
mask:
[[ True  True  True]
 [ True  True False]]
```

Two things in that output are contracts rather than incidentals, and this is the
cheapest place to see them. User symbols start at **6** — the reserved block is
`PAD`=0 … `MSK`=5 and is not configurable
([ADR 0011](../docs/design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)).
And the `0` in the second row is padding introduced by `pad_collate`, which is
the only place padding is ever introduced and always returns its `mask` — the
records themselves are ragged and carry true lengths.

## Notebooks proper

No Jupyter kernel is installed: `ipykernel` is not a dependency of any member,
and the `dev` group holds only `pytest`. Plain `.py` scripts under `uv run` work
today and need nothing added.

If you want a real notebook, add the tooling to a **separate, non-default**
group so it never becomes a dependency of the family:

```bash
uv add --group notebooks ipykernel jupyterlab
uv run --group notebooks jupyter lab
```

`.ipynb_checkpoints` is already covered by the repo-root `.gitignore`.

## Rules

- **Nothing under `packages/` may import from here.** Code in this directory
  exists on one machine only, so a distribution that reaches into it passes
  locally and fails everywhere else. Same rule `.scratch/README.md` states for
  imported source.
- **Data goes in [`.data/`](../.data/README.md), not here.** Keep the workbench
  and its inputs separable, and record data provenance where that README asks.
- **The leading dot is load-bearing.** It matches pytest's default
  `norecursedirs` entry `.*`, so `uv run pytest` collects nothing written here —
  a scratch file named `test_*.py` would otherwise join the real suite. The
  directory also sits outside `packages/`, so the `packages/*` workspace glob
  never claims it as a member.
- **If you meant to commit something here, check that it is visible.** A file
  written without a matching negation in `.gitignore` is not an error — it is
  simply absent from `git status`, so the work looks committed and is not.
  `git check-ignore -v <path>` names the rule that matched.
