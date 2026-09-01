# `.data/` — local inputs

Corpora and inputs for the workbench in [`.notebooks/`](../.notebooks/README.md):
`.sds` directories, CSVs, checkpoints — whatever a feature needs to be exercised
against real data rather than a fixture.

**Everything in this directory is gitignored except this file and
`.gitignore`.** Unlike the workbench, that is not only about tidiness: data here
is usually large, and often not ours to redistribute.

## Record provenance here

The one thing this directory asks of you is a line in this README for each
dataset you drop in — where it came from, at what revision or date, and whether
it may be redistributed. A directory of unlabelled corpora is worth very little
six months on, and the repository already has a worked example of the failure:
`.scratch/dl/`'s policy turns away MelodyHPO's `data/` precisely because it is a
checkout of a *separate* repository (MelodyData), which is a fact recorded in
`.scratch/README.md` rather than inferable from the bytes.

| Dataset | Source | Revision / date | Redistributable? |
|---|---|---|---|
| _(none yet)_ | | | |

## Existing specimens you do not need to copy here

Two real `.sds` datasets are already tracked in the repository, under
`.scratch/hmm-lush/Training/`, and they are the only specification of that
on-disk format that exists:

- `set11a_dInt.sds` — one sequence, 1449 symbols, a 25-symbol alphabet of
  genuine multi-character strings (`+4th`, `-8ve`, `|E. -2nd|`). The case
  `dataseq` exists to handle.
- `set01z0_100.sds` — 100 ragged sequences, lengths 2–20. The evidence for how
  the Lush original padded.

Read them in place. A `.sds` is a *directory*, and `_size` is a count the loader
trusts, so a partial copy is corrupt rather than small.

## Rules

- **Nothing under `packages/` may read from here.** Tests and modules in a
  distribution must be self-contained; this content exists on one machine only.
  A test that reads from `.data/` passes locally and fails in every clone.
- **Do not commit datasets by adding negations to `.gitignore`.** If something
  here genuinely belongs in the repository, it belongs somewhere a reader would
  look for it, with a policy that explains why — the way each `.scratch/` import
  does. Note also that the tracked set can only ever widen: `.gitignore` is
  consulted only for files git does not already track, so an ignore rule added
  over a tracked path is silently inert. See
  [ADR 0014](../docs/design/adr/0014-scratch-retention-and-per-package-scoping.md).
- **The leading dot is load-bearing** — it matches pytest's default
  `norecursedirs` entry `.*`, and the directory sits outside `packages/`, so
  neither the test run nor the workspace glob `packages/*` looks in here.
