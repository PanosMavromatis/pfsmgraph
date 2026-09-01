# refactor/reserved-block-renumber

**Created**: 2026-09-01
**Base**: main at 4c2aff6
**Status**: active

## Purpose

Bring the proof-of-concept alignment code in `.scratch/align-poc/tokalign/` onto the
ADR 0011 reserved block -- `PAD`=0 ... `MSK`=5, user symbols from 6 -- and audit every
hard-coded index assumption the change exposes. `tokalign` allocates user symbols from
4 with its gap code at 3, which is the one imported source that *has* a gap code at all,
and it is the ancestor of `pfsmgraph-align`. Nothing is persisted anywhere, so this is a
code-only edit today; once anything encoded is written to disk it becomes a data
migration, which is why `DEFERRED.md` requires it to land inside revision
`01-dataseq-v0.1.0` rather than during the `align` migration that will consume the code.

## Scope

- Decide the audit surface: `.scratch/align-poc/.gitignore` is phased, and the alignment
  algorithms sit behind a commented Phase 3. An audit is only trustworthy over files git
  can see.
- Renumber the `Alphabet` encoder in `tokalign/src/tokalign/_types.py`.
- Audit every hard-coded index in the alignment code -- the gap index especially, since
  it moves from 3 to 4 and is the one `align` exists to emit.
- Record the outcome in `DEFERRED.md` (the entry is wider than its title) and close the
  master-plan subgoal.

## Context

- [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) --
  fixes the block; the authority this branch renumbers *to*.
- [ADR 0014](../../design/adr/0014-scratch-retention-and-per-package-scoping.md) --
  why `.scratch/` is retained and why its policies are re-scoped per package. Relevant
  because advancing a phase is a one-way widening.
- [`DEFERRED.md`](../../plan/DEFERRED.md), trigger "the `dataseq` merge" -- the entry,
  including its 2026-08-31 widening. Note that the `dl` half of that widening was already
  satisfied by PR #2, which hard-coded the block in `dataseq/_reserved.py`.
- `.scratch/align-poc/COMPARISON.md` -- the read of `tokalign` this branch acts on.
- PR #2 (the `dataseq` merge) landed the reserved block this renumbering targets.

## Notes

- 2026-09-01: branch created from `main` at `4c2aff6`. `docs/git/` did not exist and was
  recreated here -- PR #2's branch doc was deleted at merge, as designed.
- 2026-09-01: audit surface settled. The alignment algorithms hard-code no gap index --
  `_python.py:148` and `_cython.pyx:79` both take `gap_index: int` as a parameter fed from
  `alphabet.gap_index` -- so the renumbering is confined to `_types.py`. Phase 3 of
  `.scratch/align-poc/.gitignore` was advanced by exactly one file, `_python.py`, as the
  signature that makes that negative finding checkable; `_cython.pyx` stays untracked to keep
  `DEFERRED.md`'s first-`.pyx` trigger unambiguous.
- 2026-09-01: that phase's header inventory was wrong and is corrected. `scoring.py`,
  `tests/test_scoring.py` and `algorithms/__init__.py` are 0 bytes, and `_registry.py` is a
  five-line name-to-module dict -- not the backend dispatch it claimed, which is in
  `_backends.py`. `ScoringMatrix` is in `_types.py`, already tracked under Phase 1.
- 2026-09-01: `_types.py:136` ("zero out reserved indices and gap row/column") is the real
  target of the index audit, and it sits inside the file the renumbering already had to touch.
