# refactor/reserved-block-renumber

**Status**: active
**Created**: 2026-09-01
**Subgoal**: Renumber the proof-of-concept alignment code to the reserved block
(user symbols from 6, new gap index), auditing every hard-coded index assumption.

## Goals

- [ ] Settle the audit surface before touching any code
  - [ ] Establish what Phase 3 of `.scratch/align-poc/.gitignore` would surface, and
        whether the renumbering can be verified without it
  - [ ] Decide: advance the policy to Phase 3 now, or scope the renumbering to the
        tracked encoder and hand the algorithms to the `align` migration
- [ ] Renumber the `tokalign` encoder to the ADR 0011 block
  - [ ] `_types.py`: user symbols 4 -> 6, gap 3 -> 4, and the reserved names ADR 0011
        requires that `tokalign` does not currently have
  - [ ] Confirm `tokalign`'s own tests still pass, or record why they cannot run
- [ ] Audit every hard-coded index assumption the renumbering exposes
  - [ ] Sweep for bare integer literals standing in for reserved codes
  - [ ] Check the scoring-matrix construction, which is sized from the alphabet and is
        the seam `pfsmgraph-align` will read across
- [ ] Record the outcome
  - [ ] Update the `DEFERRED.md` entry -- the `dl` half is already satisfied by PR #2
  - [ ] Close the master-plan subgoal with a `> **Done:**` note
