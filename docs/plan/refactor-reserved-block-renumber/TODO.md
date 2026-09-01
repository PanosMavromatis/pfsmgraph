# refactor/reserved-block-renumber

**Status**: active
**Created**: 2026-09-01
**Subgoal**: Renumber the proof-of-concept alignment code to the reserved block
(user symbols from 6, new gap index), auditing every hard-coded index assumption.

## Goals

- [x] Settle the audit surface before touching any code
  > **Q:** How much of Phase 3 gets tracked, given that an audit over files git cannot
  > see is not reviewable?
  > **A:** Track `_python.py` only. It is the signature that makes the negative finding
  > checkable, and it leaves the `.pyx` inert so `DEFERRED.md`'s "first `.pyx` lands"
  > trigger stays unambiguous.
  > **Measured (2026-09-01):** the algorithms hard-code no gap index -- `_python.py:148`
  > and `_cython.pyx:79` both take `gap_index: int` as a parameter, fed from
  > `alphabet.gap_index` at the call site, and the tests compare against
  > `_STD_ALPHABET.gap_symbol` rather than a literal. The renumbering surface is
  > therefore `_types.py` alone, which is already tracked under Phase 1.
  > **Also found:** Phase 3's header overstates its contents. `scoring.py`,
  > `tests/test_scoring.py` and `algorithms/__init__.py` are 0 bytes; `_registry.py` is a
  > five-line name-to-module dict, not the "algorithm + backend dispatch" described; and
  > the `ScoringMatrix` it places in `scoring.py` is in tracked `_types.py`.
  - [x] Establish what Phase 3 of `.scratch/align-poc/.gitignore` would surface, and
        whether the renumbering can be verified without it
  - [x] Decide: advance the policy to Phase 3 now, or scope the renumbering to the
        tracked encoder and hand the algorithms to the `align` migration
  > **Done:** Phase 3 advanced by exactly one file. `_python.py` is now tracked as the
  > signature evidencing that the algorithms take `gap_index` as a parameter; `_cython.pyx`,
  > `setup.py` and `benchmarks/` stay inert. Verified: one new untracked path in
  > `git status`, and `git check-ignore -v` shows the `.pyx` still matched by a deny rule.
  > The policy header's inventory was corrected in the same edit -- three files it
  > described are 0 bytes and `_registry.py` is a five-line stub.
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
