# refactor/reserved-block-renumber

**Status**: merged — PR #5 — 2026-09-01
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
- [x] Renumber the `tokalign` encoder to the ADR 0011 block
  > **Q:** Should `Alphabet` gain the reserved *names* ADR 0011 defines, or only the new
  > offsets?
  > **A:** Module-level `Final` constants mirroring `dataseq/_reserved.py` -- not imported
  > from it, since `.scratch/` is a standalone tree. The `RESERVED_INDICES` field is removed
  > entirely.
  > **Why it matters beyond style:** the field is not a `ClassVar`, so it is a constructor
  > parameter today -- `Alphabet(symbols=..., RESERVED_INDICES=99)` is legal, which ADR 0011
  > forbids. `_reserved.py` cites this exact mistake in its own docstring. Moving the block to
  > module scope removes the parameter structurally rather than by comment.
  > **The audit's one real finding:** `ScoringMatrix.identity` zeroes
  > `range(alphabet.RESERVED_INDICES + 1)`. That `+ 1` encodes the old structure -- gap sitting
  > just past the reserved block. ADR 0011 moves `GAP` inside the block, so the expression
  > would zero rows 0-6 and silently blank the first *user* symbol's scores. It does not raise
  > and no gap-count assertion catches it.
  - [x] `_types.py`: user symbols 4 -> 6, gap 3 -> 4, and the reserved names ADR 0011
        requires that `tokalign` does not currently have
  - [x] Confirm `tokalign`'s own tests still pass, or record why they cannot run
  > **Done:** Seven hunks in `_types.py`. The block is now module-level `Final`
  > constants mirroring `_reserved.py`, and the `RESERVED_INDICES` field is gone, so it
  > can no longer be passed to the constructor. `ScoringMatrix.identity` zeroes
  > `range(USER_BASE)` rather than `range(RESERVED_INDICES + 1)`.
  > **Tests:** 62 passing, matching the pre-change baseline exactly (measured by
  > restoring `HEAD` and re-running). Run with
  > `PYTHONPATH=.scratch/align-poc/tokalign/src uv run pytest .scratch/align-poc/tokalign/tests/`.
  > Our own suite is unaffected: 74 passed.
  > **Correction to goal 1's finding:** the algorithm *sources* are index-agnostic, but
  > the algorithm *tests* were not. `_idx_to_sym` read `alphabet.symbols[idx - 4]` and
  > 46 call-site lists wrote user symbols as codes from 4 -- which collided with GAP at
  > 4, so `_idx_to_sym(4)` returned the gap symbol and 30 of 36 tests failed. Rebased
  > onto 0-based ordinals, so the tests no longer encode where user codes begin.
  > `test_needleman_wunsch.py` is now tracked so the fix is reviewable.
  > **Why goal 1's grep missed it:** a sweep for `gap`, `GAP` and `== <int>` cannot
  > match `symbols[idx - 4]` or `[4, 6, 5]`. Executing the suite is what found it.
- [x] Audit every hard-coded index assumption the renumbering exposes
  > **Q:** The traceback-direction enum now shares the range 0-5 with the reserved block.
  > Where should that be recorded?
  > **A:** A comment on the `Direction` enum in `_python.py`, which is tracked and is where
  > a reader would actually meet the numbers.
  > **Q:** `decode` is still partial in `tokalign`. Where should that land?
  > **A:** A new `DEFERRED.md` trigger section, "the `align` migration". `codex.md` already
  > tells a reviewer to report it, but nothing scheduled the fix.
  - [x] Sweep for bare integer literals standing in for reserved codes
  - [x] Check the scoring-matrix construction, which is sized from the alphabet and is
        the seam `pfsmgraph-align` will read across
  > **Done:** Swept all 14 `.py`/`.pyx` files under `src/`, `tests/`, `benchmarks/` and
  > `setup.py` -- tracked and untracked alike -- for integer literals in index positions,
  > rather than for names. `setup.py`, `_backends.py`, `_registry.py` and `conftest.py`
  > have none at all; `benchmarks/run_benchmark.py` samples `alphabet.symbols` by name and
  > its integers are reps, seeds and figure sizes. The scoring-matrix seam was the one real
  > site and was fixed in the previous commit (`range(USER_BASE)`).
  > **New finding, created by the renumbering rather than exposed by it:** the traceback
  > `Direction` enum spans 0-5, exactly the range the reserved block now occupies. The
  > namespaces are unrelated -- one indexes `T[i][j]`, the other the alphabet -- and the
  > overlap became total only when the block widened from four slots to six. Nothing
  > confuses them today; documented on the enum so nothing starts to.
  > **Out of scope, now scheduled:** `decode` stays partial. Filed to `DEFERRED.md` under a
  > new trigger, "the `align` migration", with the open part named -- what the reserved
  > codes should decode *to*.
- [x] Record the outcome
  - [x] Update the `DEFERRED.md` entry -- the `dl` half is already satisfied by PR #2
  - [x] Close the master-plan subgoal with a `> **Done:**` note
  > **Done:** The `DEFERRED.md` entry is closed by appending, not deleting: the original
  > text is what makes "renumbered onto the block" mean anything a year from now, since it
  > records what it was renumbered *from*. The closing paragraph also states what the
  > trigger still leaves live -- Lush's user-symbols-from-2 is a real third offset, but it
  > belongs to the `hmm` translation, not here.
  > **Note for `/smart-merge`:** its step 7 closes the master-plan subgoal itself. That is
  > already done, so it should find the item `[x]` with a `> **Done:**` note and add only
  > the PR number rather than a second note.
