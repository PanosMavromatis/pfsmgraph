# chore/release-dataseq-0.1.0

**Status**: active
**Created**: 2026-09-01
**Subgoal**: Release `pfsmgraph-dataseq` 0.1.0, replacing the `0.0.0` placeholder, and set honest lower bounds on the intra-family dependencies that name it (revision `01-dataseq-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Settle ADR 0003's sdist/wheel question
  > **Q:** Tests ship in the sdist without the policy that makes them honest (measurement
  > reproduced: sdist carries `tests/` but no `-ra` and no `pytest_report_header`; wheel
  > carries no tests at all). Ship the policy per member, stop shipping tests, or record
  > the policy as repo-local?
  > **A:** Repo-local, with an `align` trigger. Keep shipping tests; state in ADR 0003
  > that the mechanism does not travel in any sdist; file the obligation to revisit at
  > the first member that registers a real backend.
  > **Found:** the sdist also ships `.gitignore` and carries **no README and no LICENSE
  > file**, though `license = "MIT"` is declared. The PyPI page would be blank on first
  > publish. Added to goal 4 rather than folded in here.
  - [x] Re-read the measurement and the three candidate remedies as the record states them
  - [x] Put the choice to the user and log the answer inline
  - [x] Amend ADR 0003, moving the question out of `Open` per ADR 0010's convention
  - [x] Implement whichever remedy was chosen, or record explicitly that none was
  > **Done:** the chosen remedy is documentary, so there is no packaging change to make
  > and that is stated rather than left as an apparent omission. ADR 0003's sdist bullet
  > moved `Open` -> `Resolved`; `DEFERRED.md` closes the release-trigger entry and opens
  > a replacement under the `align` migration.

- [x] Sweep the prose claims about repository state, semantically
  - [x] `README.md` — the front page, and the surface with no other source of truth
  > **README:** clean as of 2026-09-01. Dependency table verified against all five
  > `pyproject.toml` files, test count (87) and root `LICENSE` confirmed. Two claims are
  > true now and will be falsified by this branch's own publish -- the `0.0.0`-placeholder
  > sentences at lines 5 and 67 -- so they are filed under the release goal, to change in
  > the commit that makes them false rather than before it.
  - [x] `docs/agents/core.md` and `codex.md`, then regenerate the `AGENTS.*` artifacts
  > **Agent docs:** nine edits. `codex.md` carried the worst of it -- "the repo is
  > scaffolding: no algorithms, no tests" (false since 2026-08-31), a 63-test count that
  > should read 74, and a bullet calling `SymbolTable` provisional, which the same file
  > tells a reviewer to *report* as staleness 100 lines earlier. Also two stale
  > `feat/dataseq-merge` branch framings. `core.md` had a broken ADR-index link
  > (`](docs/design/adr/README.md)` resolves under `docs/agents/`), a pre-merge
  > future-tense framing of the completed merge, an undated 74-test measurement, and a
  > total-vs-package test count. All numeric claims verified and correct: the four
  > `.scratch/` tracked-file counts (34/143/73/11), the ADR count, `/agents-docs-check`.
  > `AGENTS.md` and `AGENTS.override.md` regenerated; `check-agents-md.sh` reports in sync.
  - [x] `docs/design/PRD.md` and each ADR's `Status`, plus `docs/design/adr/README.md`
  > **Q:** The PRD is a dated snapshot ("threads still open at the time of writing"), yet
  > parts of it now read as false. Annotate it, rewrite its tenses, or supersede §8?
  > **A:** Annotate, following the pattern §9 already set -- narrative tense untouched,
  > dated `> **Status:**` blockquotes over discharged items, and §8 rewritten because a
  > section titled "Open questions" asserts the present every time it is read.
  > **PRD/ADRs:** all fourteen ADR `Status` lines match their index rows exactly; no ADR
  > body carries a stale state claim (four future-tense hits, all genuine design
  > consequences). Fixed: §8's two closed entries (the encoder API, settled by ADR 0010;
  > the build backend, settled as hatchling), dated blockquotes over §1.5 and §11, and the
  > "twelve records" count clarified as the initial set against an index of fourteen. The
  > ADR index pointed the reader at `CLAUDE.md` for the inherited hard rules -- true when
  > written, but `CLAUDE.md` is now an `@import` dispatcher, so it points at
  > `docs/agents/core.md` instead.
  - [x] `docs/api/dataseq/` — every executed code block still matching its pasted output
  > **Q:** The API docs are drift-free, but ADR 0013's "executed and pasted" rule has no
  > mechanism behind it. Keep the verifier written to check them?
  > **A:** Land it as a repo-root test.
  > **API docs:** 44 of 44 examples match, pasted exception messages included -- zero
  > drift. The finding is the *absence of a guard*, not a defect. Stock doctest fails 39
  > of the 44 for harness reasons alone: setup lives in plain blocks with no `>>>` so the
  > namespace is empty, and errors are pasted as the readable last line without doctest's
  > `Traceback` header. Both make the pages better and doctest inapplicable, so
  > `tests/test_api_docs.py` reads the documents' own convention instead -- stdlib only,
  > no build step, nothing added to `dev`, so ADR 0013's decision is unchanged. Verified
  > by breaking the docs three ways (wrong value, wrong exception message, truncated setup
  > block) and confirming each fails. ADR 0013's `## Open` is now `## Resolved` and the
  > `DEFERRED.md` entry under "CI existing" is closed, ahead of CI rather than with it.
  > Suite 87 -> 91; the counts in `README.md` and `core.md` that this falsified were
  > updated in the same change.

- [x] Set honest version lower bounds
  > **Q:** All four dependents declare `pfsmgraph-dataseq>=0.1`, which today is satisfiable
  > by nothing that exists. What bound should they carry once `0.1.0` is published?
  > **A:** `>=0.1.0`, no upper cap -- the floor spelled in full, naming the first version
  > that has an API. No cap, because a cap deadlocks downstream resolution and the four
  > dependents currently use nothing from `dataseq`, so it would assert a compatibility
  > limit nobody has tested.
  - [x] List every intra-family dependency naming `pfsmgraph-dataseq` and what it currently declares
  > **Four, all identical:** `align`, `hmm`, `hseg` and `dl` each declare
  > `pfsmgraph-dataseq>=0.1`, and each carries a `{ workspace = true }` source for it.
  > `align`'s is the only one with a comment, and it already names the footgun.
  - [x] Decide the bound `0.1.0` actually earns, and write it
  > **Written:** `>=0.1` -> `>=0.1.0` in all four. The two spellings are indistinguishable
  > to a resolver, so this is not a behaviour change; it is a record that the bound has
  > been reviewed against a version that will exist. `pfsmgraph-align>=0.1` was left
  > untouched in `hmm`, `hseg` and `dl` deliberately -- the divergent spelling now marks
  > which bounds have been reviewed and which have not, and `align`'s is decided at
  > `align`'s own release with evidence this branch does not have.
  > **Not changed:** `dataseq`'s own `numpy>=1.24`. Its entire numpy surface is `ndarray`,
  > `zeros`, `full`, `empty`, `asarray`, `array`, `array_equal` and `np.int*` -- ancient
  > API, so 1.24 is conservative rather than earned. Left as is: too low breaks users,
  > too high only excludes them, and 1.24 is the first numpy supporting Python 3.11
  > against a `requires-python = ">=3.10"`.
  - [x] Confirm the workspace source is not what makes it resolve
  > **Measured, not inferred.** Outside the workspace against real PyPI,
  > `uv pip compile` on `pfsmgraph-dataseq>=0.1` fails: *"Because only
  > pfsmgraph-dataseq==0.0.0 is available and you require pfsmgraph-dataseq>=0.1, we can
  > conclude that your requirements are unsatisfiable."* Unbounded, it resolves to the
  > `0.0.0` placeholder. PEP 440 confirms the local half: `>=0.1` excludes `0.1.0.dev0`
  > **even with `prereleases=True`**, because a `.devN` sorts strictly below the final --
  > an ordering fact, which no pre-release flag can override. So the declared bound is
  > satisfiable by nothing anywhere, and `uv sync` succeeds only because the workspace
  > source short-circuits version resolution entirely. ADR 0006's footgun, caught live.
  > **And the lockfile cannot catch it.** `uv.lock` is byte-identical across this change:
  > a workspace member's `requires-dist` entry records `{ name = "pfsmgraph-dataseq",
  > editable = "packages/pfsmgraph-dataseq" }` with **no version specifier at all**. The
  > one committed artifact that normally catches dependency drift is structurally blind to
  > this class of error, which is why the `DEFERRED.md` entry recurs forever rather than
  > clearing.

- [ ] Release: bump, build, publish, tag
  - [ ] Drop `.dev0` from `pfsmgraph-dataseq` alone, leaving the other four untouched
  - [ ] Add a README and a LICENSE file to `pfsmgraph-dataseq` and make the sdist carry them, not `.gitignore`
  - [ ] `uv build --package pfsmgraph-dataseq` and inspect both artifacts before anything leaves the machine
  - [ ] Update the two `0.0.0`-placeholder claims in `README.md` (lines 5 and 67) in the publishing commit itself
  - [ ] Publish to PyPI — the user runs this; it is irreversible
  - [ ] Tag `pfsmgraph-dataseq-v0.1.0` by hand and push the tag
  - [ ] Close the `DEFERRED.md` entries this branch discharged, leaving the recurring lower-bounds one open
