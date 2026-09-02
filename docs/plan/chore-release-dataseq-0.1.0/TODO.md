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

- [~] Release: bump, build, publish, tag
  > **Q:** The member ships no README and no LICENSE, so its PyPI page would be blank.
  > Write a member-specific README, copy the root one, or ship none? And add
  > `[project.urls]`?
  > **A:** A new member-specific README, aimed at a PyPI visitor rather than at this
  > repository, plus `[project.urls]` for Homepage/Repository/Documentation.
  > **Not a question -- measured.** The LICENSE must be a real copy, not a symlink to the
  > root one. Hatchling writes a symlink verbatim into the sdist and `uv build` then
  > refuses to unpack it: *"symlink destination for ../../LICENSE is outside of the target
  > directory"*. That is a **consumer-side** failure, caught here only because `uv build`
  > builds the wheel *through* the sdist; a backend building both from the source tree
  > would have shipped a permanently broken sdist under an immutable version number.
  > Real copies need no configuration at all -- hatchling's PEP 639 default glob finds
  > `LICENSE` unaided and emits `License-File:` plus `dist-info/licenses/LICENSE`.
  > **Q:** Another session advised adding a `py.typed` file. Is it needed, and where?
  > **A:** Yes, added -- but at `src/pfsmgraph/dataseq/py.typed`, not the distribution root
  > the advice named. See the subgoal below; the advised path is measurably inert.
  - [x] Drop `.dev0` from `pfsmgraph-dataseq` alone, leaving the other four untouched
  > `0.1.0.dev0` -> `0.1.0` in `packages/pfsmgraph-dataseq/pyproject.toml` only; the other
  > four still read `0.1.0.dev0`, which is the per-package scheme working as intended.
  - [x] Add a README and a LICENSE file to `pfsmgraph-dataseq` and make the sdist carry them, not `.gitignore`
  > **Both added, and neither needed configuration for the license.** Hatchling's PEP 639
  > default glob finds a `LICENSE` in the member root unaided, so the wheel gained
  > `dist-info/licenses/LICENSE` and the metadata a `License-File:` line from the copy
  > alone. `readme = "README.md"` supplies the long description; METADATA went 463 ->
  > 4443 bytes. `[project.urls]` was added at the same time, because a PyPI page with a
  > body but no link off it is the same blankness one layer down.
  > **The README is member-specific, not a copy of the root one**, and is written for a
  > PyPI visitor rather than for this repository: what the distribution is, why it imports
  > neither torch nor pandas, strict encoding with the pasted `KeyError`, the fixed
  > reserved block, and ragged records with padding confined to `pad_collate`. Every link
  > is absolute, since a relative one is a 404 on PyPI.
  > **`tests/test_api_docs.py` now covers it.** Its glob was `docs/api/*/*.md`, so the one
  > surface where drift is *immutable* was the one surface unguarded. Discovery now also
  > takes `packages/*/README.md`; suite 91 -> 92. It earned its place on the first run,
  > failing the draft twice: the examples had been transcribed from `print()` output
  > (`[6 7 8]`) where a `>>>` line shows the repr (`array([6, 7, 8], dtype=int32)`).
  > **`.gitignore` still ships**, and that half of the goal-1 finding is left alone
  > deliberately: it is the repo-root file, which hatchling includes in every sdist so the
  > exclusion rules travel with it. Harmless, and not ours to remove.
  - [x] `uv build --package pfsmgraph-dataseq` and inspect both artifacts before anything leaves the machine
  > **Inspected, then exercised.** Both artifacts pass `twine check`. The sdist carries
  > `src/`, `tests/`, `LICENSE`, `README.md`, `pyproject.toml`, `PKG-INFO`; the wheel
  > carries the six modules plus `dist-info/licenses/LICENSE`.
  > **The check that actually decides it** is not the listing: the wheel was installed into
  > a clean venv outside the workspace, where it pulled only numpy, reported
  > `version: 0.1.0`, reproduced every README example byte for byte, and imported neither
  > torch nor pandas. `pfsmgraph.__file__` is `None` there -- the PEP 420 namespace
  > invariant holds in the shipped artifact, not merely in the source tree, which is the
  > form of it that matters to the four members that must share the namespace later.
  > **Rebuilt 2026-09-02 to add the PEP 561 marker**, `src/pfsmgraph/dataseq/py.typed`,
  > plus the `Typing :: Typed` classifier. Without the marker a type checker discards every
  > annotation in the six modules even though the source is fully annotated: measured
  > against the wheel in a clean venv, `vocab.size` revealed as `Any` and a deliberate
  > `bad: str = vocab.size` **was accepted silently**. With it, `int` and `list[str]`, and
  > the bad assignment is caught. Wheel 10 -> 11 files; `twine check` still passes both.
  > **The path was the whole question.** The advice named the *distribution* root,
  > `packages/pfsmgraph-dataseq/py.typed`; built that way the wheel contains zero `py.typed`
  > entries -- no error, no warning, simply inert, because the file is in no package and
  > `packages = ["src/pfsmgraph"]` never sees it. Inside the importable package it ships
  > with no `pyproject.toml` change at all, hatchling including every file under that tree.
  > **PEP 420 makes the placement load-bearing rather than conventional:** no single
  > distribution owns the `pfsmgraph/` level -- the same reason no `__init__.py` may sit
  > there -- so a marker at the namespace level would be one member claiming typedness on
  > behalf of four it does not ship. Each member marks its own regular subpackage.
  > **Timing is why it was worth stopping for.** `py.typed` is wheel content: adding it
  > after publishing reaches users only as `0.1.1`, leaving `0.1.0` on PyPI permanently as
  > the version whose types do not work.
  - [x] Update the two `0.0.0`-placeholder claims in `README.md` (lines 5 and 67) in the publishing commit itself
  > Both rewritten, plus the test count on line 56 (91 -> 92) that this branch's own
  > verifier change falsified. Note the ordering these three share: each is false for the
  > minutes between this commit and the publish, and true afterwards. The alternative --
  > committing after publishing -- leaves a window in which the tag exists and the front
  > page still says nothing has been released, which is the worse direction to be wrong in.
  - [ ] Publish to PyPI — the user runs this; it is irreversible
  > **Ready and unpublished.** `dist/pfsmgraph_dataseq-0.1.0.{tar.gz,whl}` are built,
  > `twine check`-clean, and verified by clean-venv install. Everything up to this line is
  > reversible; this line is not.
  > **The command is `uv publish dist/pfsmgraph_dataseq-0.1.0*`, with the files named.**
  > `uv publish` has no `--package` flag -- it takes file globs, defaulting to `dist/*`,
  > and `dist/` is one directory shared by all five members. Naming the files is the habit;
  > a **project-scoped** PyPI token is what makes forgetting it survivable, turning a stray
  > glob into a 403 instead of a burnt version number on an unready package.
  > **Nothing configures the endpoint**, verified by `--dry-run`: no `[[tool.uv.index]]` in
  > any of the six `pyproject.toml` files, no `uv.toml` at repo or user level, no
  > `UV_PUBLISH_URL`, so uv uses its default `https://upload.pypi.org/legacy/`. Auth is
  > unconfigured too and must be passed at the call: uv tries trusted publishing (OIDC)
  > first, which resolves only inside CI, so from a laptop that failure is expected rather
  > than a misconfiguration. There is no `~/.pypirc`, and uv does not read one.
  - [ ] Tag `pfsmgraph-dataseq-v0.1.0` by hand and push the tag
  - [x] Close the `DEFERRED.md` entries this branch discharged, leaving the recurring lower-bounds one open
  > **One closed, two partially discharged, one left open.** The prose sweep closes
  > outright. "Replace the `0.0.0` placeholders" and "Drop the `.dev0` suffix and tag"
  > are annotated rather than closed, because four members still owe both -- and the
  > `.dev0` entry now carries the two mechanics the version bump does not imply, the
  > README/LICENSE files and the measured fact that the LICENSE cannot be a symlink. The
  > lower-bounds entry stays open untouched; it recurs forever by design.
