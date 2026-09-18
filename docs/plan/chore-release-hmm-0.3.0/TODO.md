# chore/release-hmm-0.3.0

**Status**: merged — PR #53 — 2026-09-18
**Created**: 2026-09-18
**Subgoal**: Release `pfsmgraph-hmm` 0.3.0 — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [x] Verify what 0.3.0 adds to what ships, rather than assuming it
  > **Q:** What should 0.3.0 do with `_trials._suggest_split` and `_suggest_merge` — keep
  > with the reason recorded in the plan, keep with the reason in their docstrings, or
  > remove both?
  > **A:** Keep, reason in the plan. No shipped-code change.
  > **Done:** What ships differs from 0.2.0 by four new private modules (`_mdl`,
  > `_search`, `_topology`, `_trials`) and two public changes: the backend table,
  > `__init__.py` and both `.pyx` are unchanged, the header is byte-identical, and the
  > invariant holds. §12's prediction is confirmed rather than assumed.
  > **Note:** *(Corrected in goal 2.)* This read "and nothing else a consumer can reach",
  > and so does the body of `0f671c0`, which is pushed and stays as written. It was
  > checked against `__all__` and the backend table, which see a new name or row and not a
  > new keyword. A signature diff across the tag found `baum_welch`'s `score_backend=` and
  > `min_cycles=`, and `HMMParams.state_p` changed behaviour: a reducible chain now always
  > raises, and transient states are exactly `0.0`. All three are in `CHANGELOG.md`.
  - [x] No new ADR 0002 lifecycle phase or backend row, as `HMMLIB-ACCOUNT.md` §12
    predicts; the ADR 0003 backend header byte-identical to 0.2.0's
    > **Note:** `git diff pfsmgraph-hmm-v0.2.0..HEAD` over the package's `src/`,
    > `meson.build` and `pyproject.toml` touches twelve files: the four new modules and
    > their `install_sources` entries, the `.dev0` version, and in the five existing
    > modules docstrings and code. **Of the kernels, only comments moved** —
    > `_viterbi.py` (one tense), `_viterbi_cuda.py` and `_forward_backward_cuda.py` (the
    > `_THREADS_PER_BLOCK` comment); `_baum_welch.py` gained `min_cycles=` and
    > `score_backend=`, `_numeric.py` gained `closed_classes`, and `_params.py` gained
    > `_check_codes` (moved from `_baum_welch`) and `state_p`'s exact transient zeros —
    > none of them a phase or a row. `_backends.py`, `__init__.py`, both `.pyx`, the root `conftest.py` and
    > `_backends.py`, and `dp-compile.toml` are byte-identical to the tag.
    > **Note:** The header is 307 bytes and `cmp`-identical to the literal `core.md`
    > carried at the tag, rejoined across its line wrap. That is two grounds, not one: a
    > recorded 0.2.0 header matches a live one today, *and* every input that forms it is
    > unchanged. The test-side `packages/pfsmgraph-hmm/tests/conftest.py` did change — it
    > gained the `score_backend` fixture over `_TABLE["forward_backward"]` — but the
    > header comes from the root `pytest_report_header`, which reads neither. Watch for
    > `pytest --co -q`: `-q` suppresses report headers, so a grep of it finds nothing and
    > reads like an empty matrix.
  - [x] The four-file release invariant (member README, `LICENSE` file, `Typing :: Typed`,
    `py.typed` in `install_sources`) and `license-files = ["LICENSE"]` still intact
    > **Note:** All five hold in the tree: `LICENSE` a regular file `cmp`-identical to the
    > root one, the classifier at `pyproject.toml:40`, `license-files` at `:13`, `py.typed`
    > at `meson.build:94`, and `tests/test_meson_sources.py` 16/16. No `VERSION` at the
    > root and no namespace `__init__.py`. Only `README.md` changed since the tag (+7/-1),
    > which goal 2 re-reads. This checks the *tree*; goal 3's clean-venv install is what
    > checks the wheel.
    > **Note:** `meson.build:31-32` still reads "py.typed arrives at the release commit",
    > written before 0.1.0 when it had not. It ships in the sdist, so goal 2 is the place
    > to put it in the past tense if it is worth the line.
  - [x] Decide `_trials._suggest_split` and `_suggest_merge`, which only the tests call
    since `_suggest_move` ranks every candidate in one stable sort: remove, or keep with a
    stated reason
    > **Note:** Kept. **The premise was incomplete**: the tests are not the only caller.
    > `.scratch/hmm-lush/measurements/merge_round_cost.py` imports `_suggest_merge` and
    > times a whole round through it, and ADR 0023 (§6 and Evidence) and ADR 0024 cite that
    > script, so removal would break evidence behind two Accepted records. The stated
    > reason has three parts. They are the ports of `suggest-split` and `suggest-merge`,
    > which exist in the original, and the only calls that rank one kind alone.
    > `_suggest_move` cannot be composed from them, since a merge wins a tie only because
    > both kinds share one stable sort. And they are the seam `test_trials.py` pins
    > per-kind enumeration order, trial counts and the transient-pair exclusion through.
    > Under ADR 0025 they are out of contract, so keeping them freezes nothing. This is
    > the third time the question has opened (`feat-hmm-scored-primitives`,
    > `chore-hmm-0.3.0-audit`, here) — point the next audit at this note.
- [x] Settle what 0.3.0's immutable text says
  > **Done:** Every text 0.3.0 freezes or points at now says what ships. The new
  > `CHANGELOG.md` is backfilled to 0.1.0 and opens on ADR 0025 §7. The PyPI summary
  > reads "topology search included but not yet public". The root README's already-false
  > claims are fixed, with the PyPI-dependent ones drafted for the record commit.
  > `codex.md` no longer overrides a current `core.md` in four places. Two false claims
  > were corrected along the way, with dated notes: goal 1's "nothing a consumer can
  > reach" and `core.md`'s "a coincidence a test pins against".
  > **Q:** Where should hmm 0.3.0's release notes live — a package `CHANGELOG.md` backfilled
  > to 0.1.0, one starting at 0.3.0, a GitHub Release on the tag, or the member README's
  > existing ADR 0025 paragraph alone?
  > **A:** `packages/pfsmgraph-hmm/CHANGELOG.md`, with a 0.3.0 entry and short 0.2.0 and
  > 0.1.0 entries rebuilt from the record. Reviewed in this PR, shipped in the sdist,
  > linked from the README; the other four members adopt it at their first release.
  - [x] Release notes meeting ADR 0025 §7 — topology search ships with no supported entry
    point, said plainly. No `CHANGELOG` exists and 0.2.0 wrote none, so decide where they
    live before writing them
    > **Note:** `packages/pfsmgraph-hmm/CHANGELOG.md`, linked from the member README and
    > from `[project.urls]` as `Changelog`. 0.3.0 opens with the §7 sentence in bold, ahead
    > of any list. The member README already met §7 before this, since `f511026`
    > (PR #52), so what this adds is the change list. 0.2.0 and 0.1.0 were rebuilt from
    > each tag's `__all__`, signatures and `pyproject.toml`, not from memory. The file has
    > **no fenced code blocks** on purpose: `tests/test_api_docs.py` reads `README.md` and
    > `docs/api/`, so a block here would be the one unexecuted example in the package.
    > Both links point at `main` and return 404 from the tag push until this PR merges.
    > `uv lock --check` is unaffected, since the lockfile records no project URLs.
  - [x] The root `README.md`'s status line: "released at **0.2.0**" and the search as a
    later revision turn false together at the release commit
    > **Q:** When should the root README's `hmm` statements change — split by truth
    > (already-false ones now, "released" ones at the record commit from text drafted
    > here), all at the record commit, or all now?
    > **A:** Split by truth.
    > **Note:** The premise was half right. The "released" claims turn false at the
    > *publish*, not the release commit, since the root README ships in no artifact and
    > names a PyPI fact. Two other claims were false already. Line 16's table row still read
    > "Baum-Welch … to follow", stale since 0.2.0 shipped it. Line 5's "a later revision"
    > was stale since PR #48 implemented the search. Both are fixed now, in wording that
    > carries no version claim (the row) or is future tense until the tag (line 5).
    > **Note:** Drafted for goal 4's record commit, to apply verbatim once PyPI confirms:
    > line 5 `released at **0.2.0** (2026-09-16)` → `released at **0.3.0** (<publish
    > date>)`; line 5 `it will ship private in 0.3.0, with no supported entry point` →
    > `0.3.0 ships it private, with no supported entry point`; line 71
    > `` `pfsmgraph-hmm` at 0.2.0 `` → `` `pfsmgraph-hmm` at 0.3.0 ``. The sentence calling
    > 0.2.0 the first release of platform wheels stays, being history rather than status.
  - [x] Refresh `codex.md`'s review guide, which still describes `hmm` as "three modules
    and 167 tests" (2026-09-04)
    > **Q:** How far should this take `codex.md` — fix what contradicts `core.md` and point
    > at it, contradictions only, or a full refresh with revision 03 and 04 targets?
    > **A:** Fix the contradictions, replace the overview's counts with a pointer to
    > `AGENTS.md`'s current state, and add one short block of revision-04 false positives
    > for reviewing this release. Rebuild `AGENTS.override.md`.
    > **Note:** The named line was one of four passages that contradicted `core.md`. That
    > matters more than staleness usually does, because `AGENTS.override.md` outranks
    > `AGENTS.md`, so a wrong claim in `codex.md` beats a right one in `core.md` for Codex.
    > The worst was the backend matrix. It still said the suites were not parameterized,
    > which told Codex *not* to flag a one-backend test of a public call; since ADR 0021 that
    > is exactly the finding. The other two: `hmm`'s four release files, and a `LICENSE`
    > claim written in the hatchling era that meson-python falsified.
    > **Note:** **`core.md` carried a false claim, now corrected.** It said sorting moves on
    > `(total, kind)` instead of the total alone was "a coincidence a test pins against".
    > Mutated, that key passes all 84 `_trials` and `_search` tests: `_ranked` is always
    > handed merges first, so no input tells the two keys apart. The first draft of the new
    > revision-04 block repeated the claim from `core.md`, and the mutation run caught it
    > before it was written. No test was added. A test feeding `_ranked` a split ahead of
    > an equal-total merge would pin it, if that is ever wanted.
  - [x] Re-read `packages/pfsmgraph-hmm/README.md` and the metadata (`description`,
    classifiers, extras) as the 0.3.0 PyPI page will show them
    > **Q:** What should 0.3.0's one-line PyPI summary say about topology search — "included
    > but not yet public", drop the clause, or keep "forthcoming"?
    > **A:** "topology search included but not yet public".
    > **Note:** The `description` was the only field wrong for 0.3.0. The classifiers
    > match the cp310–cp314 wheels and keep `Typing :: Typed`, the extras are unchanged
    > since 0.2.0, and `[project.urls]` gained `Changelog` in goal 2's first subgoal. The
    > member README already stated ADR 0025 §7 since PR #52. It does not mention
    > `score_backend=` or `min_cycles=`, correctly: its Training section shows the call,
    > and `docs/api/hmm/` and the CHANGELOG carry the keywords. `meson.build:31-32` no
    > longer says `py.typed` "arrives at the release commit" (goal 1's note), since that
    > comment ships in the sdist. There are no `Topic ::` or `Operating System ::`
    > classifiers, and 0.2.0 had none either. That is a finding rather than a gap this
    > release must close.
- [x] Verify what a consumer installs
  > **Done:** Run `35301984370` built all twenty wheels and the sdist from `3316ff1` and
  > skipped `publish`. The sdist is byte-identical to one built locally from the same
  > commit. The `cp312` Linux wheel, installed alone into a clean venv with
  > `pfsmgraph-dataseq` 0.1.0 from PyPI, carries the four files, `license-files`, the new
  > summary and the ten names. Its Cython kernels are bit-identical to the numpy reference
  > in 330 of 330 comparisons.
  - [x] CI dry run of the twenty wheels and the sdist from this branch, as 0.2.0's run
    35049292620 did, with the temporary trigger removed before merge
    > **Q:** Trigger the dry run by `workflow_dispatch` against this branch, or by a
    > temporary push trigger as 0.2.0 did?
    > **A:** `workflow_dispatch`: `gh workflow run release.yml --ref chore/release-hmm-0.3.0`.
    > **Note:** The temporary trigger this subgoal names is superseded, not skipped. 0.2.0
    > needed one because GitHub offers `workflow_dispatch` only for a workflow already on
    > the default branch, and `release.yml` was not yet on `main`. It has been since the
    > 0.2.0 merge, and it keeps `workflow_dispatch:`. A dispatched run takes the workflow
    > file from the ref and builds its tip. `publish` stays skipped, because its `if:`
    > requires a `refs/tags/pfsmgraph-hmm-v*` ref. Nothing is edited, so nothing needs
    > removing before merge, and later pushes to this branch do not fire the matrix.
    > **Ran (2026-09-18):** run `35301984370`, dispatched by the user against `3316ff1`.
    > **Success, 11m26s** (03:07:09 → 03:18:35 UTC), against 0.2.0's 7m48s. `publish to
    > PyPI` **skipped**. It produced twenty `0.3.0.dev0` wheels, five per platform on
    > cp310–cp314, with the same glibc 2.17 manylinux floor
    > (`manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64`), plus
    > `pfsmgraph_hmm-0.3.0.dev0.tar.gz`. Every wheel's test session opened on `cython ✓`
    > (20 of 20 headers), with `cpu_parallel`, `cuda` and `torch` absent and naming their
    > extras. **2378 passed / 574 skipped** on every interpreter, except two Windows ones
    > at 2379 / 573: the libm-vs-numpy `log2` test again, as in 0.2.0's run.
    > **Note:** **The sdist reproduces byte for byte**: sha256 `25f7f0b9…a2f82d` for both
    > CI's and `uv build --sdist` here from the same commit. It contains `CHANGELOG.md`,
    > with no `meson.build` entry needed, since the sdist is a `git archive`.
    > **Note:** Windows logs print the header **escaped**, `cython \u2713 \xb7 cpu_parallel
    > \u2717`, because its console encoding cannot carry the symbols. So a grep for
    > `cython ✓` counts **0** on Windows and reads like a missing kernel. Match
    > `cython (✓|\\u2713)` instead; all five Windows headers say available.
  - [x] Install into a clean venv outside the workspace; check the four files, the ten
    names, and the compiled kernels bit-exact against the numpy reference
    > **Result (2026-09-18):** a `uv venv` under the session scratchpad, outside the
    > repository, holding only the `cp312` manylinux x86_64 wheel from run `35301984370` and
    > what it resolved from PyPI: `pfsmgraph-dataseq` **0.1.0** and numpy **2.5.3**. That numpy
    > is newer than the workspace's 2.4.6, which the `numba-cuda` cap holds back, so the
    > kernels also ran against a numpy the workspace never exercises. `sys.path` held no
    > workspace entry, and `hmm` imported from the venv's `site-packages`. Metadata: version
    > `0.3.0.dev0`, the new summary, `dist-info/licenses/LICENSE` with `License-File:
    > LICENSE`, `pfsmgraph/hmm/py.typed`, `Typing :: Typed`, the README as long description,
    > the `Changelog` URL, and no namespace `__init__.py`. `__all__` is the ten names.
    > `backends()` reports `python ✓ · cython ✓`, with the three optional backends absent.
    > **Cython against python: 330 comparisons, 0 not bit-identical.** Those are `viterbi`
    > and `viterbi_batch` over 40 random models of 1–8 states, empty records included, and
    > `baum_welch` with `score_backend="cython"` compared on its cycles, its
    > description-length trace and all three parameter arrays. The public API only, by a
    > scratchpad script run once first against the workspace so that a failure here would
    > point at the wheel rather than the script.
    > **Note:** **The workspace venv's own metadata is stale, and that is a second reason
    > this check needs a clean venv.** Rehearsing the check script under `uv run`,
    > `importlib.metadata` reported `pfsmgraph-hmm` **0.1.0**, with 0.1.0's summary, no
    > license file and no `py.typed`. That comes from `.venv/.../pfsmgraph_hmm-0.1.0.dist-info`,
    > written 2026-09-14 and not rewritten through three version changes since. The
    > meson-python editable finder serves live source, so imports stay current. Nothing
    > rewrites the `dist-info` beside it, and `uv sync` did not reinstall on a version
    > change. Nothing in `src/` or the tests reads that metadata (no `__version__`, no
    > `importlib.metadata` call), so it affects nothing today. It would mislead the first
    > code that asks the installed version.
- [x] Release: publish, tag, and close what the branch discharged
  > **Done:** `pfsmgraph-hmm` 0.3.0 is on PyPI as of 2026-09-18 03:40 UTC, tagged
  > `pfsmgraph-hmm-v0.3.0` at `78f04a3`. It is twenty wheels and an sdist from run
  > `35303614122`, each with a PEP 740 attestation, and the sdist digest was predicted
  > before the tag. `0.4.0.dev0` followed in the next commit, `67b887f`. `core.md` and the
  > root README now say released at 0.3.0. The master plan's item closes at `/smart-merge`.
  - [x] Release commit: `version` to `0.3.0`, date the `CHANGELOG.md` 0.3.0 heading, relock,
    rebuild and re-verify — then
    `0.4.0.dev0` straight after, closing the window `core.md` "Versioning" names
    > **Note:** **The order is forced, so the window cannot be zero.** `release-ci` tags
    > HEAD, and its `_version-matches` prerequisite reads the version from the tree. So
    > `0.4.0.dev0` cannot share the tagged commit. It is the next commit, made right after the
    > tag push, which narrows the window to the gap between two commits. 0.1.0 and 0.2.0
    > left days.
    > **Note:** The release commit changes five lines and nothing else: `version` to
    > `0.3.0`, the lockfile's one matching line (`uv lock`: "Updated pfsmgraph-hmm
    > v0.3.0.dev0 -> v0.3.0"), the CHANGELOG heading dated 2026-09-18, `core.md`'s
    > Versioning history, and the regenerated `AGENTS.md`. `uv lock --check`, the agents-docs
    > sync check and the 52 repo-root tests pass. `release-ci` runs the full suite itself as
    > a prerequisite. The content is otherwise `3316ff1`'s, which run `35301984370` built
    > and the clean venv verified.
    > **Result (2026-09-18):** the release commit is `78f04a3`. Its sdist, built here, reads
    > `Version: 0.3.0` in both `PKG-INFO` and the packed `pyproject.toml`, and carries
    > `## 0.3.0 — 2026-09-18`. Its sha256 is
    > `746cd411ad01032ba8381d4fd599f9af3e9359c82875801596c661b8ddecceeb`, the digest PyPI's
    > copy should match. The bump to `0.4.0.dev0` is the next commit, pushed minutes after
    > the tag while the publish run was still building. That run builds from the tag, which
    > the branch tip moving cannot change.
  - [x] **User:** publish (irreversible): `just release-ci 0.3.0` pushes the
    `pfsmgraph-hmm-v0.3.0` tag that triggers the publish job
    > **Ran (2026-09-18):** `just release-ci 0.3.0`, by the user. All four prerequisites
    > held. The full suite passed, **3099 in 5m40s**, with every backend available on this
    > host. `_tree-pushed` pushed `e4752de..78f04a3`. The tag `pfsmgraph-hmm-v0.3.0` points
    > at `78f04a3`, locally and on origin (`git ls-remote`). The push started run
    > `35303614122`, event `push` on the tag.
  - [x] Confirm PyPI's digests match the verified build and the tag points at the release
    commit
    > **Result (2026-09-18):** run `35303614122` (tag push, 03:32:50 → 03:40:49 UTC) built
    > and published. Its publish job's assertion printed "all 21 artifacts carry version
    > 0.3.0", and PyPI shows 0.3.0 uploaded at 03:40:18 UTC with 20 wheels and 1 sdist, not
    > yanked. **The sdist's sha256 is `746cd411…ddecceeb`, the digest the local build of
    > `78f04a3` predicted before the tag existed.** All 21 PyPI digests equal the run's
    > downloaded artifacts file for file. All 21 files have a PEP 740 provenance attestation
    > (`/integrity/…/provenance` returns 200), published by `GitHub
    > PanosMavromatis/pfsmgraph`, workflow `release.yml`, environment `pypi`. The tag is
    > `78f04a3`, locally and on origin. Installed from PyPI with `--refresh --no-cache` into
    > a fresh venv, the consumer check passes again: version 0.3.0, the four files, ten
    > names, `cython ✓`, and 330 of 330 bit-identical against the numpy reference.
    > **Note:** The subgoal's "verified build" could not be the dry run's own artifacts,
    > which carry `0.3.0.dev0` and so differ in every byte that names the version. What
    > makes the verification carry over is the reproducible sdist. The dry run established
    > that CI's sdist and a local one agree byte for byte. That made the release sdist's
    > digest predictable, and the prediction held.
  - [x] Record commit: the "released" statements in `core.md`, the root README and the
    master plan
    > **Note:** `core.md` now says `hmm` is released at 0.3.0 (2026-09-18), in its headline
    > and in the release history, where one passage records the private search, the first
    > CHANGELOG, the predicted digest and the attestations. The root README took goal 2's
    > three drafted edits verbatim. **The master plan is left to `/smart-merge`**, whose
    > step 7 marks the backlinked item `[x]` and writes its `Done:` line with the PR
    > number. Editing it here would touch that item twice, the first time without the one
    > fact the second adds.
