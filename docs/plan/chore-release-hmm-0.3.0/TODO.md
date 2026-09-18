# chore/release-hmm-0.3.0

**Status**: active
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
- [ ] Verify what a consumer installs
  - [ ] CI dry run of the twenty wheels and the sdist from this branch, as 0.2.0's run
    35049292620 did, with the temporary trigger removed before merge
  - [ ] Install into a clean venv outside the workspace; check the four files, the ten
    names, and the compiled kernels bit-exact against the numpy reference
- [ ] Release: publish, tag, and close what the branch discharged
  - [ ] Release commit: `version` to `0.3.0`, date the `CHANGELOG.md` 0.3.0 heading, relock,
    rebuild and re-verify — then
    `0.4.0.dev0` straight after, closing the window `core.md` "Versioning" names
  - [ ] **User:** publish (irreversible): `just release-ci 0.3.0` pushes the
    `pfsmgraph-hmm-v0.3.0` tag that triggers the publish job
  - [ ] Confirm PyPI's digests match the verified build and the tag points at the release
    commit
  - [ ] Record commit: the "released" statements in `core.md`, the root README and the
    master plan
