# chore/release-hmm-0.3.0

**Status**: active
**Created**: 2026-09-18
**Subgoal**: Release `pfsmgraph-hmm` 0.3.0 — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Verify what 0.3.0 adds to what ships, rather than assuming it
  - [ ] No new ADR 0002 lifecycle phase or backend row, as `HMMLIB-ACCOUNT.md` §12
    predicts; the ADR 0003 backend header byte-identical to 0.2.0's
  - [ ] The four-file release invariant (member README, `LICENSE` file, `Typing :: Typed`,
    `py.typed` in `install_sources`) and `license-files = ["LICENSE"]` still intact
  - [ ] Decide `_trials._suggest_split` and `_suggest_merge`, which only the tests call
    since `_suggest_move` ranks every candidate in one stable sort: remove, or keep with a
    stated reason
- [ ] Settle what 0.3.0's immutable text says
  - [ ] Release notes meeting ADR 0025 §7 — topology search ships with no supported entry
    point, said plainly. No `CHANGELOG` exists and 0.2.0 wrote none, so decide where they
    live before writing them
  - [ ] The root `README.md`'s status line: "released at **0.2.0**" and the search as a
    later revision turn false together at the release commit
  - [ ] Refresh `codex.md`'s review guide, which still describes `hmm` as "three modules
    and 167 tests" (2026-09-04)
  - [ ] Re-read `packages/pfsmgraph-hmm/README.md` and the metadata (`description`,
    classifiers, extras) as the 0.3.0 PyPI page will show them
- [ ] Verify what a consumer installs
  - [ ] CI dry run of the twenty wheels and the sdist from this branch, as 0.2.0's run
    35049292620 did, with the temporary trigger removed before merge
  - [ ] Install into a clean venv outside the workspace; check the four files, the ten
    names, and the compiled kernels bit-exact against the numpy reference
- [ ] Release: publish, tag, and close what the branch discharged
  - [ ] Release commit: `version` to `0.3.0`, relock, rebuild and re-verify — then
    `0.4.0.dev0` straight after, closing the window `core.md` "Versioning" names
  - [ ] **User:** publish (irreversible): `just release-ci 0.3.0` pushes the
    `pfsmgraph-hmm-v0.3.0` tag that triggers the publish job
  - [ ] Confirm PyPI's digests match the verified build and the tag points at the release
    commit
  - [ ] Record commit: the "released" statements in `core.md`, the root README and the
    master plan
