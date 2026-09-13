# chore/release-hmm-0.1.0

**Status**: active
**Created**: 2026-09-13
**Subgoal**: Release `pfsmgraph-hmm` 0.1.0 via `just release 0.1.0 pfsmgraph-hmm` —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## Goals

- [ ] Remove the unimported `pfsmgraph-align` dependency and record why it returns later
  - [ ] ADR 0019 amending ADR 0009: a declared intra-family dependency lands with its first import; `hmm` may release before `align`
  - [ ] ADR 0009's Status pointer, the ADR index row, and the PRD rows that state the edge
  - [ ] `DEFERRED.md` entry, with a trigger at the alignment-seeded-topology version (expected 0.4.0)
  - [ ] Drop the bound and its `[tool.uv.sources]` entry, relock, and correct `docs/api/hmm/README.md` and `core.md`
- [ ] Commit the `justfile` default and sweep the documents that name the old one
  - [ ] Verify the hand edit, and correct `docs/ops/release.md`, `core.md` and `README.md` where they state the default
  - [ ] Settle where the release runs: the token recipes call macOS `security`, which this Linux host lacks
- [ ] Settle what 0.1.0's immutable metadata says
  - [ ] Whether to ship the `gpu` / `cpu-parallel` extras when no public call reaches an accelerated kernel
  - [ ] `description` claims Baum-Welch and topology search, neither of which 0.1.0 contains
  - [ ] Honest lower bounds on every intra-family dependency naming `pfsmgraph-hmm`
- [ ] Ship the four files and verify the first meson-built wheel
  - [ ] Member README (executed by `test_api_docs.py`), LICENSE copy, `Typing :: Typed`, `py.typed` in `install_sources`
  - [ ] Drop `.dev0`, build, and install the wheel into a clean venv outside the workspace
  - [ ] Re-measure `docs/ops/release.md`'s hatchling-era reproducibility findings
- [ ] Release: publish, tag, and close what the branch discharged
  - [ ] Publish (the user runs it; irreversible) and tag `pfsmgraph-hmm-v0.1.0`
