# chore/release-hmm-0.1.0

**Status**: active
**Created**: 2026-09-13
**Subgoal**: Release `pfsmgraph-hmm` 0.1.0 via `just release 0.1.0 pfsmgraph-hmm` —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## Goals

- [x] Remove the unimported `pfsmgraph-align` dependency and record why it returns later
  > **Q:** `DEFERRED.md` already carries the alignment-seeded revision under "`align` able to produce a multiple alignment", and gives it a trigger rather than a number on purpose. Add the dependency item under that trigger, or under a new heading naming 0.4.0?
  > **A:** Under the existing trigger, noting there that the revision is expected to be `hmm` 0.4.0, the first after 0.3.0's merge/split search.
  > **Q:** `hseg` and `dl` declare the same unimported `pfsmgraph-align>=0.1`. How far does ADR 0019's rule reach now?
  > **A:** The rule is family-wide and checked at each member's release commit, when metadata becomes immutable; only `hmm`'s bound is removed on this branch.
  > **Q:** ADR 0019's Open section: when the edge returns, is it a hard dependency or an extra?
  > **A:** Most likely an extra in 0.4.0; by the first major release (1.x) alignment will probably be so integral to HMM training that it is required. Recorded as the expected answer in ADR 0019's Open section and in the `DEFERRED.md` entry, not as a decision.
  - [x] ADR 0019 amending ADR 0009: a declared intra-family dependency lands with its first import; `hmm` may release before `align`
    > **Note:** Written as `0019-declared-dependencies-follow-imports.md`. Its Open section raises a question the deferral would otherwise settle by default: whether the edge returns at 0.4.0 as a hard dependency or as an extra, since the alignment seed replaces merge/split's starting point rather than the search.
  - [x] ADR 0009's Status pointer, the ADR index row, and the PRD rows that state the edge
    > **Note:** PRD annotated at §1.5, §3.3, §3.4 and §11 as dated qualifications, in the form §8 already uses. §5.5's example `pyproject.toml` is deliberately untouched: it shows how to declare an edge, not which edges exist.
  - [x] `DEFERRED.md` entry, with a trigger at the alignment-seeded-topology version (expected 0.4.0)
  - [x] Drop the bound and its `[tool.uv.sources]` entry, relock, and correct `docs/api/hmm/README.md` and `core.md`
    > **Note:** `uv lock` rewrote exactly two lines, `hmm`'s `dependencies` and `requires-dist` entries for `pfsmgraph-align`, so the lockfile sees an edge vanish even though it never sees a bound change (measured 2026-09-01). Suite green at 318. The pyproject carries a comment naming ADR 0019 so the PRD §5.5 pattern is not restored by copying. Root `README.md` line 70 also stated the drawn release order and was corrected, and `core.md`'s "seventeen records" was already one stale and now reads nineteen.
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
