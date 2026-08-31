# feat/dataseq-merge

**Status**: active
**Created**: 2026-08-31
**Subgoal**: revision `01-dataseq-v0.1.0`, subgoals 1 (merge the three implementations)
and 2 (settle the encoder API and promote ADR 0010)

## Tasks

- [x] Create a scratch location where the three implementations will be imported to be analyzed and merged
  - [x] The location does not interfere with any build, should `uv` be run deliberately or accidentally: it sits **outside `packages/`**, which the workspace glob `members = ["packages/*"]` would otherwise claim as a member and fail on for want of a `pyproject.toml`
  - [x] It is not collected by `uv run pytest`, which has no `testpaths` and so walks the whole tree — a dot-prefixed name such as `.scratch/` is excluded by pytest's default `norecursedirs` (`.*`) with no config change; `_scratch/` would **not** be
  - [x] Its name is not one `.gitignore` already swallows (`lib/`, `build/`, `var/`, `share/`), since this code is meant to be committed on the branch
  > **Q:** Where should the scratch location live, and how should it escape pytest collection — `.scratch/` (zero config, dot-excluded), `scratch/` with a self-contained `conftest.py` setting `collect_ignore_glob`, or `scratch/` with `norecursedirs` added to the root `pyproject.toml`?
  > **A:** `.scratch/` at the repo root. Zero configuration to add now or remove at cleanup; deleting the directory removes every trace.
  > **Done:** `.scratch/` created with `README.md` and one subdirectory per implementation (`dl/`, `hmm-lush/`, `py-rudimentary/`, each with a `.gitkeep` so the committed tree matches the documented layout). Root `pyproject.toml` unchanged. All three criteria verified empirically rather than asserted: `uv sync` resolved 51 packages without claiming `.scratch` as a member; `git check-ignore` found no match, so the tree commits; and a deliberately-failing `test_canary.py` placed inside `.scratch/` left `uv run pytest` at "collected 0 items", while the identical file at the repo root produced `Interrupted: 1 error during collection` — demonstrating the exclusion works and that the failure mode it prevents is real. Both canaries removed. Verified against pytest 9.1.1, whose default `norecursedirs` was read from source as `["*.egg", ".*", "_darcs", "build", "CVS", "dist", "node_modules", "venv", "{arch}"]`.

- [ ] Import the existing `dl` implementation into the scratch location
  - [ ] The first implementation is readable in the scratch location and its provenance is recorded
  - [ ] Identify any features that are `dl` specific and may not serve the other packages; propose how to address this
  - [ ] Name the essential gaps in the implementation, to be filled during or after the merge

- [ ] Import the earlier `hmm` implementation (in Lush) and tabulate its divergence with the previous implementation
  - [ ] The second implementation is readable in the scratch location and its provenance is recorded
  - [ ] A semantic account of the Lush original **in its own terms** — container model, encoder shape, vocabulary handling — written before any translation, so the comparison is against the original rather than against our reading of it
  - [ ] Draft a close translation to Python alongside the original in the scratch location, noting each choice the account left open where Lush does not map mechanically
  - [ ] A written comparison against the `dl` base, and every point where that base must be overridden by this implementation, with why

- [ ] Import the last implementation (rudimentary, in Python) and tabulate its divergence with the merged implementation so far
  - [ ] The third implementation is readable in the scratch location and its provenance is recorded
  - [ ] A written comparison of container semantics, encoder shape, and vocabulary handling
  - [ ] Every point where the `dl`/`hmm` merged base must be overridden by this last implementation is named, with why

- [ ] Land the merged container in `packages/pfsmgraph-dataseq/`
  - [ ] `dl` version is the base; divergences resolved per the comparisons above
  - [ ] Conforms to `torch.utils.data.Dataset`; a stock `DataLoader` works without subclassing
  - [ ] No `pfsmgraph/__init__.py` introduced anywhere; PEP 420 namespace intact
  - [ ] `uv sync && uv run python -c "import pfsmgraph.dataseq"` succeeds, and the other four still import

- [ ] Settle and implement the encoder API
  - [ ] Constructor signature decided and recorded inline as Q&A
  - [ ] Spelling of the strictness switch decided; strict is the default, `UNK` fallback explicit opt-in
  - [ ] How `align` consumes the mapping at its boundary decided — encode-at-the-boundary must stay mechanical for Cython/CUDA
  - [ ] Reserved block hard-coded per ADR 0011: `PAD`=0 … `MSK`=5, user symbols from 6, not configurable
  - [ ] Encoder and decoder implemented; unseen symbols raise by default

- [ ] Promote ADR 0010 to `Accepted`
  - [ ] Record the settled API in the ADR's decision section
  - [ ] Status changed `Proposed` → `Accepted` with the date
  - [ ] Row updated in `docs/design/adr/README.md`

- [ ] Clean up once the merge is completed
  - [ ] All migration decisions are recorded in [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md), which is the merge's record; anything the encoder API decision does not cover gets a new ADR with the next free number and a row in `docs/design/adr/README.md`
  - [ ] Decide how the scratch code is retained **before** deleting it: a squash merge collapses the add and the delete to nothing and loses it from `main` entirely, so retention needs a merge commit, a tag on the pre-deletion SHA, or a branch left unmerged
  - [ ] Delete the scratch location
