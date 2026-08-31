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

- [x] Import the existing `dl` implementation into the scratch location
  - [x] The first implementation is readable in the scratch location and its provenance is recorded
  - [x] Identify any features that are `dl` specific and may not serve the other packages; propose how to address this
  - [x] Name the essential gaps in the implementation, to be filled during or after the merge
  > **Note:** "`dl`" names a slot in this repository's package family, not the source project. The implementation comes from **MelodyHPO**, a standalone and now defunct project that built dataset handling and DL models together without packaging them; the `dl` label is retroactive. The artefact is `.scratch/dl/MelodyHPO/melody_hpo/data/`.
  > **Done (first criterion):** MelodyHPO copied whole into `.scratch/dl/`; provenance recorded in `.scratch/README.md` (`main` at `5f42311`, 2026-03-23), along with the second repository the copy carries at `data/MelodyData` (`abe7625`). `.scratch/dl/.gitignore` is deny-by-default — 33 files (≈62 KB) tracked out of 2.2 GB on disk, all 14 rules verified with `git check-ignore` rather than assumed. Two renames were required, not cosmetic: `MelodyHPO/.git` → `.git-disabled`, because an embedded repository makes `git add` on paths inside it a *silent* no-op that exits `0`, so the branch would have merged with none of the merge base in it; and `MelodyHPO/CLAUDE.md` → `CLAUDE.md.orig`, because Claude Code loads a nested `CLAUDE.md` when reading files beside it, and presence on disk rather than tracking is what triggers that.
  > **Q:** Where should the written analysis for this goal live -- `.scratch/dl/ANALYSIS.md`, a note under `docs/design/`, or inline in this plan?
  > **A:** `.scratch/dl/ANALYSIS.md`. Goals 3 and 4 already call for their written comparisons in the scratch location, so all three read as a set; goal 8 already owns the retention decision, so this adds no new risk. `docs/design/` was rejected as pre-empting ADR 0010 and putting a working document in the authoritative tree.
  > **Q:** `PitchCode`'s structured codes emit 1207-13176; ADR 0011 wants dense user symbols from 6. Which does `dataseq` own -- the dense vocabulary index, the structured code, both selectable, or defer to goal 6?
  > **A:** The dense vocabulary index. `PitchCode`-style structured codecs are demoted to a symbol canonicaliser that runs *before* vocabulary assignment, so musical structure stays available without dictating the alphabet the whole family sees. Decisive argument: `hmm` builds V x V transition matrices, and V = 13,177 is ~1.74e8 entries (~1.4 GB float64) for an alphabet of a few dozen real symbols.
  > **Done:** Analysis written to `.scratch/dl/ANALYSIS.md`. Central finding: the base conflates two different kinds of encoder -- a dense vocabulary index (`control.py`, PAD/BOS/EOS at 0/1/2) and a stateless structured code (`PitchCode`, 100*chromatic + diatonic), fused into one `encoder_map`. Seven `dl`-specific features identified with proposals, the load-bearing one being that `dataseq` can satisfy "a stock `DataLoader` works" with **zero torch imports**, since `torch.utils.data.Dataset` is duck-typed and needs only `__len__`/`__getitem__`. Seven essential gaps named: the reserved block is both wrong and incomplete (`GAP` absent, which `align` cannot do without, and user symbols start at 3 -- a *third* offset, distinct from the proof-of-concept's 4 that `DEFERRED.md` anticipates); no frozen vocabulary, so train/test splits are inexpressible; nothing decodes; the vocabulary is neither persistable nor shareable. Also recorded: `dataseq`'s `dependencies = []` is very likely already correct, which closes a `DEFERRED.md` question in the negative. One forward-looking trap flagged -- `MiniCorpus` registers symbols by iterating a `set`, harmless today because `PitchCode.encode` is pure, but a reproducibility bug the moment codes are assigned by insertion order, since CPython randomises `str` hashing per process.
  > **Note:** Writing the analysis exposed a hole in the `.gitignore` committed one step earlier: `/*` swallowed `.scratch/dl/ANALYSIS.md`, so the file was invisible to `git status` and would have been silently lost. Fixed with an anchored `!/*.md`, verified not to reach into `MelodyHPO/`.

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
