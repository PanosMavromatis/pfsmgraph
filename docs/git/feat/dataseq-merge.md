# feat/dataseq-merge

**Created**: 2026-08-31
**Base**: main at 73e15a0
**Status**: active

## Purpose

Implement `pfsmgraph.dataseq` by merging the three data sequence implementations that
already exist outside this repo, taking the `dl` version as the base (PRD §3.5), and
settle the encoder API that the merge forces a decision on. This is the base layer of
the family: the other four packages are blocked on the symbol↔code encoder this branch
delivers, so the API settled here is the one `align`, `hseg`, `hmm`, and `dl` all consume
at their boundaries.

Executes the first two subgoals of revision `01-dataseq-v0.1.0`. They are on one branch
because they are mutually dependent — the merge surfaces the constraints the API has to
satisfy, and the API decision determines which of the three implementations' shapes
survives. Splitting them would mean landing an encoder and immediately rewriting it.

## Scope

- Bring the three implementations into the repo and read them side by side; tabulate
  where they diverge on container semantics, encoder shape, and vocabulary handling.
- Land the merged container in `packages/pfsmgraph-dataseq/pfsmgraph/dataseq/`, with the
  `dl` version as the base: `Dataset` conformance such that a stock `DataLoader` works
  without subclassing.
- Settle the encoder API — constructor signature, the spelling of the strictness switch,
  and how `align` consumes the mapping at its boundary — and implement encoder/decoder
  against the fixed reserved block.
- Promote ADR 0010 `Proposed` → `Accepted` and update its row in `docs/design/adr/README.md`.
- Write `dataseq`'s API documentation under a repo-level `docs/api/`, and settle the layout
  and tooling that the other four members will inherit — added to the plan on 2026-08-31,
  after the plan was found to have no documentation goal at all.

**Not in scope**, though they belong to the same revision and get their own branches:
renumbering the proof-of-concept alignment code to the reserved block; the ADR 0003 test
suite; the `dependencies = []` fix; the 0.1.0 release.

## Context

- Master plan: `docs/plan/TODO.md`, revision `01-dataseq-v0.1.0` — this branch is
  backlinked under subgoals 1 and 2.
- [PRD §3.5](../../design/PRD.md) — names the `dl` implementation as the merge base.
- [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md) —
  `Proposed`, and `Proposed` *solely* because the encoder API is unresolved. Settling it
  here is what promotes it.
- [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) —
  fixed reserved block `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5, user symbols
  from 6, not configurable; strict encoding by default with `UNK` fallback opt-in. Settled;
  not up for relitigation on this branch.
- [ADR 0009](../../design/adr/0009-dataseq-as-the-base-layer.md) — why `dataseq` has no
  intra-family dependencies.
- [ADR 0008](../../design/adr/0008-per-package-build-backends.md) — `dataseq` presumes
  hatchling. If the merge turns up a compiled inner loop, that presumption breaks and the
  build-backend subgoal stops being a formality.
- `docs/plan/DEFERRED.md`, trigger "the `dataseq` merge" — the register of what this
  revision must not leave behind.

**Dependency:** each import goal waits on its own source arriving in `.scratch/`. The
merge base (goal 2) landed on 2026-08-31 — MelodyHPO, `main` at `5f42311`; the Lush and
rudimentary implementations (goals 3–4) are still outside the repo and will be marked
`[!]` if they have not arrived when their turn comes. Goal 1, creating the scratch
location, never depended on any of them and is complete.

## Notes

Running log — decisions made, things tried, things deferred.

- 2026-08-31 — Branch created from `main` at 73e15a0. No code imported yet.
- 2026-08-31 — Import goal split into a prelude, one goal per implementation in merge
  order, and a postlude. Under the Lush import, the semantic account of the original now
  precedes the translation: a close translation is defined by fidelity to something you
  must already understand, so translating first would run the comparison against our
  reading of the original rather than the original itself.
- 2026-08-31 — Merge base imported. It is **MelodyHPO**, not a project named `dl`: that
  label is this repository's package slot, applied retroactively to a standalone and now
  defunct project. Copied whole (2.2 GB) and reduced to 33 tracked files by a
  deny-by-default `.scratch/dl/.gitignore`. Two renames inside the copy were forced
  rather than chosen — `.git` → `.git-disabled`, since an embedded repository makes
  `git add` on interior paths a silent no-op that exits `0`; and `CLAUDE.md` →
  `CLAUDE.md.orig`, since a nested `CLAUDE.md` is loaded when reading files beside it and
  would have injected another project's agent instructions into sessions here.
- 2026-08-31 — Scratch location is `.scratch/` at the repo root, tracked, with one
  subdirectory per implementation. The leading dot is load-bearing: `uv run pytest` has no
  `testpaths`, so its walk reaches everywhere, and the dot matches pytest's default
  `norecursedirs` entry `.*` — no config to add now or remove at cleanup. Verified with a
  deliberately-failing canary inside `.scratch/` (not collected) against the same file at
  the repo root (collection error). Root `pyproject.toml` untouched.
- 2026-08-31 — Merge base analysed; `.scratch/dl/ANALYSIS.md`. The base fuses two different
  kinds of encoder into one map: a dense vocabulary index (`PAD`/`BOS`/`EOS` at 0/1/2) and a
  stateless structured code (`PitchCode`, `100*chromatic + diatonic`, range 1207-13176).
  **Decided:** `dataseq` owns the dense index per ADR 0011; structured codecs are demoted to a
  symbol canonicaliser running before vocabulary assignment. The decisive argument is `hmm`,
  whose V x V transition matrix would be ~1.74e8 entries at V = 13,177 for an alphabet of a few
  dozen real symbols. This is the substance of what ADR 0010 is `Proposed` pending, so goal 7
  now has its decision section in draft.
- 2026-08-31 — Lush implementation imported to `.scratch/hmm-lush/`; 135 tracked files
  (≈129 KB) out of 929 MB, again deny-by-default. Not version-controlled, so provenance
  is dates rather than a revision: sources span 2008–2011. Neither of the MelodyHPO
  renames applied — no `.git`, no nested `CLAUDE.md` — both checked before reading.
  The import already answers the question goals 3–4 were given: this encoder is a
  **dense vocabulary index**, built in first-appearance order, and its reserved block is
  `0 begin`, `1 end` with user symbols from **2**. That is a third offset, and the first
  one that collides with ADR 0011 destructively rather than merely differing: the loader
  packs ragged sequences into a dense `size × seq_size_max` integer matrix, so the unused
  tail of every short row is `0`, which here means `begin`. The original is safe only
  because `_seq_sizes` is consulted at every read. This is the ADR 0011 `PAD`=0 rationale
  observed in the wild rather than argued from first principles.
- 2026-08-31 — Precedence between the ADRs and the three imports stated explicitly, as a
  new invariant in `docs/agents/core.md`: the `dl` version is the merge's *base* because
  it is the most mature, but base means starting point, not authority. Where any import
  disagrees with an Accepted ADR the ADR wins, unless that ADR carves out an exception in
  its own text — as ADR 0011 does for the *spelling* of the strictness switch while
  holding its semantics settled. Prompted by re-reading goals 3–4, whose "every point
  where that base must be overridden" reads, on its own, as licence to let an import
  reopen a closed decision; it means the points the ADRs leave open. Sharpest instance:
  the three imports start user symbols at 3 (`dl`), 2 (Lush) and 4 (proof-of-concept),
  and none of them has a `GAP` code at all, so there was never a majority to defer to —
  ADR 0011 is supplying something all three lack rather than breaking a tie. This also
  reclassifies the dense-matrix finding logged above: it is an illustration of the
  failure ADR 0011 prevents, not evidence for `PAD`=0, and goal 3 now carries a note
  saying so before its semantic account is written.
- 2026-08-31 — Goal 3 subgoals 2 and 3 done. `.scratch/hmm-lush/ACCOUNT.md` describes the
  Lush original in its own terms, before any translation; `.scratch/hmm-lush/translation/`
  is a runnable, stdlib-only rendering of it. Three findings changed the shape of the merge
  question. **(1)** `format-sds` is a free `de`, not a method, so the vocabulary is a
  build-time artefact and a loaded container has no encoding path at all — which is why the
  strict-vs-`UNK` question never arises in the original. It is a structural absence, not a
  lenient choice, and that reframes what ADR 0011's strictness clause adds. **(2)** The dense
  `size × seq_size_max` matrix is a staging buffer, not the corpus representation:
  `hmm-trainer.lsh:66` is the only caller of `fprop-all` and nothing else reads `seq-data`,
  so `load` unpacks ragged data into a rectangle whose sole consumer immediately repacks it
  ragged. 71% of `set01z0`'s matrix is `begin`-valued padding bought for nothing — the
  hazard was taken on without even the performance argument that usually excuses it.
  **(3)** The `%l`/`%s` split between the two `_alphabet` writers is not an oversight but a
  downstream cost of the compile boundary: `alphabet` is `-idx1- (-gptr-)` because
  `fprop-all` and `set-alphabet` are what `dhc-make` compiles, and `dsource-seq.lsh:83`
  (`str-ptr (symbol->string s)`) is where the knowledge that a name ever needed delimiters
  is discarded. `format-sds`, outside the class, keeps symbols and can still use `%l`. Dates
  agree: 2009-07-05 precedes 2009-07-15. Corrected in the account after review — the tracked
  corpora are fine, `|…|` is multiple-escape and round-trips, and only the unmeasured half
  (that `%s` emits the string bare) was ever at issue; it is now flagged as inference.
  The translation's rebuild check re-derives both corpora from their own `_raw_data` and
  reproduces the 2009 output exactly, so first-appearance ordering, `begin`/`end` wrapping
  and the dense packing are confirmed rather than assumed. The goal-2 gitignore trap fired
  again — deny-by-default `/*` swallowed the whole `translation/` directory — and was caught
  before the commit this time, by running `git check-ignore` per file rather than assuming.
- 2026-08-31 — Goal 3 complete. `.scratch/hmm-lush/COMPARISON.md` sets the Lush
  implementation against the `dl` base, with §2 (overrides, on matters the ADRs leave open)
  and §3 (divergences, ADR-settled) kept in separate sections so the two cannot blur when
  this feeds ADR 0010. Five overrides: first-appearance ordering rather than set iteration;
  per-sequence true lengths as container state; vocabulary persistence with the format owning
  its quoting rule; frozen-as-explicit-state; decode on the tested surface. Two of those turn
  arguments already made in `ANALYSIS.md` into evidence — the ordering fix was proposed there
  on stability grounds and is now backed by the translation reproducing the 2009 `_alphabet`
  code for code, and `dependencies = []` was argued there and is now supported by a
  stdlib-only translation that handles both corpora.
  The finding neither analysis could have produced alone: **both implementations bake a
  consumer's view into the container** — `dl` next-token `(input, target)` pairs, Lush the
  flat concatenated stream Baum-Welch wants. Two unrelated consumers, the same category
  error, arrived at independently, which promotes `ANALYSIS.md` §2.2's separation from a
  judgement call about layering to a correction of a mistake made twice.
  Scoping honoured: `seq-state` is named as an `hmm` object, not a `dataseq` one. What
  `dataseq` owes it is a per-sequence view an annotation can align against, which is the
  concrete reason the true-lengths override is not merely tidiness.
  The branch plan also gained a `## Constraint on every comparison below` header block, and
  three subgoal criteria were reworded to carry the ADR-precedence rule inline. The notes
  alone were not enough: `/hitl-step` treats subgoals as the acceptance criteria and notes as
  context, so a constraint stated only in commentary does not bind the executor.
- 2026-08-31 — Third implementation imported and its tracking policy set, ahead of goal 4.
  The search surfaced **two** trees rather than one, both now under
  `.scratch/py-rudimentary/`: `segalign` (`ca97809`, 2025-09-05) and the predecessor it was
  refactored from, `SegAlign-Draft` (`9dc37b9`, 2025-06-30). They are counted as **one**
  implementation. Only `segalign` has anything resembling a `dataseq` — a `seq/` subpackage
  with `Dataset` and `Alignment` — and `SegAlign-Draft` earns its place by lacking one, both
  of its alignment entry points taking bare `List[Any]`. Since that is a negative claim, it is
  tracked at a single file, `glob/ss2_alignment.py`, whose signature makes it checkable after
  `.scratch/` is gone.
  `.scratch/py-rudimentary/.gitignore` admits 72 files (356 KB) of 1.7 GB. The bar is
  deliberately higher than `hmm-lush`'s, and the reason is provenance rather than importance:
  that import had no version control and was irreplaceable, whereas both of these are clean
  checkouts of live GitHub repositories at recorded revisions, so anything excluded costs one
  `git clone`. The largest thing turned away is `SegAlign-Draft/src/segalign/tcoffee/` — six
  modules of T-Coffee multiple alignment — which is `align`'s scope, and which tracking here
  would not have preserved for `align` anyway, since `.scratch/` is deleted by this branch's
  last goal.
  Both documented import renames were needed this time, where `hmm-lush` needed neither: two
  live `.git` directories and one nested `CLAUDE.md`. Revisions were captured *before*
  disabling, which is the only convenient moment. `segalign`'s working copy is dirty at
  `ca97809` in `glob/needleman_wunsch.py`, which is tracked regardless because
  `src/segalign/__init__.py` imports `glob` — without it the merge target does not import and
  its tests do not run. The 50 `All.csv` files (200 KB) are tracked for the same reason: the
  `seq/` tests hit the real corpus through `from_directories` rather than fixtures. Unlike the
  `.sds` directories, a partial copy would be safe here — nothing records an expected count.
  The three `.scratch/*/.gitkeep` placeholders were removed, having been made redundant by the
  content each directory now holds, along with the `!/.gitkeep` negation each import's
  `.gitignore` carried to punch them through its `/*` catch-all.

  **Two measurements, and one discrepancy that goal 4 must settle before either is acted on.**
  `segalign`'s `Dataset` allocates `{':EOS': 0, ':PAD': 1}` with user symbols from **2**, so
  `PAD` is not 0 — the one part of [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
  `core.md` calls non-negotiable, since PyTorch's zero-fill idioms would otherwise mean
  something other than "absent". Its `_encode_sequences` also falls back to the *`PAD`* index
  for unseen tokens, collapsing "unknown" and "absent" onto one integer. And neither tree has
  any integer gap index at all: both represent a gap as in-band Python `None`, rendered `"-"`
  at print time only. That is the third implementation with no `GAP`, in the one project of
  the three that exists to do alignment.
  The discrepancy: `docs/plan/DEFERRED.md` and the `core.md` invariant both state that "the
  proof-of-concept allocates user symbols from 4, with a different gap index". Neither tree
  matches that — the offset is 2 and there is no gap index. Either "the proof-of-concept" is a
  fourth codebase (`core.md` does separately say the proof-of-concept alignment/HMM code "has
  not been moved in"), or the recollection was wrong. **Both claims were left unedited**,
  because if `py-rudimentary` *is* the proof-of-concept then `core.md`'s "every one of the
  three uses a different offset" also fails — Lush and this one would both be 2 — and that is
  a tabulation for goal 4, not a docs-sync edit made in passing on an import commit.
- 2026-08-31 — The fourth implementation found, and `.scratch/` made permanent. Two changes,
  and the second is a reversal.
  **`tokalign` is the proof-of-concept, and it exists.** The previous entry recorded that
  `DEFERRED.md` and the `core.md` invariant described a proof-of-concept allocating user
  symbols from 4 with a distinct gap index, and that neither `py-rudimentary` tree matched.
  Verification widened the gap rather than closing it: `Alphabet`, `ScoringMatrix`,
  `AlignmentResult` and a Needleman-Wunsch `.pyx` -- all named in PRD §1.2 and ADR 0002 --
  were absent too, which put four ADRs and the PRD's opening section in doubt. The choice
  offered was to trim those claims or to hold; holding was chosen, and it was right. The
  library is `github.com/PanosMavromatis/tokalign`, now imported at `.scratch/align-poc/`,
  and every claim is exact: all three types are in `src/tokalign/_types.py`,
  `algorithms/needleman_wunsch/_cython.pyx` is present, and
  `Alphabet.RESERVED_INDICES = 3` gives padding/BOS/EOS at 0-2, gap at 3 and user symbols
  from 4 -- PRD §11 verbatim. **Nothing was trimmed.** The near-miss is recorded in
  `.scratch/README.md` because it generalises: a documentation claim that cannot be matched
  to code is not thereby false, and six documents were close to being rewritten to assert
  that a real library never existed.
  This also corrects the previous entry's framing. `Alphabet.encode` raises `KeyError` on an
  unseen symbol -- strict by default, which is what ADR 0011 mandates -- and it has a real
  `gap_index` and a working `decode`. Three of the four implementations are flawed on the
  reserved block; this one needs *renumbering* only, ADR 0011 inserting `UNK` and `MSK` to
  move user symbols from 4 to 6. That is unsurprising in hindsight: ADRs 0001-0004 were
  derived from this library, so it agrees with them by construction, and it is a *source* of
  the invariants rather than evidence to be weighed against them.
  **`.scratch/` is no longer deleted when this branch merges.** The same imports seed `hmm`
  and `align` 0.1.0, so the tree is retained and what is re-scoped per package is the
  `.gitignore` policies rather than the contents. `.scratch/align-poc/.gitignore` is the
  first written in explicit phases -- `dataseq` active (the `Alphabet` encoder, plus
  `_backends.py` and `conftest.py` because pytest loads that conftest for the whole `tests/`
  tree), `hmm` empty by design and recorded as a finding rather than an oversight, `align`
  written out but commented. Advancing a phase is an uncomment, not a re-derivation.
  The reversal contradicted a claim in seven places, all corrected. The urgent one was
  `docs/plan/feat-dataseq-merge/TODO.md`, whose last goal read `[ ] Delete the scratch
  location` and would have been executed as written by the next `/hitl-step` to reach it.
  Its retention reasoning is kept under an "if this directory is ever deleted" heading rather
  than discarded, since the squash-merge hazard returns intact if the decision is ever
  revisited. One consequence worth noting: `py-rudimentary`'s exclusion of `tcoffee/` was
  argued partly *from* deletion, and that premise is now void. The exclusion stands on better
  footing -- `tokalign` is the actual ancestor of `pfsmgraph-align` and supersedes
  SegAlign-Draft's alignment code as a migration source, so the `align` phase widens
  `align-poc`'s policy, not `py-rudimentary`'s.
  **Eight renames on import, the most of any so far**: three nested repositories and five
  agent-instruction files. Two are worth knowing before re-running it. `AGENTS.md` and
  `AGENTS.override.md` sat at tokalign's root because that project uses the same agent-docs
  toolchain as this one -- they are the *generated artefacts* of another project, and
  `protect-agent-docs.py` matches on filename, so under their own names this repository would
  have treated them as its own. And `dev/plugins/workflow-claude/` is mode 555, a vendored
  clone of the same plugin installed here; renaming a child needs write permission on the
  parent, so those two renames required `chmod u+w` first, with the mode restored afterwards.
  The failure presents as a bare `Permission denied` that reads like a sandbox restriction.
  A drag-and-drop slip on the owner's part turned out to be load-bearing. The import first
  landed at `.scratch/tokalign/` with its policy inside it, where the `!/*.md` negation --
  meant for our own analysis files -- also matched tokalign's `TODO.md` and `TODO.human.md`,
  silently tracking another project's planning documents as ours. Moving the tree into
  `align-poc/` and the policy one level above it, as `.scratch/dl/.gitignore` already governs
  `MelodyHPO/`, excludes them structurally rather than by a named exception.
- 2026-08-31 — Goal 4 reworded before running it, and a distinction recovered in the process.
  The goal read "Import the last implementation (rudimentary, in Python) and tabulate its
  divergence with the merged implementation so far", which was wrong on three counts once
  both remaining sources were in hand: two are left rather than one, both are already
  imported so the work is tabulation rather than import, and "rudimentary" fits `segalign`
  but not `tokalign`, which is the most developed of the four. It now reads "Tabulate the two
  remaining implementations against the merged base", with five subgoals — the first already
  `[x]`, since both imports and their provenance landed in the two preceding commits.
  Two new criteria were added rather than left to the written comparison: the reserved block
  tabulated across all four sources and ADR 0011, this being the first point at which the
  full picture exists; and the `Alphabet` reconciliation, which
  [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md)
  requires as *part of* this merge. The comparison criterion also asks for the two accounts
  to be kept separable, since `segalign` bears on the container and `tokalign` on the
  encoder, and one merged document would obscure which source supports which claim.
  **The distinction, which was mine to get wrong and the documents' to get right.** Chasing
  what looked like stale "three implementations" wording across the plan turned up that it is
  not stale at all: `tokalign` is *not* a fourth `dataseq` implementation but an alignment
  library contributing the encoder half. There are **four imported sources and three
  containers**. That is exactly why ADR 0010 is titled `merging-three-implementations` while
  its §49 separately requires reconciling `Alphabet`, and why the master plan's subgoal 51
  needed no change. The loose phrasing was in the previous commit's `core.md` edit ("the four
  existing implementations"), now corrected there and given as an explicit caveat in
  `codex.md` so a reviewer does not file "three" as an error.
  What did need widening is the **precedence rule** in this plan's constraint block, which
  named three implementations and now names every imported source including `tokalign` —
  with the warning that the rule reads backwards there. For the other three a divergence from
  an ADR is evidence of a flaw; for `tokalign`, ADRs 0001–0004 were written *from* it, so a
  divergence is almost always a later decision. `Alphabet` puts user symbols at 4 and ADR
  0011 moves them to 6: a renumbering, not a repair. Without that in the criteria, goal 4
  would file the most ADR-conformant of the four as the most deviant. Same failure mode as
  the ADR-precedence rule in goal 3 — `/hitl-step` binds on subgoals, not on notes.

### 2026-08-31 — goal 4 complete: the four-way picture, and two documents it falsified

Goal 4 executed and closed. Three documents: `.scratch/py-rudimentary/COMPARISON.md` (the
container half, `segalign` against the `dl` base), `.scratch/align-poc/COMPARISON.md` (the
encoder half, `tokalign`'s `Alphabet` against the merged encoder), and
`.scratch/RESERVED-BLOCK.md` (all four sources against ADR 0011). Two documents rather than
one because `segalign` bears on the container and `tokalign` on the encoder; merging the
accounts would have obscured which implementation supports which claim. Per the goal's second
Q&A, the `Alphabet` reconciliation is **tabulated and proposed, not decided** — goal 6 owns
the encoder API, goal 7 the ADR 0010 promotion.

**The reserved-block table falsified two standing claims**, in `docs/agents/core.md`'s
invariant and in `hmm-lush/COMPARISON.md` §3.1 that it derives from. Both read "every one of
the three uses a different offset — `dl` at 3, the Lush original at 2, the proof-of-concept at
4 — and none of them has a `GAP` code at all". The trio was miscounted: it named two
containers and the proof-of-concept, which is the *fourth source* rather than the third
container. The actual third container is `segalign`, which starts users at **2** and collides
with Lush — so the containers are 3, 2, 2, and three distinct offsets appear only across all
four sources. And `tokalign` **does** have a `GAP` code, at index 3, which is precisely why
its user symbols start at 4: the offset the claim cited and the gap it denied are the same
fact.

Neither was careless, and the distinction matters for how they were repaired. Both were
written before `segalign` and `tokalign` had been read, with the proof-of-concept row filled
in from `DEFERRED.md`'s recollection — which predicted the offset correctly and the gap
wrongly. `core.md` is live guidance, so it was **corrected**, with a dated parenthetical
naming `RESERVED-BLOCK.md` §2 as authoritative for that table. `hmm-lush/COMPARISON.md` is a
goal-3 artefact, so it was **superseded in place** by a dated note and its table left
standing: what it got wrong is the record of what was knowable in goal 3. Same append-only
discipline as the plan's `> **Done:**` notes.

Worth recording that the first draft of `RESERVED-BLOCK.md` tried to *defend* the `core.md`
sentence rather than correct it, and that the first correction to `core.md` introduced a fresh
false claim ("no two sources agree on where user symbols start", when Lush and `segalign`
agree on 2). Both were caught by re-reading the edit against the table. The measurement was
never in doubt; the prose about it was wrong twice.

**Two real defects in `tokalign`**, both surviving the "it is a source of the ADRs" caveat
because neither is a decision. `RESERVED_INDICES: int = 3` is annotated as a dataclass field
rather than a `ClassVar`, so the block ADR 0011 requires to be *fixed* is a positional
constructor argument — `Alphabet(("D3","F3"), ".", 7)` constructs and relocates everything,
and the two alphabets compare unequal while both stay hashable. And `decode` raises
`KeyError: 0` on any reserved code, so the zero-padded batch that `PAD`=0 exists to make
meaningful is the one array shape that cannot be decoded. Both were verified by running the
code rather than reading it; no test decodes a reserved index, which is how the second
survived. `docs/agents/codex.md`'s counting caveat was widened accordingly, since as written
it would have had a reviewer dismiss both.

Also corrected there: the tracked-file counts for `dl` and `hmm-lush` were each high by one,
and `align-poc` was missing from the list.

### 2026-08-31 — goal 5 complete: the container lands, and the repository gets its first tests

`packages/pfsmgraph-dataseq/` now holds six modules and 63 passing tests — the first code and
the first tests in this repository. `uv sync` is clean and all five members import together.

Three decisions were taken by Q&A before any code was written, and each changed the shape of
the result. **Ragged items plus a shipped `pad_collate`**, rather than pre-padded fixed-width
items: the subgoal's "a stock `DataLoader` works without subclassing" is satisfiable by storing
padding in the container, which is what the `dl` base does, and that would have inherited
`ANALYSIS.md` §3.6's unmasked-padding gap instead of closing it. **numpy int32 arrays**, which
settles `DEFERRED.md`'s "numpy may earn its way in" in the affirmative while leaving its "not
torch, not pandas" standing. And **a `Vocabulary` protocol** rather than a concrete encoder, so
goal 6 can settle the encoder API without touching the container.

**The `DataLoader` criterion turned out to have two readings that disagree.** A stock
`DataLoader` does work, verified by running it rather than asserting it. But
`isinstance(ds, torch.utils.data.Dataset)` is `False`, because that class is a plain class rather
than a protocol or ABC — satisfying `isinstance` would mean inheriting from it, and so importing
torch into the base layer. `DataLoader` never makes that check for map-style datasets. Recorded
as a passing test asserting the `False`, so a later reader meets it as a decision rather than a
defect. Also verified: without a `collate_fn`, `default_collate` raises `TypeError` on ragged
items, which is the concrete case for shipping `pad_collate` rather than leaving batching to
every caller.

Goal 4's two `tokalign` defects became tests rather than inheritances. `decode` is total over
the whole code range, so `vocabulary.decode(batch["codes"][2])` returns `['D3', 'PAD', 'PAD']`
where the proof-of-concept raised `KeyError: 0` — and the reserved block is module constants
with no class or parameter to relocate it, asserted by a test that checks the module defines
nothing callable at all.

Two tests failed first and both were the tests' own fault, worth recording because both were
testing something other than what they named. One asserted the reserved-block module had no
callable surface, but was reading `typing.Final` off its imports. The other asserted
`"torch" not in sys.modules` in a process where a sibling test had imported torch — a claim
that means nothing except in a fresh interpreter, so it now runs as a subprocess check.

`DEFERRED.md`: the build backend (hatchling holds — no source has a compiled inner loop
belonging to `dataseq`; `tokalign`'s Cython is `align`'s migration) and the runtime dependencies
(`numpy` only) are both closed. The ADR 0003 test-suite entry is **not** closed and was
re-keyed: `dataseq` has no DP algorithm and therefore no backends, so the `pytest_report_header`
backend matrix has nothing to report and its real trigger is the first `.pyx`.
