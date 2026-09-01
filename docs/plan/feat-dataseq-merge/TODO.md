# feat/dataseq-merge

**Status**: active
**Created**: 2026-08-31
**Subgoal**: revision `01-dataseq-v0.1.0`, subgoals 1 (merge the three implementations)
and 2 (settle the encoder API and promote ADR 0010)

## Constraint on every comparison below

The ADRs outrank every imported source — the three `dataseq` implementations, the `dl` base
included, **and `align-poc/tokalign`**. "Base" means starting point, not authority: where any
of them disagrees with an Accepted ADR, the ADR wins and the implementation changes — unless that ADR carves out an exception in its own text, as
[ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) does for
the *spelling* of the strictness switch while holding its semantics settled. So wherever a
goal below says the base "must be overridden", it means the points the ADRs leave **open**,
never the ones they close. Recorded as an invariant in `docs/agents/core.md`.

`tokalign` is the case to watch, because the rule reads backwards there. The other three are
evidence weighed against the ADRs; `tokalign` is what ADRs 0001–0004 were written **from**,
so it agrees with them by construction and its divergences are overwhelmingly later
decisions rather than defects. The ADR still wins — `Alphabet` puts user symbols at 4 and
[ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) moves
them to 6 — but that is a renumbering, not a repair, and it is the only one of the four that
already satisfies 0011 on strictness, `gap_index` and `decode`. Note also that it is not a
fourth `dataseq` implementation: there are three, and `tokalign` contributes the *encoder*
half, which is why [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md)
names reconciling its `Alphabet` as part of this merge while its own title still says three.

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

- [x] Import the earlier `hmm` implementation (in Lush) and tabulate its divergence with the previous implementation
  > **Done:** import (`aee7a3d`), `ACCOUNT.md`, `translation/`, `COMPARISON.md`. The goal was
  > set to answer whether this encoder is a vocabulary or a codec and where its reserved block
  > starts; both are answered — a dense vocabulary index in first-appearance order, reserved
  > `0 begin` / `1 end`, user symbols from 2 — and the first answer independently corroborates
  > the dense-index decision taken during goal 2, arrived at fifteen years earlier under no
  > influence from it. The `dl` base is the only one of the three that fuses a structured codec
  > into the vocabulary.
  - [x] The second implementation is readable in the scratch location and its provenance is recorded
    > **Done:** commit `aee7a3d`. 135 files (~129 KB) out of 929 MB, deny-by-default;
    > provenance recorded in `.scratch/README.md` as dates rather than a revision (sources
    > span 2008-2011, tree reorganised 2022-08-26), the project never having been under
    > version control.
  - [x] A semantic account of the Lush original **in its own terms** — container model, encoder shape, vocabulary handling — written before any translation, so the comparison is against the original rather than against our reading of it
    > **Done:** `.scratch/hmm-lush/ACCOUNT.md` (337 lines), written from the three
    > `Code/SeqData/*.lsh` sources, the `util.lsh` helpers they call, and both tracked
    > `.sds` specimens; every quantitative claim measured rather than inferred. Load-bearing
    > findings: (1) `format-sds` is a free `de`, not a method, so **the vocabulary is a
    > build-time artefact and a loaded container has no encoding path at all** — which is why
    > the strict-vs-`UNK` question never arises in the original, rather than arising and being
    > answered leniently. (2) The dense `size x seq_size_max` matrix is a *staging buffer*, not
    > the corpus representation: `hmm-trainer.lsh:66` is the only caller of `fprop-all`,
    > nothing else reads `seq-data`, so `load` unpacks ragged data into a rectangle whose sole
    > consumer immediately repacks it ragged — 71% of `set01z0`'s matrix is `begin`-valued
    > padding bought for nothing. (3) the two writers spell `_alphabet` differently —
    > `format-sds` prints the symbol with `%l`, `save` prints `(ptr-str …)` with `%s`. The
    > tracked corpora are **fine**: `%l` emits the multiple-escape delimiters `|…|`, and the
    > reader takes everything between them as one symbol name whatever it contains, so
    > `set11a_dInt`'s two whitespace-bearing symbols round-trip correctly. The open question
    > concerns only `save`-written output, which exists nowhere in the tree, and its final
    > step is **inference, not measurement** — Lush cannot be run here to confirm `%s` emits
    > the string bare. Carried forward as a regression test for the merged package, not as a
    > defect of the original. (4) `seq-state`'s parallel `size` / `size+1` arrays are
    > the design idea most worth carrying forward, and are an annotation idea rather than a
    > container one.
  - [x] Draft a close translation to Python alongside the original in the scratch location, noting each choice the account left open where Lush does not map mechanically
    > **Done:** `.scratch/hmm-lush/translation/` — six stdlib-only modules
    > (`lush_reader`, `seq_state`, `dsource_seq`, `format_sds`, `__init__`, `__main__`).
    > Idiomatic rather than literal per the Q&A above; every divergence carries a
    > `DEVIATION` comment giving the original's behaviour and the reason for departing.
    > `python3 -m translation` loads both tracked `.sds` corpora and reproduces ACCOUNT.md's
    > appendix from measurement. The load-bearing check is the **rebuild**: re-deriving each
    > corpus from its own `_raw_data` reproduces the 2009 output exactly — alphabet (so
    > first-appearance ordering is confirmed, not assumed), `_size`, `_seq_size_max`,
    > `_seq_sizes`, and the full dense matrix including its padding. A save/load round-trip
    > also carries `set11a_dInt`'s two whitespace-bearing symbols intact, demonstrating that
    > one writer owning the quoting rule closes what the original's `%l`/`%s` split left open.
    > Four modules where the plan anticipated two: the `_raw_data` tokenizer is isolated
    > because it is the one place faithfulness is not optional (the corpora are its
    > specification), and `seq_state` follows its own Lush file.
    > **Note:** the goal-2 gitignore trap fired again and was caught before committing, not
    > after — deny-by-default `/*` swallowed the whole `translation/` directory, so six
    > verified-working files were invisible to `git status`. Fixed with an anchored
    > `!/translation/`, `/translation/*`, `!/translation/*.py` triple, which also keeps
    > `__pycache__/` out for free, since children of an excluded directory cannot be
    > re-included. Verified with `git check-ignore` per file rather than assumed.
  - [x] A written comparison against the `dl` base, and every point where that base must be overridden by this implementation — **among the points the ADRs leave open** — with why; a point an Accepted ADR settles is recorded as a divergence of the original, never as a candidate to adopt
    > **Done:** `.scratch/hmm-lush/COMPARISON.md` (281 lines), with §2 (overrides, ADR-open)
    > and §3 (divergences, ADR-settled) deliberately in separate sections so the two cannot be
    > confused when this feeds ADR 0010. **Five overrides**, all on matters no Accepted ADR
    > settles: (2.1) first-appearance ordering, not set iteration — the `dl` base's
    > `symbol_set - self.alphabet` iteration is a reproducibility bug under a dense index, and
    > the translation's rebuild against 2009 output is a sixteen-year determinism test we could
    > not have written; (2.2) per-sequence true lengths as container state — both pad to a
    > global max, but `dl` *discards* the lengths so a mask is not even derivable, where Lush's
    > `seq_sizes` is consulted by every reader; (2.3) vocabulary persistence, with the format
    > owning its own quoting rule, since a name unwrapped to a plain string has lost the
    > knowledge that it needed quoting; (2.4) frozen as an explicit state — `dl` has an encoder
    > that cannot freeze, Lush a freeze achieved by having no encoder, so neither is a model to
    > copy; (2.5) decode on the tested surface, `dl`'s `decoder_map` being built, kept in sync,
    > and never read.
    > **Finding neither analysis could have produced alone (§4):** both implementations bake a
    > consumer's view into the container — `dl` next-token `(input, target)` pairs, Lush the
    > flat concatenated stream Baum-Welch wants. Two unrelated consumers, the same category
    > error, independently. That promotes `ANALYSIS.md` §2.2's separation from a judgement call
    > to a correction of a mistake made twice.
    > **Scoping honoured (§5.1):** `seq-state` is identified as an `hmm` object, not a
    > `dataseq` one. What `dataseq` owes it is a per-sequence view an annotation can align
    > against, which is the concrete reason §2.2 is not merely tidiness.
  > **Q:** Goal 3 has three deliverables — account, translation, comparison. How should they
  > be laid out in `.scratch/hmm-lush/`?
  > **A:** Three artefacts: `ACCOUNT.md`, `translation/` (`dsource_seq.py`, `format_sds.py`),
  > `COMPARISON.md`. Splitting the account from the comparison keeps subgoal 2's "written
  > before any translation" ordering legible in the diff, and the two documents have
  > different audiences — the account describes the original, the comparison serves the merge.
  > **Q:** Was `dsource-seq save` ever actually used, or did every `.sds` come from
  > `format-sds`? Decides whether the `%s`/`%l` alphabet asymmetry is a live defect or dead code.
  > **A:** Don't recall — sixteen years. The account states what the code does and declines to
  > claim what was run; the finding is flagged "provenance unknown".
  > **Q:** How faithful should the Python translation be?
  > **A:** Runnable, and it reads the real `.sds` specimens. The two specimens were tracked for
  > exactly this, so the account's claims about padding, empty sequences and alphabet parsing
  > are demonstrated rather than asserted.
  > **Q:** How literal should the translation be, and how much does the `hmm` side matter?
  > **A:** Not literal. Idiomatic Python that reproduces the *implementation decisions*
  > rather than the Lush constructs — a transliteration would preserve accidents of the
  > language and obscure the choices. The `hmm` branch matters only insofar as it shows how
  > the `hmm` model implementation will have to be modified to suit the new decisions, so
  > the translation's job is to expose the seams `hmm` depends on (the flat concatenated
  > stream from `fprop-all`, `seq-state`'s parallel `size`/`size+1` arrays), not to model
  > the trainer.
  > **Note (cause of the `%l`/`%s` asymmetry, resolved after the account was first written):**
  > not interpreted-vs-compiled code — neither writer is compiled — but the representation
  > compilation *forced*. `alphabet` is `-idx1- (-gptr-)` because `fprop-all` and
  > `set-alphabet` are what `dhc-make` compiles and the alphabet must cross that boundary;
  > that slot type routes every class-side access through `str-ptr`/`ptr-str`, and
  > `dsource-seq.lsh:83` is where `symbol->string` discards the knowledge that a name ever
  > needed delimiters. `format-sds` sits outside the class, never touches a `gptr`, and keeps
  > symbols. Dates agree: `format-sds.lsh` 2009-07-05 precedes `dsource-seq.lsh` 2009-07-15.
  > The asymmetry is a downstream cost of the compile boundary, which is why it is worth
  > carrying into the merge — the same boundary exists there for the same reason.
  > **Note (framing, settled before the account is written):** the ADRs outrank all three
  > imported implementations — see `docs/agents/core.md` under "Invariants". So the fourth
  > subgoal's "every point where that base must be overridden" means the points the ADRs
  > leave open, never the ones they close. In particular the reserved block is not among
  > them: [ADR 0011](../../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
  > fixes `PAD`=0 … `MSK`=5 with user symbols from 6, and Lush's `0 begin`, `1 end`, users
  > from 2 is a divergence to record, not a candidate to adopt.
  > **Note (for the semantic account):** `dsource-seq.lsh` packs ragged sequences into a
  > dense `size x seq_size_max` integer matrix, so the unused tail of every short row is
  > `0` — which in that alphabet means `begin`, a real symbol. The original is correct
  > only because `_seq_sizes` is consulted at every read; the padding is indistinguishable
  > from data by inspection. Write this into the account as an **illustration of the
  > failure ADR 0011 prevents**, not as evidence for `PAD`=0. The ADR is Accepted and does
  > not need corroborating from an import, and treating an implementation as support for a
  > decision already made is the same category error as treating it as authority against
  > one. It earns its place because it is that hazard observed in working code rather than
  > argued from first principles — the strongest available answer to "is the zero-fill
  > argument really load-bearing, or just tidy?"

- [x] Tabulate the two remaining implementations against the merged base
  > **Reworded (2026-08-31).** This goal read "Import the last implementation (rudimentary,
  > in Python) and tabulate its divergence with the merged implementation so far", which is
  > wrong on three counts now that both are in hand. There are **two** implementations left,
  > not one; both are already imported, so the work here is tabulation rather than import;
  > and "rudimentary" fits only one of them. `.scratch/align-poc/tokalign` is the *most*
  > developed of the four — it is the proof-of-concept alignment library PRD §1.2 describes
  > and [ADRs 0001–0004](../../design/adr/README.md) derive from.
  > **That asymmetry changes how it is read.** The other three are evidence about what has
  > been tried, weighed against the ADRs. This one is a *source* of the ADRs, so it agrees
  > with them by construction, and a divergence is far more likely to be a deliberate later
  > decision than a defect. The ADRs are still the later word where the two disagree — the
  > constraint at the top of this file is unchanged — but "tokalign does X and the merge does
  > not" needs checking against the record that was written *from* it before it counts as a
  > finding.
  > **Q:** Where should this goal's writing land, given that goals 2-3 set a one-document-per-import
  > precedent but the reserved-block table spans all four sources?
  > **A:** Two per-import documents plus a shared table: `.scratch/py-rudimentary/COMPARISON.md`
  > (container), `.scratch/align-poc/COMPARISON.md` (encoder), and `.scratch/RESERVED-BLOCK.md`
  > for the four-way tabulation against ADR 0011. The table belongs to no single import, so
  > burying it under one would misfile it.
  > **Q:** Subgoal 4 requires `Alphabet` "reconciled with the merged encoder" as part of this
  > merge, but goal 6 is "Settle and implement the encoder API". How far does this goal go?
  > **A:** Tabulate and propose, do not decide. Name every point where the two express the same
  > mapping differently and recommend a shape with reasoning; goal 6 owns the decision and the
  > implementation, goal 7 the ADR 0010 promotion. Goal 4 stays an analysis goal, as its title says.
  - [x] Both remaining implementations are readable in the scratch location with their provenance recorded
    > **Done (2026-08-31):** `.scratch/py-rudimentary/` holds `segalign` (`ca97809`) and its
    > predecessor `SegAlign-Draft` (`9dc37b9`); `.scratch/align-poc/` holds `tokalign`
    > (`6d27936`, branch `feat/docker-vertex-ai`) with two nested repositories of its own.
    > All revisions are in `.scratch/README.md`, captured before each `.git` was disabled.
    > Note `segalign`'s working copy is dirty at `ca97809` in `glob/needleman_wunsch.py`.
  - [x] A written comparison of container semantics, encoder shape and vocabulary handling for **both**, kept separable — `segalign` bears on the container, `tokalign` on the encoder, and merging the two accounts would obscure which implementation supports which claim
    > **Done (2026-08-31):** `.scratch/py-rudimentary/COMPARISON.md` (container, 184 lines) and
    > `.scratch/align-poc/COMPARISON.md` (encoder, 172 lines), both following the section shape
    > `hmm-lush/COMPARISON.md` established. Central container finding: `Dataset` has `__len__`
    > and **no `__getitem__`**, so it is not `Dataset`-compatible even under the duck-typed bar
    > `ANALYSIS.md` §2.1 sets — it is a corpus loader with whole-collection derived views, a
    > stage earlier than the other two. Central encoder finding: `Alphabet` already answers four
    > gaps the containers leave open (it *is* a vocabulary object, frozen by construction,
    > decodes, and is strict), so the reconciliation is a merge into the merged encoder rather
    > than a rewrite of it.
  - [x] The reserved block tabulated across all four implementations and ADR 0011, since this is the first point at which the full picture exists: `dl` from 3, Lush from 2, `segalign` from 2 with `PAD` at **1**, `tokalign` from 4 with a real gap index at 3 — and only `tokalign` strict by default with a working `decode`
    > **Done (2026-08-31):** `.scratch/RESERVED-BLOCK.md`. Every value in the subgoal's own
    > prediction verified against source and cited by line. Two findings the prediction did not
    > contain. **Four sources, three offsets, because two collide:** Lush and `segalign` both
    > start users at 2, but `0` means `begin` in one and `:EOS` in the other, so the same two
    > integers carry different meanings — a collision, not a fourth offset, which leaves
    > `core.md`'s "every one of the three uses a different offset" exactly true. And **`segalign`
    > supplies the sharpest evidence for `GAP` by deleting it**: its corpus uses `'.'` as a
    > no-pitch sentinel, stripped from both the sequences and the vocabulary, while `tokalign`'s
    > `gap_symbol` defaults to the *same character* with a reserved index. One implementation
    > destroys what the other reserves a code for.
    > Also recorded: `segalign`'s unknown → `:PAD` collapse is **deliberate and pinned by a
    > test**, unlike `dl`'s silent `NaN` — the only one of the four that chose the failure ADR
    > 0011's separate `UNK` prevents. And no source persists a vocabulary, so the renumbering has
    > no migration path to write, which is why it can land inside the merge.
  - [x] `tokalign`'s `Alphabet` reconciled with the merged encoder, which [ADR 0010](../../design/adr/0010-dataseq-composition-merging-three-implementations.md) requires as **part of** this merge rather than a later question, since the two express the same symbol ↔ code mapping
    > **Done (2026-08-31):** tabulated and proposed, not decided — per this goal's second Q&A,
    > goal 6 owns the API. `align-poc/COMPARISON.md` §5 is the six-point proposed reconciliation.
    > §2 confirms the ADR 0011 move is a **renumbering, not a repair**, with one substantive
    > addition: `Alphabet` has no `UNK` slot at all, so its strictness is *total*, and ADR 0011's
    > strict-by-default-with-opt-in is strictly more than it offers.
    > **Two genuine defects found, both surviving the "source of the ADRs" caveat** because
    > neither is a decision. `RESERVED_INDICES: int = 3` is annotated as a field rather than a
    > `ClassVar`, so the "fixed" block is a positional constructor argument —
    > `Alphabet(("D3","F3"), ".", 7)` constructs and relocates everything, and the two alphabets
    > compare unequal while both remaining hashable. Verified by running it, not by reading it.
    > And `decode` raises `KeyError: 0` on any reserved code, because `_idx_to_sym` is built from
    > the gap index up — so the zero-padded batch that `PAD`=0 exists to make meaningful is the
    > one array shape that cannot be decoded. No test decodes a reserved index, which is how it
    > survived.
  - [x] Every point where the `dl`/`hmm` merged base must be overridden by either implementation is named, with why — **among the points the ADRs leave open**, on the same terms as goal 3
    > **Done (2026-08-31):** `py-rudimentary/COMPARISON.md` §2 (four points) and
    > `align-poc/COMPARISON.md` §3 (two defects) and §4 (two contributions), each with a
    > `## What this hands forward` table routing it to goal 5, 6, 7 or the later `align`
    > migration. The two strongest: **the loader must not be a constructor** — `segalign`'s
    > `from_directories` and `dl`'s `MiniCorpus` independently fused corpus layout, file format,
    > vocabulary construction and container construction, which is the second time two
    > implementations made the same structural error separately; and **the private mapping is
    > already public API** — `ScoringMatrix` reaches into `alphabet._sym_to_idx`, which inside
    > the family becomes one distribution reaching into another, so goal 6 must publish that
    > accessor deliberately rather than inherit a private name.
    > Divergences explicitly recorded as **not** candidates: the reserved block and strictness
    > (settled by ADR 0011), the parallel `toks`/`toks_enc` attributes, and the documented-but-
    > absent decode direction.

- [ ] Land the merged container in `packages/pfsmgraph-dataseq/`
  - [ ] `dl` version is the base; divergences resolved per the comparisons above, and **per the ADRs wherever the two disagree**
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
  - [ ] Narrow `.scratch/*/.gitignore` to what the *next* migration needs, rather than deleting anything — `.scratch/` now survives this branch (see the note below), so cleanup here means re-scoping the four policies, not removing the tree
  - [ ] `.scratch/align-poc/.gitignore` advanced from its Phase 1 (`dataseq`) block to Phase 2/3 as appropriate, and the equivalent judgement recorded for the other three imports
  > **Superseded (2026-08-31):** this goal originally read "decide how the scratch code is
  > retained **before** deleting it" and "delete the scratch location", on the assumption
  > that `.scratch/` existed only for the `dataseq` merge. It does not: the imports are now
  > the migration source for `hmm` and `align` 0.1.0 as well, so the tree stays and the
  > per-package `.gitignore` policies are re-scoped as each migration begins. The retention
  > problem the old wording guarded against — a squash merge collapsing the add and the
  > delete into nothing, losing the code from `main` — **no longer arises**, because nothing
  > is deleted. It returns only if `.scratch/` is ever removed for real, at which point the
  > original reasoning still applies: retention needs a merge commit, a tag on the
  > pre-deletion SHA, or a branch left unmerged.
