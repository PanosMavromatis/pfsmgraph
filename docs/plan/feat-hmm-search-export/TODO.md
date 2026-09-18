# feat/hmm-search-export

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Decide whether revision 04 **exports** the topology search, and record the
decision either way (`docs/plan/TODO.md`, revision 04-hmm-v0.3.0)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked ·
`[-]` deferred/descoped

## Goals

- [x] Assemble the case each way, from the code rather than from the plan's summary of it
  > **Done:** both cases are below, read from the code. The case against exporting is
  > stronger than this branch's doc expected, and one of the two documents said to lean
  > the other way had already been turned before the branch opened.
  - [x] Read `_search`, `_trials`, `_topology` and `_mdl` as a consumer would: what would
        a public call have to promise, and which of those promises does ADR 0023 already
        record as unresolved?
    > **Note:** exporting the search is not one name but **five**. `SearchResult` holds
    > `TrialResult` twice and a `tuple[Round, ...]`, and `Round.move` is a `Move` whose
    > `result` is a `TrialResult` — so `search`, `SearchResult`, `Round`, `Move` and
    > `TrialResult` all become contract together, the whole dataclass graph being reachable
    > from the return value. The promises are the signature (`records`, `start`, `seed`,
    > `max_rounds`, `patience` and eight keywords, four of them tuning constants), ADR 0023
    > §2's `spawn_key` seeding contract, the stop set
    > `Literal["max_rounds", "patience", "dead_end"]`, and `best` as the strict-`<` minimum
    > over the walk with `start` included.
    > **Note:** three of those promises sit on ADR 0023's own Open section. Near the
    > optimum `best` is decided by where EM stopped rather than by the criterion — one
    > `m008` candidate's total moved up to 72 bits across EM floors of 200, 400 and 800
    > while its unrounded data length moved by at most 2.4 — so `split_min_cycles=200` is a
    > default that changes *which model comes back*, and the record says it "was not
    > measured on other corpora" and that the floor needed depends on the model (60 cycles
    > on `set02a`'s one- and five-state models, 200 on its eight-state one). §5's `d`-scan
    > assumption is measured on trained models rather than on search candidates. A held-out
    > score is explicitly outside the record, and adding one later is a new field or a new
    > stop value.
  - [x] Establish what is actually frozen by exporting — the signature, the return value,
        the seeding contract, the stop rules — and what stays free.
    > **Note:** the sharpest thing exporting would freeze is the **two-part code**.
    > `TrialResult` carries `d` and `data_bits`, and `_total_description_length` returns a
    > bare `float` precisely so that structure is not asserted: "a one-part code has no
    > data/model split, so a return type carrying those two fields would assert the very
    > structure that may be replaced" (`_mdl.py:276`, for PRD §8). Publishing `TrialResult`
    > re-asserts in a frozen `__all__` exactly what the criterion's seam was built to leave
    > unasserted. `baum_welch` never faced this, which is why revision 03 could export its
    > result on terms revision 04 cannot: `BaumWelchResult.description_lengths` is a corpus
    > length, and a corpus length survives a one-part code intact.
    > **Note:** what stays free either way — `backend=` and `score_backend=` are already
    > public vocabulary under ADR 0021, and `batch_size` is `baum_welch`'s. What does *not*
    > stay free is the log's column layout: under ADR 0013 a documented block is executed
    > and its output pasted, so documenting the search documents the table.
    > **Note:** the two choices are not symmetric. `__all__` may **widen** at 0.4.0 at no
    > cost to any consumer and may never narrow, so not exporting is the reversible option
    > and exporting is the one-way door. That asymmetry is the shape of the whole subgoal.
  - [x] Check what revision 05 and the PRD's §11 order need from this surface, since
        `align` re-declares against `hmm` and the alignment-seeded search is next.
    > **Note:** there is no revision 05. `docs/plan/TODO.md`'s `## Planned revisions` is
    > empty, and the next `hmm` work is the alignment-seeded search, given a *trigger*
    > rather than a number (`DEFERRED.md`, `## Trigger: align able to produce a multiple
    > alignment`) and expected at 0.4.0.
    > **Note:** it needs no new entry point — a seed FSM after epsilon removal is an
    > `HMMParams`, which `start:` already accepts by the `hmm-train-load-nw` resume path,
    > so a signature frozen at 0.3.0 would still serve 0.4.0. But that same entry leaves
    > open a question about the *public* shape: "confirm ADR 0019's expected answer against
    > how the seed is exposed: a caller who can still start merge/split from a single state
    > may never need `align`". Exporting now answers it by accident.
    > **Note:** one of the two documents said to lean toward exporting had already been
    > turned. `docs/api/hmm/baum_welch.md` no longer dangles: PR #50 added the paragraph
    > saying the search "is **private**: nothing under `pfsmgraph.hmm` exports it, and the
    > public surface is still exactly ten names". Only the audit subgoal's "whatever the
    > search loop and training log export" is still standing, so this plan's goal-3 "if
    > not" arm is one phrase in the master plan rather than two documents.

- [x] Decide, and record the decision where the next revision will find it
  > **Done:** the search is **not** exported at 0.3.0; `__all__` stays at ten names. The
  > decision rests on the reversibility asymmetry, not on a judgement that the surface is
  > wrong, and it is recorded in ADR 0025 with a `DEFERRED.md` trigger so it is re-made
  > rather than inherited. The strongest alternative — a narrowed result withholding `d`
  > and `data_bits` — is rejected on timing and written into the record's Open section and
  > the trigger, so the next reader finds the argument instead of rebuilding it.
  > **Q:** Does revision 04 make the topology search part of `pfsmgraph.hmm`'s public API
  > at 0.3.0 — no (keep `__all__` at ten names), yes (export the five-name dataclass graph
  > as it stands), or a narrowed surface designed to withhold `d` and `data_bits`?
  > **A:** No — keep it private. `__all__` stays at the ten names it has carried since
  > `viterbi_batch`, and the question is revisited at 0.4.0 or when PRD §8 settles. The
  > reversibility asymmetry decided it: `__all__` may widen later at no cost to any
  > consumer and may never narrow. The accepted cost is that 0.3.0 ships a headline feature
  > no pip user can call, and the release notes have to say so.
  - [x] Choose the venue: an ADR, or a section in an existing document. An ADR only if
  > **Q:** Where does the decision get recorded, so revision 04's audit and 0.4.0 both find
  > it — a new ADR, a new ADR plus a `DEFERRED.md` trigger, a `DEFERRED.md` entry alone, or
  > a section amending ADR 0023?
  > **A:** A new ADR plus a `DEFERRED.md` trigger, and a row in `adr/README.md`. The ADR
  > holds the argument, since the subgoal's own test is met — three alternatives were live
  > and the narrowed export is a design someone will re-propose. The `DEFERRED.md` entry
  > holds only what fires it, which is the mechanism this repository already has for
  > decided-but-not-yet-actionable work indexed by its trigger. ADR 0023 was not amended:
  > its Open section is part of what this decision rests on, so a surface decision inside
  > it would make that record argue with itself.
        the decision has alternatives worth arguing rather than a single reason.
  - [x] Write the decision with its reason, in the terms the plan asked for.
    > **Note:** written as [ADR 0025](../../design/adr/0025-topology-search-not-exported.md),
    > 151 lines, with the index's four places updated (the table row, a "reads after"
    > bullet, the no-PRD-counterpart list and the closing sentence) and a `DEFERRED.md`
    > trigger keyed on PRD §8 settling or `hmm` 0.4.0, placed beside the alignment
    > entry it cross-references. Two sections of the record are worth knowing before
    > reopening it: §6 says private means out of contract rather than unreachable, since
    > `pfsmgraph.hmm._search` imports and runs and what is withheld is the promise, not the
    > capability; and §7 names the cost instead of arguing it away, which obliges 0.3.0's
    > release notes to say the headline feature has no supported entry point.

- [x] Carry out whatever the decision requires
  > **Done:** the export arm is `[-]`, since ADR 0025 decided against it. What the "if
  > not" arm actually required was two edits, not the one it named: the master plan's
  > audit subgoal no longer presumes a surface, and `docs/api/hmm/README.md`'s
  > public-surface paragraph now lists all ten private modules and says why the last four
  > are private, citing ADR 0025. The API-docs guard passes at 11.
  - [-] If export: names, `__all__`, `docs/api/hmm/search.md` under ADR 0013 with every
        block executed and its output pasted, `docs/api/hmm/README.md`'s public-surface
        table, `packages/pfsmgraph-hmm/README.md`.
    > **Descoped:** [ADR 0025](../../design/adr/0025-topology-search-not-exported.md)
    > decided not to export, so none of this is owed at 0.3.0. The page, the
    > public-surface table row and the PyPI long description all become due together if
    > the `DEFERRED.md` trigger fires, and that entry names them so they are not
    > re-derived from scratch.
  - [x] If not: close `docs/api/hmm/baum_welch.md:160`'s pointer and the audit subgoal's
    > **Note:** the two things this subgoal named cost one edit between them.
    > `docs/api/hmm/baum_welch.md`'s pointer was already closed by PR #50 before the
    > branch opened, so only the master plan's audit subgoal needed repairing: "whatever
    > the search loop and training log export" now says they export nothing, and cites
    > ADR 0025.
    > **Note:** the code turned up the mirror-image defect, which this subgoal did not
    > predict and the audit would have found. `docs/api/hmm/README.md`'s public-surface
    > paragraph enumerated six private modules and **omitted all four revision 04 added**
    > — `_search`, `_trials`, `_topology` and `_mdl` — so the page defining what is
    > private was silent about the four things this branch decided are private. The list
    > was written in the 0.2.0 era and nothing makes it grow when a module lands. Both
    > are drift in one sentence; only the false-positive direction was anticipated.
    > **Note:** the asymmetry that let it drift is worth carrying to any ADR 0013 page.
    > `tests/test_api_docs.py` executes the code blocks against pasted output, so the
    > ten-name `__all__` listing beneath that paragraph *cannot* drift — but the prose
    > around it is unexecutable and drifts freely. The blocks are verified; the sentences
    > are not.
    > **Ran:** `uv run pytest tests/test_api_docs.py -q` — 11 passed, after the edit.
        "whatever the search loop and training log export", so neither is left asserting
        a surface that does not exist.

- [x] Sync the docs and leave the audit a smaller job than it was
  > **Done:** `core.md` and `AGENTS.md` carry the decision, and the master plan's audit
  > subgoal now says both what it no longer needs to ask and what this branch already
  > repaired. Nothing was pre-empted from `/smart-merge`: the export subgoal at
  > `docs/plan/TODO.md:225` stays open for it to close with the PR number.
  - [x] `docs/agents/core.md` and `AGENTS.md`, including test counts if any moved.
    > **Note:** `core.md` gains the decision in two places, for two readers: the `hmm`
    > section's export sentence, which now says the ten names are ten *by decision* and
    > names the four modules that stay out of contract, and the ADR paragraph, which
    > describes 0025 as it already describes 0015, 0017, 0020, 0022, 0023 and 0024. The
    > ADR count was repaired earlier on this branch, from twenty-four to twenty-five.
    > `AGENTS.md` rebuilt to 906 lines from 897, the nine being the wrapped insert;
    > `AGENTS.override.md` rebuilt byte-identical, since `codex.md` never moved.
    > **Note:** **no test count moved, and that is established by the diff rather than by
    > a run.** `git diff main...HEAD --stat` shows nine files, every one under `docs/` or
    > the generated `AGENTS.md` — nothing under `packages/` or `tests/` — so the suite
    > cannot have changed and stands at 3099. Running it would have confirmed a number the
    > diff already proves. The one guard that *could* have been disturbed was run, since
    > the branch edited a page it reads: `uv run pytest tests/test_api_docs.py -q`, 11
    > passed.
  - [x] Say in the master plan's audit subgoal what this branch already settled.
    > **Note:** said in two places, because they answer different questions. The subgoal's
    > own sentence no longer presumes a surface — "whatever the search loop and training
    > log export" now says they export nothing and cites ADR 0025 — and a second
    > `> **Note:**` beneath it lists the three parts of the audit this branch discharged,
    > so the auditor learns what to *stop* checking in the place they will be reading. An
    > audit that re-verifies what was already verified costs the same as one that never
    > knew, so the whole saving is in where the note sits.
    > **Note:** `docs/plan/TODO.md:225`, the export subgoal itself, is deliberately left
    > `[ ]` with its `> **Branch:**` backlink intact. `/smart-merge` step 7 marks it
    > complete and writes its `> **Done:**` with the PR number, which is the only moment
    > when both the PR number and the open branch exist. Closing it here would leave the
    > merge nothing to record.
