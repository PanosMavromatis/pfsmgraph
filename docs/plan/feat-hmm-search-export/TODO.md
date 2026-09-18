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

- [ ] Decide, and record the decision where the next revision will find it
  - [ ] Choose the venue: an ADR, or a section in an existing document. An ADR only if
        the decision has alternatives worth arguing rather than a single reason.
  - [ ] Write the decision with its reason, in the terms the plan asked for.

- [ ] Carry out whatever the decision requires
  - [ ] If export: names, `__all__`, `docs/api/hmm/search.md` under ADR 0013 with every
        block executed and its output pasted, `docs/api/hmm/README.md`'s public-surface
        table, `packages/pfsmgraph-hmm/README.md`.
  - [ ] If not: close `docs/api/hmm/baum_welch.md:160`'s pointer and the audit subgoal's
        "whatever the search loop and training log export", so neither is left asserting
        a surface that does not exist.

- [ ] Sync the docs and leave the audit a smaller job than it was
  - [ ] `docs/agents/core.md` and `AGENTS.md`, including test counts if any moved.
  - [ ] Say in the master plan's audit subgoal what this branch already settled.
