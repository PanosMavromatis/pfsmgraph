# docs/hmm-migration-plan

**Status**: active
**Created**: 2026-09-03
**Subgoal**: standalone — revision 01 is closed and no revision is open; opening revision 02 is this branch's own goal 4

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

The goals are ordered as a dependency chain, not by preference. Goal 1 exists because the
HMM library has never been read — `ACCOUNT.md`, `COMPARISON.md` and `translation/` are all
scoped to `Code/SeqData/`, the container half that fed `dataseq`. Goals 2 and 3 are
decisions that cannot honestly be taken before that reading, and goal 4 is the only one
that writes anything outside `.scratch/`. Nothing under `packages/pfsmgraph-hmm/src/` is
touched on this branch.

## Goals

- [ ] Read `Code/HMMlib/` in its own terms and write the account
  - [ ] `hmm.lsh` and `hmm-param.lsh` — the model object and how its parameters are held, named and initialised
  - [ ] `hmm-trainer.lsh` — Baum-Welch, and whatever topology search is interleaved with it; 1102 lines is over half the library
  - [ ] Measure every quantitative claim against the two tracked specimen corpora, as `ACCOUNT.md` did, and collect the measurements in an appendix
  - [ ] Mark behaviours the code admits but may never have exercised as **provenance unknown**, rather than asserting them as bugs that bit
  - [ ] Write it to `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, keeping it free of any reference to the target design — the comparison is a separate document, for the same reason `ACCOUNT.md` and `COMPARISON.md` are separate

- [ ] Decide the translation strategy and the shape of `pfsmgraph.hmm`
  - [ ] Name the public surface: what a caller constructs, trains, and reads back
  - [ ] Fix where `hmm` meets `dataseq` — which of `SymbolTable`, the record container and `pad_collate` it consumes, and whether a ragged trainer wants padding at all
  - [ ] Apply *encode at the boundary* to a trainer specifically: the inner loops are integer-only by invariant, so state the entry and exit points where strings are still permitted
  - [ ] Decide whether 0.1.0 is ADR 0002 phase 1 only — and so whether this migration **fires or defers** `DEFERRED.md`'s `## Trigger: the first .pyx`, with it the ADR 0012 hatchling revert and the meson-python namespace problem
  - [ ] Follow that through to ADR 0003: a phase-1 Baum-Welch is the first DP kernel in the repository, so the backend matrix stops being empty and the header prints real content for the first time
  - [ ] Log each answer inline

- [ ] Settle the scope questions the source forces
  - [ ] `hmm-trainer-view.lsh` (237 lines) — in the distribution, out of it, or properly `dl`'s; decide on what it actually does, not on the filename
  - [ ] Topology search by state merge/split — in 0.1.0, or a later minor; the architecture table names it as `hmm`'s role, which is an argument but not a schedule
  - [ ] The `.sds` corpus directory format — does it come across, or die with the Lush tree; `dataseq` already owns the container, so adopting it needs a reason beyond the specimens being in it
  - [ ] For each, record the *negative* findings too: a scope question answered "out" is a decision, and reads as an oversight if only the inclusions are written down

- [ ] Record the outcome so the next branch can execute it
  - [ ] Write the ADR(s) the decisions warrant, starting at 0015, with a row added to `docs/design/adr/README.md`; prefer an in-place amendment where a decision only specifies an existing ADR further, per the precedent set on ADR 0003
  - [ ] Add a `## Trigger: the hmm migration` section to `DEFERRED.md` for anything decided here that must land *as part of* the migration rather than after it
  - [ ] Open revision 02 in `docs/plan/TODO.md` with its subgoals, each one a branch a later `/new-branch` can take
  - [ ] Sweep `docs/agents/core.md` for the claims this branch falsifies — the "Still to do" line, and the `.scratch/` paragraph's account of what has been read
